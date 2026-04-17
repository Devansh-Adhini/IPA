'''
  DALF v2 - Improved Architecture for Deformation-Aware Local Feature Extraction
  ================================================================================
  Improvements over the Vanilla DALF (CVPR 2023):

    1. CBAM Attention (Woo et al. 2018) in every encoder block
       - Channel attention: selects which feature channels matter
       - Spatial attention: selects which spatial locations matter
       Both proven to improve feature quality with minimal overhead.

    2. FPN-style Decoder (Lin et al. 2017) 
       - Lateral 1x1 projections at each scale
       - Top-down 3x3 refinement after upsampling
       - Multi-scale semantic richness flows through the heatmap prediction

    3. Deeper Fusion Layer
       - LayerNorm instead of BatchNorm for stability
       - GELU activations (smoother gradients than ReLU)
       - 128 -> 256 -> 128 bottleneck instead of shallow 128->128

    4. Coordinate-Enhanced ThinPlateNet
       - Appends normalised (x,y) coordinate channels before FCN
       - TPS attention can use explicit position info to compensate for
         viewpoint-dependent sampling errors

  Interface is identical to Vanilla DEAL for drop-in compatibility with:
    - train.py
    - test_accuracy.py
    - evaluate_on_landmarks.py
'''

import torch
from torch import nn
import torch.nn.functional as F
import math
import numpy as np
import cv2
import os
import time

from modules.tps import pytorch as TPS


# =============================================================================
# CBAM: Convolutional Block Attention Module (Woo et al. 2018)
# =============================================================================

class ChannelAttention(nn.Module):
    '''
    Squeeze-and-excitation style channel attention.
    Learns which feature channels are most discriminative.
    '''
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(),
            nn.Linear(mid, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        avg = self.mlp(self.avg_pool(x).view(B, C))
        mx  = self.mlp(self.max_pool(x).view(B, C))
        attn = self.sigmoid(avg + mx).view(B, C, 1, 1)
        return x * attn


class SpatialAttention(nn.Module):
    '''
    Spatial attention: highlights which positions carry the most information.
    '''
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size,
                              padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = torch.mean(x, dim=1, keepdim=True)
        mx  = torch.max(x,  dim=1, keepdim=True)[0]
        pool = torch.cat([avg, mx], dim=1)
        attn = self.sigmoid(self.conv(pool))
        return x * attn


class CBAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 8, spatial_ks: int = 7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(spatial_ks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sa(self.ca(x))


# =============================================================================
# Encoder with CBAM attention after each DownBlock
# =============================================================================

class DownBlockV2(nn.Module):
    '''
    Two conv layers + optional residual shortcut + CBAM + MaxPool.
    '''
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch, affine=False),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch, affine=False),
            nn.ReLU(),
        )
        # Skip connection when channels differ
        self.skip = (nn.Conv2d(in_ch, out_ch, 1, bias=False)
                     if in_ch != out_ch else nn.Identity())
        self.cbam = CBAM(out_ch)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)                  # match channel dim
        # Downsample residual to same spatial size as after pool
        out = self.conv1(x)
        out = self.conv2(out) + residual          # residual before CBAM
        out = self.cbam(out)
        return self.pool(out)


class EncoderV2(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.norm = nn.InstanceNorm2d(channels[0], affine=False)
        self.blocks = nn.ModuleList(
            [DownBlockV2(channels[i], channels[i + 1])
             for i in range(len(channels) - 1)]
        )

    def forward(self, x: torch.Tensor):
        x = self.norm(x)
        features = [x]
        for b in self.blocks:
            x = b(x)
            features.append(x)
        return features


# =============================================================================
# FPN Decoder (lateral connections + top-down refinement)
# =============================================================================

class FPNDecoder(nn.Module):
    '''
    Feature Pyramid Network style decoder.
    Each skip level gets a 1x1 lateral projection.
    After upsampling + add, a 3x3 conv refines the features.
    '''
    def __init__(self, enc_channels):
        super().__init__()
        # enc_channels = [1, 32, 64, 64]
        # We work top-down so reverse: [64, 64, 32, 1]
        ch = enc_channels[::-1]               # top -> bottom

        # Lateral 1x1 projections: project each skip level to ch[0]
        fpn_ch = ch[0]
        self.laterals = nn.ModuleList([
            nn.Conv2d(c, fpn_ch, 1, bias=False)
            for c in ch
        ])

        # Refinement 3x3 convs after merge
        self.refines = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(fpn_ch, fpn_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(fpn_ch, affine=False),
                nn.ReLU(),
            )
            for _ in range(len(ch) - 1)
        ])

        # Heatmap head
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(fpn_ch, fpn_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_ch, affine=False),
            nn.ReLU(),
            nn.Conv2d(fpn_ch, 1, 1),
        )

    def forward(self, features):
        # features: [x_in, feat_enc1, feat_enc2, feat_enc3]
        enc = features[::-1]   # top-level first

        # Start with top-level lateral
        x = self.laterals[0](enc[0])

        for i in range(1, len(enc)):
            # Upsample and add lateral
            x = F.interpolate(x, size=enc[i].shape[-2:],
                              mode='bilinear', align_corners=True)
            x = x + self.laterals[i](enc[i])
            x = self.refines[i - 1](x)

        return self.heatmap_head(x)


# =============================================================================
# Coordinate-enhanced ThinPlateNet
# =============================================================================

class ThinPlateNetV2(nn.Module):
    '''
    Identical to vanilla ThinPlateNet but with positional coordinate channels
    appended to the FCN input so the TPS corrector can reason about image
    position explicitly.
    '''
    def __init__(self, in_channels, nchannels=1, ctrlpts=(8, 8),
                 fixed_tps=False):
        super().__init__()
        self.ctrlpts = ctrlpts
        self.nctrl = ctrlpts[0] * ctrlpts[1]
        self.nparam = (self.nctrl + 2)
        self.interpolator = InterpolateSparse2d(mode='bilinear')
        self.fixed_tps = fixed_tps

        # +2 for normalised (x, y) coordinate channels
        self.fcn = nn.Sequential(
            nn.Conv2d(in_channels + 2, in_channels * 2, 3, padding=1,
                      stride=1, bias=False),
            nn.BatchNorm2d(in_channels * 2, affine=False),
            nn.ReLU(),
            nn.Conv2d(in_channels * 2, in_channels * 2, 3, bias=False),
            nn.BatchNorm2d(in_channels * 2, affine=False),
            nn.ReLU(),
        )

        self.attn = nn.Sequential(
            nn.Linear(in_channels * 2, in_channels * 4),
            nn.BatchNorm1d(in_channels * 4, affine=False),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(in_channels * 4, in_channels * 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(in_channels * 4, self.nparam * 2),
            nn.Tanh(),
        )
        # zero-init for identity TPS at start
        for i in [-2, -5, -9]:
            self.attn[i].weight.data.normal_(0., 1e-5)
            self.attn[i].bias.data.zero_()

    def _coord_channels(self, feat: torch.Tensor) -> torch.Tensor:
        '''Append normalised (x,y) coordinate channels to feature map.'''
        B, C, H, W = feat.shape
        ys = torch.linspace(-1, 1, H, device=feat.device)
        xs = torch.linspace(-1, 1, W, device=feat.device)
        grid_y, grid_x = torch.meshgrid(ys, xs)
        coords = torch.stack([grid_x, grid_y], dim=0)  # (2, H, W)
        coords = coords.unsqueeze(0).expand(B, -1, -1, -1)
        return torch.cat([feat, coords], dim=1)          # (B, C+2, H, W)

    def get_polar_grid(self, keypts, Hs, Ws, coords='linear',
                       gridSize=(32, 32), maxR=32.):
        maxR_t = torch.ones_like(keypts[:, 0]) * maxR
        batchSize = keypts.shape[0]
        ident = torch.tensor(
            [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]],
            device=keypts.device).expand(batchSize, -1, -1)
        grid = F.affine_grid(ident, (batchSize, 1) + gridSize,
                             align_corners=False)
        grid_y = grid[..., 0].view(batchSize, -1)
        grid_x = grid[..., 1].view(batchSize, -1)
        maxR_t = maxR_t.unsqueeze(-1).expand(-1, grid_y.shape[-1]).float()
        normGrid = (grid_y + 1) / 2
        if coords == 'log':
            r_s_ = torch.exp(normGrid * torch.log(maxR_t))
        else:
            r_s_ = 1 + normGrid * (maxR_t - 1)
        r_s = (r_s_ - 1) / (maxR_t - 1) * 2 * maxR_t / Ws
        t_s = (grid_x + 1) * np.pi
        x_coord = keypts[:, 0].unsqueeze(-1).expand(-1, grid_x.shape[-1]) / Ws * 2. - 1.
        y_coord = keypts[:, 1].unsqueeze(-1).expand(-1, grid_y.shape[-1]) / Hs * 2. - 1.
        aR = Ws / Hs
        x_s = r_s * torch.cos(t_s) + x_coord
        y_s = r_s * torch.sin(t_s) * aR + y_coord
        return torch.cat((x_s.view(batchSize, gridSize[0], gridSize[1], 1),
                          y_s.view(batchSize, gridSize[0], gridSize[1], 1)),
                         -1)

    def forward(self, x, in_imgs, keypts, Ho, Wo):
        B = len(keypts)
        patches = []

        # Append coordinate channels to feature map once per batch image
        x_coord = self._coord_channels(x)                 # (B, C+2, H, W)
        # Run FCN on coord-enhanced feature map once per image
        Theta = self.fcn(x_coord)                         # (B, C*2, H', W')

        for b in range(B):
            kps = keypts[b]['xy']
            if kps is None or len(kps) == 0:
                patches.append(None)
                continue

            Ni = len(kps)
            chunk_size = 512
            Hi, Wi = 32, 32
            curr_patches_list = []
            kfactor, offset = 0.3, (1.0 - 0.3) / 2.

            # Interpolate FCN features at keypoint positions, then run attn
            all_theta = self.interpolator(Theta[b], kps.float(), Ho, Wo)
            all_theta = self.attn(all_theta).view(-1, self.nparam, 2)

            for start in range(0, Ni, chunk_size):
                end = min(start + chunk_size, Ni)
                chunk_kps = kps[start:end].float()
                chunk_theta = all_theta[start:end]
                Nc = len(chunk_kps)

                polargrid_i = self.get_polar_grid(chunk_kps, Ho, Wo,
                                                  gridSize=(Hi, Wi))
                vmin = polargrid_i.view(Nc, -1, 2).min(1)[0].unsqueeze(1).unsqueeze(1)
                vmax = polargrid_i.view(Nc, -1, 2).max(1)[0].unsqueeze(1).unsqueeze(1)
                ptp = vmax - vmin
                polargrid_i = (polargrid_i - vmin) / (ptp + 1e-8)
                polargrid_i = polargrid_i * kfactor + offset

                grid_img_i = polargrid_i.permute(0, 3, 1, 2)
                ctrl_i = F.interpolate(grid_img_i, self.ctrlpts
                                       ).permute(0, 2, 3, 1).view(Nc, -1, 2)

                I_polargrid_i = chunk_theta.new(Nc, Hi, Wi, 3)
                I_polargrid_i[..., 0] = 1.0
                I_polargrid_i[..., 1:] = polargrid_i

                if not self.fixed_tps:
                    z_i = TPS.tps(chunk_theta, ctrl_i, I_polargrid_i)
                    tps_warper_i = I_polargrid_i[..., 1:] + z_i
                else:
                    tps_warper_i = polargrid_i

                tps_warper_i = (tps_warper_i - offset) / kfactor
                tps_warper_i = tps_warper_i * ptp + vmin

                p_chunk = F.grid_sample(
                    in_imgs[b].expand(Nc, -1, -1, -1),
                    tps_warper_i, align_corners=False, padding_mode='zeros')
                curr_patches_list.append(p_chunk)

            patches.append(torch.cat(curr_patches_list, dim=0))
        return patches


# =============================================================================
# Improved HardNet with depth-wise separable convs + residual
# =============================================================================

class Pad2D(torch.nn.Module):
    def __init__(self, pad, mode):
        super().__init__()
        self.pad = pad
        self.mode = mode

    def forward(self, x):
        return F.pad(x, pad=self.pad, mode=self.mode)


class HardNetV2(nn.Module):
    '''
    Rotation-invariant polar patch descriptor matching vanilla HardNet spatial
    reduction pattern (strided convs reduce spatial dims from 32x32 to 8x8,
    AvgPool then collapses height to 1, final convs reduce width to 1).
    Improvements: extra conv layers per stage for richer representation.
    '''
    def __init__(self, nchannels=1, out_ch=64):
        super().__init__()
        self.nchannels = nchannels
        self.features = nn.Sequential(
            nn.InstanceNorm2d(self.nchannels),
            # Stage 1: 32x32 -> 32x32
            Pad2D(pad=(0, 0, 1, 1), mode='circular'),
            nn.Conv2d(self.nchannels, 32, 3, bias=False, padding=(0, 1)),
            nn.BatchNorm2d(32, affine=False), nn.ReLU(),
            Pad2D(pad=(0, 0, 1, 1), mode='circular'),
            nn.Conv2d(32, 32, 3, bias=False, padding=(0, 1)),
            nn.BatchNorm2d(32, affine=False), nn.ReLU(),
            # Strided -> 32x32 to 16x16
            Pad2D(pad=(0, 0, 1, 1), mode='circular'),
            nn.Conv2d(32, 64, 3, bias=False, padding=(0, 1), stride=2),
            nn.BatchNorm2d(64, affine=False), nn.ReLU(),
            Pad2D(pad=(0, 0, 1, 1), mode='circular'),
            nn.Conv2d(64, 64, 3, bias=False, padding=(0, 1)),
            nn.BatchNorm2d(64, affine=False), nn.ReLU(),
            # Strided -> 16x16 to 8x8
            Pad2D(pad=(0, 0, 1, 1), mode='circular'),
            nn.Conv2d(64, 64, 3, bias=False, padding=(0, 1), stride=2),
            nn.BatchNorm2d(64, affine=False), nn.ReLU(),
            Pad2D(pad=(0, 0, 1, 1), mode='circular'),
            nn.Conv2d(64, 64, 3, bias=False, padding=(0, 1)),
            nn.BatchNorm2d(64, affine=False), nn.ReLU(),
            nn.Dropout(0.2),
            # Rotation invariance block: pool height (angle axis) to 1
            nn.AvgPool2d((8, 1), stride=1),
            # Width reduction: 8 -> 6 -> 4 -> 2 -> 1
            nn.Conv2d(64, out_ch, (1, 3), bias=False),
            nn.BatchNorm2d(out_ch, affine=False), nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, (1, 3), bias=False),
            nn.BatchNorm2d(out_ch, affine=False), nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, (1, 3), bias=False),
            nn.BatchNorm2d(out_ch, affine=False), nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, (1, 2), bias=False),
            nn.BatchNorm2d(out_ch, affine=False),
        )

    def forward(self, x):
        if x is not None:
            x = self.features(x).squeeze(-1).squeeze(-1)
        return x


# =============================================================================
# Reuse vanilla interpolator and keypoint sampler unchanged
# =============================================================================

class InterpolateSparse2d(nn.Module):
    def __init__(self, mode='bicubic'):
        super().__init__()
        self.mode = mode

    def normgrid(self, x, H, W):
        return (2. * (x / torch.tensor([W - 1, H - 1],
                                       device=x.device, dtype=x.dtype)) - 1.)

    def forward(self, x, pos, H, W):
        grid = self.normgrid(pos, H, W).unsqueeze(0).unsqueeze(-2)
        x = F.grid_sample(x.unsqueeze(0), grid,
                           mode=self.mode, align_corners=True)
        return x.permute(0, 2, 3, 1).squeeze(0).squeeze(-2)


class KeypointSampler(nn.Module):
    def __init__(self, window_size=8):
        super().__init__()
        self.window_size = window_size

    def gridify(self, x):
        B, C, H, W = x.shape
        return (x.unfold(2, self.window_size, self.window_size)
                 .unfold(3, self.window_size, self.window_size)
                 .reshape(B, C,
                           H // self.window_size,
                           W // self.window_size,
                           self.window_size ** 2))

    def sample(self, grid):
        chooser = torch.distributions.Categorical(logits=grid)
        choices = chooser.sample()
        selected = torch.gather(grid, -1, choices.unsqueeze(-1)).squeeze(-1)
        flipper = torch.distributions.Bernoulli(logits=selected)
        accepted = flipper.sample()
        log_probs = chooser.log_prob(choices) + flipper.log_prob(accepted)
        return log_probs.squeeze(1), choices, accepted.gt(0).squeeze(1)

    def forward(self, x):
        B, C, H, W = x.shape
        kp_cells = self.gridify(x)
        idx_cells = self.gridify(
            torch.dstack(torch.meshgrid(
                torch.arange(H, dtype=torch.float32),
                torch.arange(W, dtype=torch.float32),
            )).permute(2, 0, 1).unsqueeze(0).expand(B, -1, -1, -1).to(x.device))

        log_probs, idx, mask = self.sample(kp_cells)
        keypoints = torch.gather(
            idx_cells, -1, idx.repeat(1, 2, 1, 1).unsqueeze(-1)
        ).squeeze(-1).permute(0, 2, 3, 1)

        return [{'xy': keypoints[b][mask[b]].flip(-1),
                 'logprobs': log_probs[b][mask[b]]}
                for b in range(B)]


# =============================================================================
# UNet V2: EncoderV2 (CBAM) + FPN Decoder
# =============================================================================

class UNetV2(nn.Module):
    def __init__(self, enc_channels=[1, 32, 64, 64]):
        super().__init__()
        self.encoder = EncoderV2(enc_channels)
        self.decoder = FPNDecoder(enc_channels)
        self.nchannels = enc_channels[0]
        self.features = nn.Sequential(
            nn.Conv2d(enc_channels[-1], enc_channels[-1], 3,
                      padding=1, bias=False),
            nn.BatchNorm2d(enc_channels[-1], affine=False),
            nn.ReLU(),
            nn.Conv2d(enc_channels[-1], enc_channels[-1], 1),
            nn.BatchNorm2d(enc_channels[-1], affine=False),
        )

    def forward(self, x):
        if self.nchannels == 1 and x.shape[1] != 1:
            x = torch.mean(x, dim=1, keepdim=True)
        feats = self.encoder(x)
        heatmap = self.decoder(feats)
        feat = self.features(feats[-1])
        return {'map': heatmap, 'feat': feat}


# =============================================================================
# DEAL V2 - Same interface as vanilla DEAL
# =============================================================================

class DEAL(nn.Module):
    '''
    DALF v2 feature extractor.
    Drop-in replacement for vanilla DEAL with identical forward signature.
    '''
    def __init__(self, enc_channels=[1, 32, 64, 64],
                 fixed_tps=False, mode=None):
        super().__init__()
        self.net = UNetV2(enc_channels)
        self.detector = KeypointSampler()
        self.interpolator = InterpolateSparse2d()

        hn_out_ch = 128 if mode == 'end2end-tps' else 64

        print('backbone: %d hardnet: %d' % (enc_channels[-1], hn_out_ch))
        self.tps_net = ThinPlateNetV2(
            in_channels=enc_channels[-1],
            nchannels=enc_channels[0],
            fixed_tps=fixed_tps)
        self.hardnet = HardNetV2(nchannels=enc_channels[0], out_ch=hn_out_ch)

        self.nchannels = enc_channels[0]
        self.enc_channels = enc_channels
        self.mode = mode

        if self.mode == 'ts-fl':
            print('adding improved fusion layer...')
            # Deeper fusion: 128 -> 256 -> 128 with LayerNorm + GELU
            self.fusion_layer = nn.Sequential(
                nn.Linear(128, 256),
                nn.LayerNorm(256),
                nn.GELU(),
                nn.Linear(256, 128),
                nn.LayerNorm(128),
                nn.GELU(),
                nn.Linear(128, 128),
                nn.Sigmoid(),
            )

    # ---- Unchanged utility methods ----

    def NMS(self, x, threshold=3., kernel_size=3):
        pad = kernel_size // 2
        local_max = nn.MaxPool2d(kernel_size=kernel_size, stride=1,
                                 padding=pad)(x)
        pos = (x == local_max) & (x > threshold)
        return pos.nonzero()[..., 1:].flip(-1)

    def sample_descs(self, feature_map, kpts, H, W):
        return self.interpolator(feature_map, kpts, H, W).contiguous()

    def forward(self, x, NMS=False, threshold=3.,
                return_tensors=False, top_k=None):

        if self.nchannels == 1 and x.shape[1] != 1:
            x = torch.mean(x, dim=1, keepdim=True)

        B, C, H, W = x.shape
        out = self.net(x)

        if not NMS:
            kpts = self.detector(out['map'])
        else:
            kpts = [{'xy': self.NMS(out['map'][b], threshold)}
                    for b in range(B)]

        # Border filter during training
        if not NMS:
            for b in range(B):
                fx = kpts[b]['xy'][:, 0] >= 16
                fy = kpts[b]['xy'][:, 1] >= 16
                fx2 = kpts[b]['xy'][:, 0] < W - 16
                fy2 = kpts[b]['xy'][:, 1] < H - 16
                f = fx & fy & fx2 & fy2
                kpts[b]['xy'] = kpts[b]['xy'][f]
                if 'logprobs' in kpts[b]:
                    kpts[b]['logprobs'] = kpts[b]['logprobs'][f]

        if top_k is not None:
            for b in range(B):
                scores = out['map'][b].squeeze(0)[
                    kpts[b]['xy'][:, 1].long(),
                    kpts[b]['xy'][:, 0].long()]
                idx = torch.argsort(-scores)
                kpts[b]['xy'] = kpts[b]['xy'][idx[:top_k]]
                if 'logprobs' in kpts[b]:
                    kpts[b]['logprobs'] = kpts[b]['logprobs'][idx[:top_k]]

        optimize_tps = self.mode in ('ts2', 'end2end-tps',
                                     'end2end-full', 'ts-fl')
        if optimize_tps:
            patches = self.tps_net(out['feat'], x, kpts, H, W)

        for b in range(B):
            if optimize_tps:
                kpts[b]['patches'] = patches[b]
            else:
                with torch.no_grad():
                    kpts[b]['patches'] = torch.zeros(
                        len(kpts[b]['xy']), 1, 32, 32).to(x.device)

        descs = []
        if NMS:
            for b in range(B):
                if len(kpts[b]['xy']) == 0 or kpts[b].get('patches') is None:
                    descs.append(None)
                    continue

                if self.mode in ('end2end-full', 'ts2', 'ts-fl'):
                    hn_out = self.hardnet(kpts[b]['patches'])   # (N, 64)
                    bb_out = self.interpolator(out['feat'][b],
                                              kpts[b]['xy'], H, W)  # (N, 64)
                    if self.mode == 'ts-fl':
                        fd = torch.cat((hn_out, bb_out), dim=1)  # (N, 128)
                        fd = self.fusion_layer(fd) * fd
                    else:
                        fd = torch.cat((hn_out, bb_out), dim=1)
                elif self.mode in ('end2end-backbone', 'ts1'):
                    fd = self.interpolator(out['feat'][b],
                                           kpts[b]['xy'], H, W)
                else:
                    fd = self.hardnet(kpts[b]['patches'])

                descs.append(F.normalize(fd))

        if not NMS:
            if not return_tensors:
                return kpts
            else:
                return kpts, out
        else:
            if not return_tensors:
                return kpts, descs
            else:
                return kpts, descs, out


# =============================================================================
# DALF_extractor wrapper (identical interface to vanilla)
# =============================================================================

class DALF_extractor:
    def __init__(self, model=None, dev=torch.device('cpu'), fixed_tps=False):
        self.dev = dev
        print('running DALF v2 on', self.dev)

        if model is None:
            abs_path = os.path.dirname(os.path.abspath(__file__))
            model = abs_path + '/../../weights/model_ts-fl_final.pth'

        backbone_nfeats = 128 if 'end2end-backbone' in model else 64
        modes = ['end2end-backbone', 'end2end-tps', 'end2end-full',
                 'ts1', 'ts2', 'ts-fl']
        mode = None
        for m in modes:
            if m in model:
                mode = m
        if mode is None:
            raise RuntimeError(
                'Could not parse network mode from file name')

        self.net = DEAL(enc_channels=[1, 32, 64, backbone_nfeats],
                        fixed_tps=fixed_tps, mode=mode).to(dev)
        self.net.load_state_dict(torch.load(model, map_location=dev))
        self.net.eval().to(dev)

    def detectAndCompute(self, og_img, mask=None, top_k=2048,
                         return_map=False, threshold=25., MS=False):
        if isinstance(og_img, str):
            og_img = cv2.cvtColor(cv2.imread(og_img), cv2.COLOR_BGR2RGB)

        scales = [1/6, 1/4, 1/2, 1] if MS else [1]
        kpts_list, descs_list, scores_list = [], [], []
        hd_map = None

        for scale in scales:
            with torch.no_grad():
                img = (cv2.resize(og_img, None, fx=scale, fy=scale,
                                  interpolation=cv2.INTER_AREA)
                       if scale != 1. else og_img)
                img = (torch.tensor(img, dtype=torch.float32,
                                    device=self.dev)
                       .permute(2, 0, 1).unsqueeze(0) / 255.)
                kpts, descs, fmap = self.net(
                    img, NMS=True, threshold=threshold,
                    return_tensors=True, top_k=top_k)
                score_map = fmap['map'][0].squeeze(0).cpu().numpy()
                kpts_np = kpts[0]['xy'].cpu().numpy().astype(np.int16)
                descs_np = descs[0].cpu().numpy()
                scores = score_map[kpts_np[:, 1], kpts_np[:, 0]]
                scores /= score_map.max()
                si = np.argsort(-scores)
                kpts_np = kpts_np[si] / scale
                descs_np = descs_np[si]
                scores = scores[si]
                kpts_list.append(kpts_np)
                descs_list.append(descs_np)
                scores_list.append(scores)

        if len(scales) > 1:
            ppk = top_k // len(scales)
            all_k = np.vstack([k[:ppk] for k in kpts_list[:-1]])
            all_d = np.vstack([d[:ppk] for d in descs_list[:-1]])
            all_s = np.hstack([s[:ppk] for s in scores_list[:-1]])
            all_k = np.vstack([all_k, kpts_list[-1][:top_k - len(all_k)]])
            all_d = np.vstack([all_d, descs_list[-1][:top_k - len(all_d)]])
            all_s = np.hstack([all_s, scores_list[-1][:top_k - len(all_s)]])
        else:
            all_k, all_d, all_s = kpts_list[0], descs_list[0], scores_list[0]

        cv_kps = [cv2.KeyPoint(all_k[i][0], all_k[i][1], 6, 0, all_s[i])
                  for i in range(len(all_k))]
        return (cv_kps, all_d, hd_map) if return_map else (cv_kps, all_d)

    def detect(self, img, _=None):
        return self.detectAndCompute(img)[0]
