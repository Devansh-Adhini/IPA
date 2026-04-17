"""
geometric_transform.py
-----------------------
Apply random geometric transformations to any image and visualise results.

Supports three modes:
  homography  – random perspective / affine / rotation warp (frame-filling)
  tps         – thin-plate spline warp  (smooth organic curves, Kanagawa-style)
  both        – homography followed by TPS

The output always fills the entire frame (no black voids).

Usage examples
--------------
    # TPS warp (Kanagawa-wave style) – default output fills the frame
    python geometric_transform.py -i wave.jpeg -o out.jpg --mode tps --difficulty 0.5

    # Homography only
    python geometric_transform.py -i test.jpg -o out.jpg --mode homography

    # Both, with stronger deformation
    python geometric_transform.py -i test.jpg -o out.jpg --mode both --difficulty 0.45

    # Add colour augmentation on top, save side-by-side comparison
    python geometric_transform.py -i test.jpg -o out.jpg --mode tps \\
        --difficulty 0.6 --color-aug --side-by-side --seed 7
"""

import argparse
import os
import sys
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import kornia
from kornia.geometry.transform import get_tps_transform as findTPS
from kornia.geometry.transform import warp_image_tps

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from modules.dataset.augmentation import generateRandomTPS

# ══════════════════════════════════════════════════════════════════════════════
# Homography generator
# Replicates generateRandomHomography but with the translation constrained so
# that the image stays within the frame (no large black borders).
# ══════════════════════════════════════════════════════════════════════════════

def make_homography(shape, difficulty=0.3):
    """
    Build a random homography that keeps the image filling the output frame.

    Strategy
    --------
    - Rotate / shear / scale around the image centre.
    - Keep translation well within the image boundaries.
    - Normalise the final matrix so the image corners stay inside the frame.
    """
    h, w = shape

    # ── Random parameters (scaled by difficulty) ─────────────────────────
    theta    = np.radians(np.random.uniform(-25 * difficulty, 25 * difficulty))
    # isotropic scale: 1/(1+d) .. (1+d)
    log_s    = np.random.uniform(-0.4 * difficulty, 0.4 * difficulty)
    scale    = np.exp(log_s)
    # aspect ratio perturbation
    log_ar   = np.random.uniform(-0.3 * difficulty, 0.3 * difficulty)
    scale_r  = np.exp(log_ar)
    # shear
    sx, sy   = np.random.uniform(-0.3 * difficulty, 0.3 * difficulty, 2)
    # projective (small)
    p1, p2   = np.random.uniform(-0.004 * difficulty, 0.004 * difficulty, 2)
    # translation – bounded to ±20% of image dimension
    max_tx   = 0.20 * w * difficulty
    max_ty   = 0.20 * h * difficulty
    txn, tyn = np.random.uniform(-max_tx, max_tx), np.random.uniform(-max_ty, max_ty)

    cx, cy   = w / 2.0, h / 2.0
    c, s     = np.cos(theta), np.sin(theta)

    # Build around image centre so rotation stays inside frame
    T_to_origin = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], dtype=np.float64)
    T_back      = np.array([[1, 0,  cx + txn], [0, 1, cy + tyn], [0, 0, 1]], dtype=np.float64)

    H_rot   = np.array([[c, -s, 0], [s,  c, 0], [0, 0, 1]], dtype=np.float64)
    H_scale = np.array([[scale, 0, 0], [0, scale * scale_r, 0], [0, 0, 1]], dtype=np.float64)
    H_shear = np.array([[1, sy, 0], [sx, 1, 0], [0, 0, 1]], dtype=np.float64)
    H_proj  = np.array([[1, 0, 0], [0, 1, 0], [p1, p2, 1]], dtype=np.float64)

    H = T_back @ H_scale @ H_proj @ H_shear @ H_rot @ T_to_origin
    return H


def _warp_corners(H, w, h):
    """Warp the 4 image corners through H, return their 2-D coords."""
    corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], dtype=np.float64).T
    warped  = H @ corners
    warped /= warped[2:3]
    return warped[:2].T  # (4, 2)


def centre_homography(H, w, h):
    """
    Add a corrective translation so that all 4 warped corners remain
    inside a [0,w]x[0,h] canvas (no cropping of image content).
    """
    pts = _warp_corners(H, w, h)
    min_x, min_y = pts[:, 0].min(), pts[:, 1].min()
    max_x, max_y = pts[:, 0].max(), pts[:, 1].max()

    # Scale to fit inside the canvas if the warped region is larger
    scale_fit = min(w / (max_x - min_x), h / (max_y - min_y))
    if scale_fit < 1.0:
        S = np.array([[scale_fit, 0, 0], [0, scale_fit, 0], [0, 0, 1]], dtype=np.float64)
        H = S @ H
        pts = _warp_corners(H, w, h)
        min_x, min_y = pts[:, 0].min(), pts[:, 1].min()

    # Shift to put top-left corner at (0,0)
    T_corr = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]], dtype=np.float64)
    return T_corr @ H


# ══════════════════════════════════════════════════════════════════════════════
# Colour augmentation (identical to AugmentationPipe.aug_list)
# ══════════════════════════════════════════════════════════════════════════════

def build_color_aug():
    return kornia.augmentation.ImageSequential(
        kornia.augmentation.RandomChannelShuffle(p=0.5),
        kornia.augmentation.ColorJitter(0.2, 0.2, 0.2, 0.2, p=1.0),
        kornia.augmentation.RandomEqualize(p=0.5),
        kornia.augmentation.RandomGrayscale(p=0.2),
        kornia.augmentation.RandomPosterize(bits=4, p=0.2),
        kornia.augmentation.RandomGaussianBlur(p=0.3, sigma=(2.5, 2.5), kernel_size=(7, 7)),
    )


# ══════════════════════════════════════════════════════════════════════════════
# I/O helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_tensor(path, warp_size, device):
    """BGR disk image → (1,3,H,W) float32 [0,1] on device."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {path}")
    img = cv2.resize(img, (warp_size[0], warp_size[1]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    t   = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    return t.to(device)


def to_bgr_uint8(t):
    """(1,3,H,W) float [0,1] → HWC BGR uint8."""
    arr = t.squeeze(0).permute(1, 2, 0).cpu().numpy()
    arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


# ══════════════════════════════════════════════════════════════════════════════
# Main transform
# ══════════════════════════════════════════════════════════════════════════════

def apply_transform(img_t, mode, difficulty, tps_prob, tps_grid, color_aug, device, out_size):
    """
    Apply geometric + optional colour augmentation.

    Key changes vs training AugmentationPipe
    ----------------------------------------
    * padding_mode = 'border'  → no black voids; edges are replicated.
    * NO 20 % border crop      → image always fills the full canvas.
    * Homography is centred     → warped content stays inside the frame.
    * TPS is applied directly on the full image.

    Returns
    -------
    out_t     (1,3,H,W)  transformed image tensor [0,1]
    H_matrix  (3,3) numpy array or None
    tps_info  dict or None
    """
    with torch.no_grad():
        x = img_t.clone()
        shape = x.shape[-2:]   # (H, W)
        h, w  = shape

        # ── Colour augmentation (optional) ───────────────────────────────
        if color_aug:
            aug = build_color_aug().to(device)
            x   = aug(x)
            if np.random.uniform() > 0.5:
                noise = F.interpolate(torch.randn_like(x) * (16 / 255), (h // 2, w // 2))
                noise = F.interpolate(noise, (h, w), mode='bicubic', align_corners=False)
                x = torch.clip(x + noise, 0.0, 1.0)

        H_matrix = None
        tps_info  = None

        # ── Homography ───────────────────────────────────────────────────
        if mode in ('homography', 'both'):
            H_np     = make_homography((h, w), difficulty)
            # Invert: warp_perspective expects dest→src mapping
            H_inv    = np.linalg.inv(H_np)
            H_matrix = torch.tensor(H_inv, dtype=torch.float32, device=device).unsqueeze(0)
            x = kornia.geometry.transform.warp_perspective(
                x, H_matrix, dsize=(h, w),
                padding_mode='border',   # ← replicate edges; no black fill
            )

        # ── TPS warp ─────────────────────────────────────────────────────
        # (Note: generateRandomTPS uses prob to decide if deformation occurs)
        if mode in ('tps', 'both'):
            src, weights, A = generateRandomTPS(
                (h, w), grid=tps_grid,
                GLOBAL_MULTIPLIER=difficulty,
                prob=tps_prob,
            )
            src     = src.to(device)
            weights = weights.to(device)
            A       = A.to(device)
            x       = warp_image_tps(x, src, weights, A)
            tps_info = {'src': src, 'weights': weights, 'A': A}

        # ── Final resize ─────────────────────────────────────────────────
        x = F.interpolate(x, (out_size[1], out_size[0]), mode='bilinear', align_corners=False)

    return x, H_matrix, tps_info


# ══════════════════════════════════════════════════════════════════════════════
# Visualisation
# ══════════════════════════════════════════════════════════════════════════════

def save_side_by_side(orig_bgr, trans_bgr, path):
    oh, ow = trans_bgr.shape[:2]
    orig_r = cv2.resize(orig_bgr, (ow, oh))
    div    = np.full((oh, 6, 3), 40, dtype=np.uint8)
    combo  = np.hstack([orig_r, div, trans_bgr])
    font   = cv2.FONT_HERSHEY_SIMPLEX
    fs, th = 0.85, 2
    cv2.putText(combo, 'Original',     (12, 34),      font, fs, (255, 255, 255), th)
    cv2.putText(combo, 'Transformed',  (ow + 16, 34), font, fs, (255, 255, 255), th)
    cv2.imwrite(path, combo)
    print(f"[✓] Side-by-side → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description='Geometric transform of an image using DALF augmentation logic.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--input',  '-i', required=True,  help='Input image path.')
    p.add_argument('--output', '-o', default='out_transformed.jpg', help='Output path.')
    p.add_argument('--mode',   '-m',
                   choices=['homography', 'tps', 'both'], default='tps',
                   help='Transform type: homography | tps | both')
    p.add_argument('--difficulty', '-d', type=float, default=0.4,
                   help='Warp strength 0→1  (0=identity, 1=extreme).')
    p.add_argument('--tps-prob', type=float, default=1.0,
                   help='Probability that TPS deformation is non-trivial.')
    p.add_argument('--tps-grid', type=int, nargs=2, default=[8, 6],
                   metavar=('ROWS', 'COLS'), help='TPS control-point grid.')
    p.add_argument('--warp-size', type=int, nargs=2, default=[900, 600],
                   metavar=('W', 'H'), help='Internal warp resolution (W H).')
    p.add_argument('--out-size', type=int, nargs=2, default=None,
                   metavar=('W', 'H'), help='Output resolution (W H). Default=warp-size.')
    p.add_argument('--color-aug', action='store_true',
                   help='Apply colour jitter + correlated noise before warping.')
    p.add_argument('--side-by-side', '-s', action='store_true',
                   help='Also save original-vs-transformed comparison image.')
    p.add_argument('--seed', type=int, default=None, help='Random seed.')
    p.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    return p.parse_args()


def main():
    args = parse_args()

    # ── Seeding ───────────────────────────────────────────────────────────
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        print(f"[i] Seed: {args.seed}")

    # ── Device ────────────────────────────────────────────────────────────
    device = (torch.device('cuda' if torch.cuda.is_available() else 'cpu')
              if args.device == 'auto' else torch.device(args.device))
    print(f"[i] Device       : {device}")

    warp_size = tuple(args.warp_size)
    out_size  = tuple(args.out_size) if args.out_size else warp_size
    print(f"[i] Warp size    : {warp_size[0]}×{warp_size[1]}")
    print(f"[i] Output size  : {out_size[0]}×{out_size[1]}")
    print(f"[i] Mode         : {args.mode}   difficulty={args.difficulty}")

    # ── Load ──────────────────────────────────────────────────────────────
    img_t    = load_tensor(args.input, warp_size, device)
    orig_bgr = cv2.imread(args.input)

    # ── Transform ─────────────────────────────────────────────────────────
    out_t, H_mat, tps_info = apply_transform(
        img_t, args.mode, args.difficulty,
        args.tps_prob, tuple(args.tps_grid),
        args.color_aug, device, out_size,
    )

    # ── Save ──────────────────────────────────────────────────────────────
    out_bgr = to_bgr_uint8(out_t)
    cv2.imwrite(args.output, out_bgr)
    print(f"[✓] Saved        → {args.output}")

    if args.side_by_side:
        base, ext = os.path.splitext(args.output)
        save_side_by_side(orig_bgr, out_bgr, f"{base}_comparison{ext}")

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n── Transform summary ─────────────────────────────────────────")
    print(f"   mode        : {args.mode}")
    print(f"   difficulty  : {args.difficulty}")
    if H_mat is not None:
        print(f"   homography H (inv, as passed to warp_perspective):\n"
              f"{np.array2string(H_mat.squeeze(0).cpu().numpy(), precision=4, suppress_small=True)}")
    if tps_info is not None:
        print(f"   TPS grid    : {args.tps_grid[0]}×{args.tps_grid[1]}, "
              f"prob={args.tps_prob}, deformed={'yes' if tps_info else 'no'}")
    print("──────────────────────────────────────────────────────────────")


if __name__ == '__main__':
    main()
