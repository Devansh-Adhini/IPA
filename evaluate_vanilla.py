import torch
import torch.nn.functional as F
import numpy as np
import argparse
import os
import sys

# Ensure we can import from the current directory
sys.path.append(os.getcwd())

from modules.models.DALF_vanilla import DEAL, DALF_extractor
from modules.dataset.augmentation import AugmentationPipe
from modules.utils import make_batch_sfm, get_positive_corrs
from modules.losses import hardnet_loss, differentiable_ap_loss, SSIMLoss

def evaluate_vanilla(model_path, datapath, mode, num_batches, verbose=True):
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if verbose: print(f"--- Vanilla Evaluation ({mode}) ---")
    if verbose: print("Loading vanilla model and dataset...")
    
    if mode == 'ts2' or mode == 'ts-fl':
        # Load through DALF_extractor
        extractor = DALF_extractor(model_path, dev=dev)
        net = extractor.net
    else:
        # Load standalone DEAL
        backbone_nfeats = 128 if mode == 'end2end-backbone' else 64
        net = DEAL(enc_channels=[1, 32, 64, backbone_nfeats], fixed_tps=False, mode=mode)
        
        if model_path and os.path.exists(model_path):
            if verbose: print(f"Loading weights from {model_path}...")
            net.load_state_dict(torch.load(model_path, map_location=dev))
        else:
            if verbose: print("[!] No valid model path provided. Evaluating with random initialization.")
            
    net.to(dev)
    net.eval()
    
    # Initialize the data augmentor
    # Using Madrid Metropolis or other standard dataset images
    augmentor = AugmentationPipe(device=dev, img_dir=datapath, max_num_imgs=200, 
                                 num_test_imgs=20, out_resolution=(300, 200), batch_size=4)
    
    # Tracking variables
    all_acc = []
    all_hard_loss = []
    all_ap_loss = []
    all_ssim = []
    all_logprobs = []
    
    with torch.no_grad():
        for step in range(num_batches):
            if verbose and step % 5 == 0: print(f"Processing batch {step}/{num_batches}...")
            
            p1, p2, Hs = make_batch_sfm(augmentor, difficulty=0.2)
            kpts1, desc1, out1 = net(p1, return_tensors=True, NMS=True)
            kpts2, desc2, out2 = net(p2, return_tensors=True, NMS=True)
            
            # Fix Long tensor issue for cdist
            for b in range(len(kpts1)):
              kpts1[b]['xy'] = kpts1[b]['xy'].float()
              kpts2[b]['xy'] = kpts2[b]['xy'].float()
            
            for b in range(p1.size(0)): 
                if kpts1[b]['patches'] is None or kpts2[b]['patches'] is None \
                   or len(kpts1[b]['xy']) < 16 or len(kpts2[b]['xy']) < 16:
                    continue
                
                idx, patches1, patches2 = get_positive_corrs(kpts1[b], kpts2[b], Hs[b], augmentor, i=1000)
                
                if len(patches1) < 16: 
                    continue
                
                # Extract localized descriptors for matched points
                # For vanilla, we follow the same sampling logic based on mode
                if mode == 'end2end-backbone' or mode == 'ts1':
                  l_pdesc1 = desc1[b][idx[:,0]]
                  l_pdesc2 = desc2[b][idx[:,1]]
                else:
                    nrdesc1 = net.hardnet(patches1)
                    nrdesc2 = net.hardnet(patches2)
                    
                    if mode != 'end2end-tps':
                        rdesc1 = net.interpolator(out1['feat'][b], kpts1[b]['xy'][idx[:,0],:], H=p1.shape[2], W=p1.shape[3])
                        rdesc2 = net.interpolator(out2['feat'][b], kpts2[b]['xy'][idx[:,1],:], H=p2.shape[2], W=p2.shape[3])
                        
                        if mode == 'ts-fl':
                            l_pdesc1 = F.normalize(net.fusion_layer(torch.cat((nrdesc1, rdesc1), dim=1)))
                            l_pdesc2 = F.normalize(net.fusion_layer(torch.cat((nrdesc2, rdesc2), dim=1)))
                        else:
                            l_pdesc1 = F.normalize(torch.cat((nrdesc1, rdesc1), dim=1))
                            l_pdesc2 = F.normalize(torch.cat((nrdesc2, rdesc2), dim=1))
                    else:
                        l_pdesc1 = F.normalize(nrdesc1)
                        l_pdesc2 = F.normalize(nrdesc2)
                
                # Metrics
                dist_matrix = torch.cdist(l_pdesc1, l_pdesc2)
                good_matches = torch.argmin(dist_matrix, dim=1) == torch.arange(len(l_pdesc1), device=l_pdesc1.device)
                all_acc.append(good_matches.sum().item() / len(good_matches))
                all_hard_loss.append(hardnet_loss(l_pdesc1, l_pdesc2, distance_weighted=True).item())
                all_ap_loss.append(differentiable_ap_loss(l_pdesc1, l_pdesc2).item())
                all_ssim.append(SSIMLoss(patches1, patches2).mean().item())

    if verbose:
        print("\n================= VANILLA EVALUATION RESULTS =================")
        print(f"Total Valid Image Pairs Analyzed : {len(all_acc)}")
        if len(all_acc) > 0:
            print(f"Feature Matching Accuracy        : {np.mean(all_acc)*100:.2f} %")
            print(f"Hard Triplet Loss                : {np.mean(all_hard_loss):.4f}")
            print(f"Differentiable AP Loss           : {np.mean(all_ap_loss):.4f}")
            print(f"Patch SSIM Loss                  : {np.mean(all_ssim):.4f}")
        print("==============================================================\n")
    
    return all_acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate Vanilla DALF accuracy and core losses")
    parser.add_argument('--model', type=str, default='./weights/model_ts-fl_final.pth', help='Path to vanilla .pth model')
    parser.add_argument('--datapath', type=str, default='./dataset/Madrid_Metropolis/images/*.jpg', help='Path to images')
    parser.add_argument('--mode', type=str, default='ts-fl', help='Model parsing mode')
    parser.add_argument('--batches', type=int, default=10, help='Number of batches to evaluate')
    args = parser.parse_args()
    
    evaluate_vanilla(args.model, args.datapath, args.mode, args.batches, verbose=True)
