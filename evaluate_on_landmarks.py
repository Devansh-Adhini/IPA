import torch
import cv2
import numpy as np
import os
import argparse
import tqdm
import sys
import time
import gc

# Ensure we can import from the current directory
sys.path.append(os.getcwd())

from modules.utils import get_positive_corrs
from modules.dataset.augmentation import generateRandomHomography

def get_model(arch, model_path, dev):
    if arch == 'vanilla':
        from modules.models.DALF_vanilla import DEAL
        mode = 'ts-fl' 
        if 'end2end-backbone' in model_path: mode = 'end2end-backbone'
        
        backbone_nfeats = 128 if mode == 'end2end-backbone' else 64
        net = DEAL(enc_channels=[1, 32, 64, backbone_nfeats], fixed_tps=False, mode=mode).to(dev)
        net.load_state_dict(torch.load(model_path, map_location=dev))
        return net
    else:
        from modules.models.DALF import DEAL
        net = DEAL(enc_channels=[1, 32, 64, 64], mode='ts-fl').to(dev)
        net.load_state_dict(torch.load(model_path, map_location=dev))
        return net

def evaluate_on_landmarks(arch, model_path, img_dir, difficulty=0.3, top_k=2000):
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[{arch.upper()}] Initializing model on {dev} (Top-K: {top_k})...")
    
    try:
        net = get_model(arch, model_path, dev)
        net.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    img_files = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not img_files:
        print(f"No images found in {img_dir}")
        return

    print(f"Evaluating {len(img_files)} images...")
    
    accuracies = []
    kp_counts = []
    latencies = []

    for img_path in tqdm.tqdm(img_files, desc="Eval"):
        # Load image
        img = cv2.imread(img_path)
        if img is None: continue
        
        img = cv2.resize(img, (800, 600))
        h, w = img.shape[:2]
        
        # Convert to gray tensor [1, 1, 600, 800]
        img_t = torch.tensor(img, dtype=torch.float32, device=dev).permute(2,0,1).unsqueeze(0)/255.
        img_t_gray = torch.mean(img_t, axis=1, keepdim=True)
        del img_t
        
        # Generate random homography
        H_np = generateRandomHomography((h, w), GLOBAL_MULTIPLIER=difficulty)
        H = torch.tensor(H_np, dtype=torch.float32, device=dev).unsqueeze(0)
        
        import kornia
        img_warped_t = kornia.geometry.transform.warp_perspective(img_t_gray, H, dsize=(h, w), padding_mode='zeros')
        
        try:
            # First pass
            with torch.no_grad():
                kpts1_raw, desc1_raw, out1 = net(img_t_gray, return_tensors=True, NMS=True, top_k=top_k)
                # Keep only what's needed
                kpts1 = {'xy': kpts1_raw[0]['xy'].float().cpu()}
                desc1 = desc1_raw[0].cpu()
                del kpts1_raw, desc1_raw, out1
                torch.cuda.empty_cache()
                
                # Second pass
                torch.cuda.synchronize()
                t0 = time.time()
                kpts2_raw, desc2_raw, out2 = net(img_warped_t, return_tensors=True, NMS=True, top_k=top_k)
                torch.cuda.synchronize()
                t1 = time.time()
                
                kpts2 = {'xy': kpts2_raw[0]['xy'].float().cpu()}
                desc2 = desc2_raw[0].cpu()
                del kpts2_raw, desc2_raw, out2
                torch.cuda.empty_cache()

            # Matching logic (CPU to save VRAM)
            if len(kpts1['xy']) < 10 or len(kpts2['xy']) < 10:
                continue

            # Mutual Nearest Neighbors on CPU
            Dmat = 2. - torch.cdist(desc1.unsqueeze(0), desc2.unsqueeze(0)).squeeze(0)
            choice_rows = torch.argmax(Dmat, dim=1)
            choice_cols = torch.argmax(Dmat, dim=0)
            
            seq = torch.arange(len(choice_cols))
            mutual = choice_rows[choice_cols] == seq
            
            idx1 = choice_cols[mutual]
            idx2 = seq[mutual]
            
            if len(idx1) == 0:
                accuracies.append(0.0)
                continue
                
            m_kpts1 = kpts1['xy'][idx1]
            m_kpts2 = kpts2['xy'][idx2]
            
            # Verify matches using H (on CPU)
            pts1_h = torch.cat([m_kpts1, torch.ones(len(m_kpts1), 1)], dim=1)
            H_cpu = H[0].cpu()
            pts1_warped = (H_cpu @ pts1_h.T).T
            pts1_warped = pts1_warped[:, :2] / pts1_warped[:, 2:3]
            
            dist = torch.norm(pts1_warped - m_kpts2, dim=1)
            correct = dist < 3.0
            
            acc = correct.float().mean().item() * 100
            accuracies.append(acc)
            kp_counts.append(len(kpts1['xy']))
            latencies.append((t1 - t0) * 1000)
            
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print(f"Skipping image due to OOM: {os.path.basename(img_path)}")
                torch.cuda.empty_cache()
                gc.collect()
                continue
            else:
                raise e
        finally:
            del img_t_gray, img_warped_t, H
            gc.collect()

    if not accuracies:
        print("No valid pairs processed.")
        return

    print("\n" + "="*50)
    print(f" LANDMARK ACCURACY REPORT [{arch.upper()}] ")
    print("="*50)
    print(f"Model Path: {os.path.basename(model_path)}")
    print(f"Top-K Limit:      {top_k}")
    print(f"Processed Images: {len(accuracies)}")
    print(f"Mean Accuracy:    {np.mean(accuracies):.2f} %")
    print(f"Mean Keypoints:   {np.mean(kp_counts):.1f}")
    print(f"Mean Latency:     {np.mean(latencies):.2f} ms")
    print("="*50 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate DALF accuracy on custom landmarks")
    parser.add_argument("--arch", type=str, required=True, choices=['vanilla', 'enhanced'], help="Architecture type")
    parser.add_argument("--model", type=str, required=True, help="Path to model weights (.pth)")
    parser.add_argument("--img_dir", type=str, default="./dataset/google_landmarks", help="Directory of images")
    parser.add_argument("--difficulty", type=float, default=0.3, help="Warp difficulty (0.1 to 1.0)")
    parser.add_argument("--top_k", type=int, default=2000, help="Limit to top K keypoints")
    args = parser.parse_args()
    
    evaluate_on_landmarks(args.arch, args.model, args.img_dir, args.difficulty, args.top_k)
