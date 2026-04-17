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

from modules.dataset.augmentation import generateRandomHomography

def detect_architecture(model_path, dev):
    """Detects if the weights belong to Enhanced or Vanilla DALF."""
    state_dict = torch.load(model_path, map_location='cpu')
    keys = state_dict.keys()
    
    # Enhanced DALF has residual 'skip' connections and specific naming conventions
    is_enhanced = any('.skip.' in k for k in keys) or any('.se.' in k for k in keys)
    
    arch = 'enhanced' if is_enhanced else 'vanilla'
    print(f"Detected Architecture: {arch.upper()}")
    
    # Detect mode from filename
    modes = ['end2end-backbone', 'end2end-tps', 'end2end-full', 'ts1', 'ts2', 'ts-fl']
    mode = 'ts-fl' # Default
    for m in modes:
        if m in model_path:
            mode = m
            break
            
    if arch == 'vanilla':
        from modules.models.DALF_vanilla import DEAL
        backbone_nfeats = 128 if mode == 'end2end-backbone' else 64
        net = DEAL(enc_channels=[1, 32, 64, backbone_nfeats], fixed_tps=False, mode=mode).to(dev)
    else:
        from modules.models.DALF import DEAL
        net = DEAL(enc_channels=[1, 32, 64, 64], mode=mode).to(dev)
    
    net.load_state_dict(state_dict)
    net.eval()
    return net

def evaluate_accuracy(net, img_dir, difficulty=0.3, top_k=2000, limit=None):
    dev = next(net.parameters()).device
    
    img_files = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not img_files:
        print(f"No images found in {img_dir}")
        return None

    if limit:
        img_files = img_files[:limit]

    print(f"Evaluating {len(img_files)} images...")
    
    accuracies = []
    
    for img_path in tqdm.tqdm(img_files, desc="Eval"):
        # Load and resize
        img = cv2.imread(img_path)
        if img is None: continue
        img = cv2.resize(img, (800, 600))
        h, w = img.shape[:2]
        
        # Preprocess
        img_t = torch.tensor(img, dtype=torch.float32, device=dev).permute(2,0,1).unsqueeze(0)/255.
        img_t_gray = torch.mean(img_t, axis=1, keepdim=True)
        del img_t
        
        # Generate Warp
        H_np = generateRandomHomography((h, w), GLOBAL_MULTIPLIER=difficulty)
        H = torch.tensor(H_np, dtype=torch.float32, device=dev).unsqueeze(0)
        
        # We use cv2 for warping input to stay consistent with training pipeline if needed,
        # but kornia is better for GPU. Using kornia for efficiency.
        import kornia
        img_warped_t = kornia.geometry.transform.warp_perspective(img_t_gray, H, dsize=(h, w), padding_mode='zeros')
        
        try:
            with torch.no_grad():
                # Extract features (Using the fixed Hard-NMS and batching internally)
                kpts1_raw, desc1_raw, out1 = net(img_t_gray, return_tensors=True, NMS=True, top_k=top_k)
                kpts1_xy = kpts1_raw[0]['xy'].cpu()
                desc1 = desc1_raw[0].cpu()
                del kpts1_raw, desc1_raw, out1
                torch.cuda.empty_cache()
                
                kpts2_raw, desc2_raw, out2 = net(img_warped_t, return_tensors=True, NMS=True, top_k=top_k)
                kpts2_xy = kpts2_raw[0]['xy'].cpu()
                desc2 = desc2_raw[0].cpu()
                del kpts2_raw, desc2_raw, out2
                torch.cuda.empty_cache()

            # CPU Matching
            if len(kpts1_xy) < 10 or len(kpts2_xy) < 10:
                continue

            Dmat = 2. - torch.cdist(desc1.unsqueeze(0), desc2.unsqueeze(0)).squeeze(0)
            choice_rows = torch.argmax(Dmat, dim=1)
            choice_cols = torch.argmax(Dmat, dim=0)
            
            seq = torch.arange(len(choice_cols))
            mutual = (choice_rows[choice_cols] == seq)
            
            idx1, idx2 = choice_cols[mutual], seq[mutual]
            if len(idx1) == 0:
                accuracies.append(0.0)
                continue
                
            m_kpts1, m_kpts2 = kpts1_xy[idx1], kpts2_xy[idx2]
            
            # Warp verification
            pts1_h = torch.cat([m_kpts1, torch.ones(len(m_kpts1), 1)], dim=1)
            H_cpu = H[0].cpu()
            pts1_warped = (H_cpu @ pts1_h.T).T
            pts1_warped = pts1_warped[:, :2] / pts1_warped[:, 2:3]
            
            dist = torch.norm(pts1_warped - m_kpts2, dim=1)
            correct = dist < 3.0
            accuracies.append(correct.float().mean().item() * 100)
            
        except RuntimeError as e:
            if 'out of memory' in str(e):
                torch.cuda.empty_cache()
                continue
            raise e
        finally:
            del img_t_gray, img_warped_t, H
            gc.collect()

    return np.mean(accuracies) if accuracies else 0.0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate DALF accuracy on a dataset")
    parser.add_argument("--weights", type=str, required=True, help="Path to model weights (.pth)")
    parser.add_argument("--dataset", type=str, required=True, help="Path to image directory")
    parser.add_argument("--top_k", type=int, default=2000, help="Number of keypoints (default: 2000)")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of images (default: all)")
    args = parser.parse_args()
    
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    net = detect_architecture(args.weights, dev)
    
    limit = args.limit if args.limit > 0 else None
    acc = evaluate_accuracy(net, args.dataset, top_k=args.top_k, limit=limit)
    
    print("\n" + "="*30)
    print(f" FINAL ACCURACY: {acc:.2f} %")
    print("="*30 + "\n")
