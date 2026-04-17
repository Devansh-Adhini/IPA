import torch
import cv2
import numpy as np
import os
import argparse
import sys
import gc

# Ensure we can import from the current directory
sys.path.append(os.getcwd())

from modules.dataset.augmentation import generateRandomHomography

def get_model(arch, model_path, dev):
    """Loads the appropriate DALF model based on architecture choice."""
    if arch == 'vanilla':
        from modules.models.DALF_vanilla import DEAL
        # For vanilla, we often check the filename for the mode, defaulting to ts-fl
        mode = 'end2end-backbone' if 'end2end-backbone' in model_path else 'ts-fl'
        backbone_nfeats = 128 if mode == 'end2end-backbone' else 64
        net = DEAL(enc_channels=[1, 32, 64, backbone_nfeats], fixed_tps=False, mode=mode).to(dev)
    else:
        from modules.models.DALF import DEAL
        # Enhanced architecture defaults to ts-fl with specific channel counts
        net = DEAL(enc_channels=[1, 32, 64, 64], mode='ts-fl').to(dev)
    
    net.load_state_dict(torch.load(model_path, map_location=dev))
    net.eval()
    return net

def calculate_accuracy(net, img_path, dev, top_k=2000, difficulty=0.3):
    """Calculates matching accuracy for a single image and its synthetic warp."""
    img = cv2.imread(img_path)
    if img is None: return None
    
    img = cv2.resize(img, (800, 600))
    h, w = img.shape[:2]
    
    # Preprocess image
    img_t = torch.tensor(img, dtype=torch.float32, device=dev).permute(2,0,1).unsqueeze(0)/255.
    img_gray = torch.mean(img_t, axis=1, keepdim=True)
    
    # Generate random homography and warp
    H_np = generateRandomHomography((h, w), GLOBAL_MULTIPLIER=difficulty)
    img_warped_np = cv2.warpPerspective(img_gray.squeeze().cpu().numpy(), H_np, (w, h))
    img_warped_t = torch.tensor(img_warped_np, device=dev).unsqueeze(0).unsqueeze(0)
    
    with torch.no_grad():
        # Feature extraction
        kpts1_data, desc1_list = net(img_gray, NMS=True, top_k=top_k)
        kpts2_data, desc2_list = net(img_warped_t, NMS=True, top_k=top_k)
        
        if not desc1_list[0] is not None or not desc2_list[0] is not None:
            return None
        
        desc1, desc2 = desc1_list[0], desc2_list[0]
        kpts1_xy, kpts2_xy = kpts1_data[0]['xy'].cpu(), kpts2_data[0]['xy'].cpu()
        
        if len(kpts1_xy) < 10 or len(kpts2_xy) < 10:
            return None

        # Mutual Nearest Neighbors
        dmat = 2. - torch.cdist(desc1.unsqueeze(0), desc2.unsqueeze(0)).squeeze(0)
        choice_rows = torch.argmax(dmat, dim=1)
        choice_cols = torch.argmax(dmat, dim=0)
        
        seq = torch.arange(len(choice_cols), device=dmat.device)
        mutual = choice_rows[choice_cols] == seq
        
        idx1, idx2 = choice_cols[mutual], seq[mutual]
        if len(idx1) == 0: return 0.0
            
        m_kpts1, m_kpts2 = kpts1_xy[idx1.cpu()], kpts2_xy[idx2.cpu()]
        
        # Homography verification
        pts1_h = np.concatenate([m_kpts1.numpy(), np.ones((len(m_kpts1), 1))], axis=1)
        pts1_warped = (H_np @ pts1_h.T).T
        pts1_warped = pts1_warped[:, :2] / pts1_warped[:, 2:3]
        
        dist = np.linalg.norm(pts1_warped - m_kpts2.numpy(), axis=1)
        correct = dist < 3.0
        return np.mean(correct) * 100

def main():
    parser = argparse.ArgumentParser(description="Calculate Test Accuracy ONLY")
    parser.add_argument("--arch", type=str, required=True, choices=['vanilla', 'enhanced'], help="Architecture type")
    parser.add_argument("--model", type=str, required=True, help="Path to weights")
    parser.add_argument("--dataset", type=str, required=True, help="Image directory")
    parser.add_argument("--difficulty", type=float, default=0.3, help="Warp difficulty")
    parser.add_argument("--limit", type=int, default=50, help="Limit number of images for speed")
    args = parser.parse_args()

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        net = get_model(args.arch, args.model, dev)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    img_files = [os.path.join(args.dataset, f) for f in os.listdir(args.dataset) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not img_files:
        print("No images found.")
        sys.exit(1)

    img_files = img_files[:args.limit]
    accuracies = []

    for img_path in img_files:
        acc = calculate_accuracy(net, img_path, dev, difficulty=args.difficulty)
        if acc is not None:
            accuracies.append(acc)
    
    if accuracies:
        mean_acc = np.mean(accuracies)
        print(f"Accuracy: {mean_acc:.2f}%")
    else:
        print("Accuracy: 0.00% (No valid pairs)")

if __name__ == "__main__":
    main()
