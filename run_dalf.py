
'''
    Full DALF feature extraction pipeline with preprocessing, multi-scale 
    detection, and postprocessing for improved accuracy.
    
    Pipeline:
      1. CLAHE contrast enhancement (preprocessing)
      2. Multi-scale DALF keypoint detection
      3. Keypoint confidence filtering (postprocessing)
      4. Descriptor matching (BFMatcher example)
'''

from modules.models.DALF import DALF_extractor as DALF
from modules.preprocessing.clahe import apply_clahe
from modules.pipeline.multiscale_dalf import multiscale_detect
from modules.postprocessing.keypoint_filter import filter_keypoints
import argparse
import torch
import cv2
import numpy as np

parser = argparse.ArgumentParser(description="Run DALF feature extraction and matching")
parser.add_argument("--img1", type=str, default="./assets/kanagawa_1.png", help="Path to first image")
parser.add_argument("--img2", type=str, default="./assets/kanagawa_2.png", help="Path to second image")
parser.add_argument("--model", type=str, default=None, help="Path to custom model weights")
parser.add_argument("--out", type=str, default="matches.jpg", help="Path to save the matching visualization")
parser.add_argument("--num_points", type=int, default=None, help="Exact number of keypoints to extract and match (bypasses threshold)")
args = parser.parse_args()

dalf = DALF(dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), model=args.model)

# --- Load images ---
img1 = cv2.imread(args.img1)
img2 = cv2.imread(args.img2)
if img1 is None or img2 is None:
    raise ValueError("One or both images could not be loaded. Please check the paths.")

def resize_if_larger(img, max_dim=800):
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        return cv2.resize(img, (int(w * scale), int(h * scale)))
    return img

img1 = resize_if_larger(img1)
img2 = resize_if_larger(img2)

# --- Step 1: CLAHE Preprocessing ---
img1_enhanced = apply_clahe(img1)
img2_enhanced = apply_clahe(img2)

# Convert back to 3-channel for DALF (expects color input)
img1_input = cv2.cvtColor(img1_enhanced, cv2.COLOR_GRAY2BGR)
img2_input = cv2.cvtColor(img2_enhanced, cv2.COLOR_GRAY2BGR)

# --- Step 2: Multi-Scale Detection ---
top_k_arg = args.num_points if args.num_points is not None else 2048
kps1, desc1 = multiscale_detect(dalf, img1_input, scales=[1.0], top_k=top_k_arg)
kps2, desc2 = multiscale_detect(dalf, img2_input, scales=[1.0], top_k=top_k_arg)

print('--------------------------------------------------')
print("Before filtering:")
print(f"  Image 1 — Keypoints: {len(kps1)}, Descriptors: {desc1.shape}")
print(f"  Image 2 — Keypoints: {len(kps2)}, Descriptors: {desc2.shape}")

# --- Step 3: Keypoint Confidence Filtering ---
kps1, desc1 = filter_keypoints(kps1, desc1, score_threshold=0.5, num_points=args.num_points)
kps2, desc2 = filter_keypoints(kps2, desc2, score_threshold=0.5, num_points=args.num_points)

print("After filtering:")
print(f"  Image 1 — Keypoints: {len(kps1)}, Descriptors: {desc1.shape}")
print(f"  Image 2 — Keypoints: {len(kps2)}, Descriptors: {desc2.shape}")

# --- Step 4: Descriptor Matching ---
if len(kps1) > 0 and len(kps2) > 0:
    bf = cv2.BFMatcher(cv2.NORM_L2)
    knn_matches = bf.knnMatch(desc1.astype(np.float32), desc2.astype(np.float32), k=2)
    
    # Apply Lowe's ratio test
    good_matches = []
    ratio_thresh = 0.85
    for m, n in knn_matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)

    matches = sorted(good_matches, key=lambda x: x.distance)

    print(f"\nTotal good matches: {len(matches)}")
    print(f"Top-10 match distances: {[f'{m.distance:.3f}' for m in matches[:10]]}")
    
    # Draw matches
    num_draw = min(args.num_points if args.num_points is not None else 100, len(matches))
    match_img = cv2.drawMatches(
        img1, kps1, 
        img2, kps2, 
        matches[:num_draw], 
        None, 
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    
    cv2.imwrite(args.out, match_img)
    print(f"\nSaved matching visualization to: {args.out}")
else:
    print("\nNot enough keypoints for matching.")

print('--------------------------------------------------')