import os
import cv2
import torch
import numpy as np
import argparse

# Original DALF
from modules.models.DALF import DALF_extractor

# Enhanced Modules
from modules.preprocessing.clahe import apply_clahe
from modules.pipeline.multiscale_detection import multiscale_detect_dalf
from modules.postprocessing.subpixel_refinement import subpixel_refine
from modules.postprocessing.keypoint_filter import filter_keypoints
from clear_cuda import clear_cuda

def resize_if_larger(img, max_dim=800):
    """Utility to prevent massive images from OOM."""
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        return cv2.resize(img, (int(w * scale), int(h * scale)))
    return img

def run_enhanced_pipeline(
    img, 
    dalf_model, 
    use_clahe=True, 
    use_multiscale=True, 
    use_subpixel=True, 
    use_filtering=True,
    score_threshold=0.5,
    top_k=2048,
    num_points=None
):
    """
    Executes the enhanced feature matching pipeline.
    """
    print(f"[Pipeline] Processing image of shape {img.shape}")
    
    # 1. CLAHE Preprocessing
    if use_clahe:
        print("[1] Applying CLAHE...")
        processed_img = apply_clahe(img)
    else:
        processed_img = img.copy()
        
    # Ensure image is color BGR for DALF
    if len(processed_img.shape) == 2:
        processed_img = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2BGR)

    # 2. Multi-scale DALF detection
    print(f"[2] Running DALF (Multi-scale: {use_multiscale})...")
    if use_multiscale:
        kps, descs, heatmap = multiscale_detect_dalf(dalf_model, processed_img, top_k=top_k, return_map=True)
    else:
        outputs = dalf_model.detectAndCompute(processed_img, MS=False, return_map=True, top_k=top_k)
        if len(outputs) == 3:
            kps, descs, heatmap = outputs
        else:
            kps, descs = outputs
            heatmap = None
            
    print(f"    Raw keypoints extracted: {len(kps)}")

    # 3. Subpixel refinement
    if use_subpixel and heatmap is not None:
        print("[3] Refining keypoints to subpixel accuracy...")
        kps = subpixel_refine(kps, heatmap, window_size=3)

    # 4. Confidence-based keypoint filtering
    if use_filtering:
        print(f"[4] Filtering keypoints (score > {score_threshold} or top '{num_points}')...")
        kps, descs = filter_keypoints(kps, descs, score_threshold=score_threshold, num_points=num_points)
        print(f"    Keypoints remaining: {len(kps)}")
        
    return kps, descs

def geometric_filtering(kps1, kps2, good_matches, threshold=3.0):
    """Optional RANSAC-based geometric filtering (runs over matches)."""
    if len(good_matches) < 4:
        return good_matches, None
        
    src_pts = np.float32([kps1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kps2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, threshold)
    
    # Keep only inliers
    if mask is not None:
        inliers = [m for idx, m in enumerate(good_matches) if mask[idx][0] == 1]
    else:
        inliers = []
        
    return inliers, M

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced DALF Pipeline")
    parser.add_argument("--img1", type=str, default="./assets/book1.jpeg", help="Path to first image")
    parser.add_argument("--img2", type=str, default="./assets/book2.jpeg", help="Path to second image")
    parser.add_argument("--model", type=str, default=None, help="Custom DALF weights")
    parser.add_argument("--out", type=str, default="enhanced_matches.jpg", help="Output visualization path")
    parser.add_argument("--disable_clahe", action="store_true")
    parser.add_argument("--disable_multiscale", action="store_true")
    parser.add_argument("--disable_subpixel", action="store_true")
    parser.add_argument("--disable_filtering", action="store_true")
    parser.add_argument("--use_ransac", action="store_true", help="Apply geometric RANSAC filtering")
    parser.add_argument("--top_k", type=int, default=5000, help="Number of maximal keypoints to natively extract per image (Candidate pool for ANMS)")
    parser.add_argument("--num_points", type=int, default=None, help="Exact number of keypoints to extract and match (bypasses threshold)")
    parser.add_argument("--score_threshold", type=float, default=0.5, help="Confidence threshold for keypoints (default: 0.5)")
    
    args = parser.parse_args()
    
    # Setup Device & Model
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading DALF model on {dev}...")
    clear_cuda()
    dalf_model = DALF_extractor(dev=dev, model=args.model)
    
    # Load Images
    im1 = cv2.imread(args.img1)
    im2 = cv2.imread(args.img2)
    
    if im1 is None or im2 is None:
        print("Error: Could not load images.")
        exit(1)
        
    im1 = resize_if_larger(im1)
    im2 = resize_if_larger(im2)
    
    top_k_arg = args.top_k
    
    # Ensure candidate pool is large enough for ANMS to properly distribute keypoints
    if args.num_points is not None and top_k_arg <= args.num_points * 2:
        top_k_arg = args.num_points * 5
        print(f"Info: Bumping native extraction pool --top_k to {top_k_arg} to enable proper spatial distribution for {args.num_points} points.")
    
    # Extraction Wrap
    try:
        # Extract features with toggles
        kps1, desc1 = run_enhanced_pipeline(
            im1, dalf_model,
            use_clahe=not args.disable_clahe,
            use_multiscale=not args.disable_multiscale,
            use_subpixel=not args.disable_subpixel,
            use_filtering=not args.disable_filtering,
            score_threshold=args.score_threshold,
            top_k=top_k_arg,
            num_points=args.num_points
        )
        
        kps2, desc2 = run_enhanced_pipeline(
            im2, dalf_model,
            use_clahe=not args.disable_clahe,
            use_multiscale=not args.disable_multiscale,
            use_subpixel=not args.disable_subpixel,
            use_filtering=not args.disable_filtering,
            score_threshold=args.score_threshold,
            top_k=top_k_arg,
            num_points=args.num_points
        )
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("\n" + "="*50)
            print("ERROR: CUDA Out of Memory during extraction.")
            print(f"Details: {e}")
            print("SUGGESTION: Try reducing --top_k (e.g., --top_k 2048) or reducing image resolution.")
            print("="*50)
            clear_cuda()
            exit(1)
        else:
            raise e
    
    if len(kps1) > 0 and len(kps2) > 0:
        print("\\n[5] Matching Descriptors...")
        # L2 norm is used for DALF
        bf = cv2.BFMatcher(cv2.NORM_L2)
        knn_matches = bf.knnMatch(desc1.astype(np.float32), desc2.astype(np.float32), k=2)
        
        # Lowe's ratio test
        good_matches = []
        for m, n in knn_matches:
            if m.distance < 0.85 * n.distance:
                good_matches.append(m)
                
        print(f"    Raw good matches: {len(good_matches)}")
        
        # Optional geometric filtering
        if args.use_ransac:
            print("[6] Geometric filtering (RANSAC)...")
            good_matches, M = geometric_filtering(kps1, kps2, good_matches)
            print(f"    Inlier matches: {len(good_matches)}")
            
        # Draw matches
        num_draw = args.num_points if args.num_points is not None else 100
        matches_to_draw = sorted(good_matches, key=lambda x: x.distance)[:num_draw]
        
        out_img = cv2.drawMatches(
            im1, kps1, 
            im2, kps2, 
            matches_to_draw, 
            None, 
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        
        cv2.imwrite(args.out, out_img)
        print(f"Saved visualization to {args.out}")
    else:
        print("Not enough keypoints to perform matching.")
