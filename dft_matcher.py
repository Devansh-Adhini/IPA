"""
DFT-Enhanced Feature Matching Pipeline
=======================================
A fully traditional image processing pipeline using:
  1. CLAHE                             -- local contrast equalisation
  2. DFT spectral whitening            -- removes global illumination bias
  3. DFT high-pass enhancement         -- sharpens structural edges
  4. Spectral Residual Saliency        -- guides keypoint selection
  5. RootSIFT descriptors              -- provably better nearest-neighbour matching
  6. Cross-check + ratio-test          -- mutual consistency filter
  7. RANSAC geometric pre-filter       -- removes structural outliers

Usage:
  python dft_matcher.py pair --img1 a.jpg --img2 b.jpg
  python dft_matcher.py eval --dataset ./dataset/Gendarmenmarkt --limit 50
"""

import cv2
import numpy as np
import argparse
import os
import tqdm
import time

from modules.dataset.augmentation import generateRandomHomography

# ---------------------------------------------------------------------------
# 1. DFT Preprocessing: Spectral Whitening
#    Divides spectrum by its magnitude -- equalises frequency energy,
#    making keypoints from textures rather than contrast gradients.
# ---------------------------------------------------------------------------
def spectral_whiten(gray: np.ndarray) -> np.ndarray:
    """
    Applies spectral whitening (phase-only transform) to an 8-bit grayscale
    image.  The output shares the same range [0,255] so downstream detectors
    work without modification.
    """
    f = np.fft.fft2(gray.astype(np.float32))
    magnitude = np.abs(f)
    magnitude[magnitude == 0] = 1e-8
    phase_only = f / magnitude          # unit-magnitude complex spectrum
    whitened = np.fft.ifft2(phase_only).real
    # normalise to [0,255]
    w_min, w_max = whitened.min(), whitened.max()
    whitened = (whitened - w_min) / (w_max - w_min + 1e-8) * 255
    return whitened.astype(np.uint8)


# ---------------------------------------------------------------------------
# 2. DFT High-Pass Enhancement
#    Keeps structural edges while suppressing flat/illumination regions.
# ---------------------------------------------------------------------------
def highpass_enhance(gray: np.ndarray, cutoff: float = 0.05) -> np.ndarray:
    """
    Zero-outs the central (low-frequency) circle of the DFT spectrum,
    then reconstructs with only high/mid frequencies.
    cutoff: fraction of image diagonal to suppress (0.05 = 5%)
    """
    h, w = gray.shape
    f_shift = np.fft.fftshift(np.fft.fft2(gray.astype(np.float32)))

    # Build radial high-pass mask
    cy, cx = h // 2, w // 2
    radius = int(min(h, w) * cutoff)
    Y, X = np.ogrid[:h, :w]
    mask = ((Y - cy)**2 + (X - cx)**2) > radius**2

    filtered = np.fft.ifft2(np.fft.ifftshift(f_shift * mask)).real
    filtered = np.clip(filtered, 0, 255).astype(np.uint8)

    # Blend: original * 0.6 + highpass * 0.4 -- keeps colour/contrast context
    out = cv2.addWeighted(gray, 0.6, filtered, 0.4, 0)
    return out


# ---------------------------------------------------------------------------
# 3. Spectral Residual Saliency Map (Hou & Zhang 2007)
#    Finds regions whose frequency content is *unusual* -- these are the
#    visually distinctive patches best suited for matching.
# ---------------------------------------------------------------------------
def spectral_residual_saliency(gray: np.ndarray) -> np.ndarray:
    """
    Returns a float32 saliency map in [0,1].
    High values indicate semantically distinctive regions.
    """
    f = np.fft.fft2(gray.astype(np.float32))
    log_mag = np.log(np.abs(f) + 1e-8)

    # Spectral residual = log magnitude − smoothed log magnitude
    smooth = cv2.blur(log_mag, (3, 3))
    residual = log_mag - smooth

    # Reconstruct with residual magnitude + original phase
    phase = np.angle(f)
    restored = np.fft.ifft2(np.exp(residual + 1j * phase)).real

    saliency = (restored ** 2)
    # Gaussian smoothing for spatial coherence
    saliency = cv2.GaussianBlur(saliency, (11, 11), 2.5)
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    return saliency.astype(np.float32)


# ---------------------------------------------------------------------------
# 4. Full Extraction Pipeline
#    DFT preprocessing  ->  Saliency-weighted SIFT detection  ->  SIFT description
# ---------------------------------------------------------------------------
def extract_features(img_gray: np.ndarray,
                     top_k: int = 2000,
                     use_saliency: bool = True):
    """
    DFT preprocessing steps:
      1. Spectral whitening        - removes illumination bias
      2. High-pass enhancement     - sharpens structural edges
      3. Spectral residual saliency- down-weights uninteresting regions
    Then standard SIFT detection + description on the enhanced image.
    """
    h, w = img_gray.shape[:2]

    # --- 1. CLAHE: local contrast normalisation ---
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(img_gray)

    # --- 2. DFT preprocessing stack ---
    whitened = spectral_whiten(clahe_img)
    enhanced = highpass_enhance(whitened)

    # --- 3. Spectral residual saliency map ---
    saliency = spectral_residual_saliency(enhanced) if use_saliency else None

    # --- 4. SIFT detection on enhanced image ---
    sift = cv2.SIFT_create(nfeatures=top_k * 3, contrastThreshold=0.01, edgeThreshold=15)
    kpts = sift.detect(enhanced, None)

    if not kpts:
        return [], np.empty((0, 128), np.float32)

    # Weight response by saliency, keep top_k
    if saliency is not None and len(kpts) > top_k:
        scores = []
        for kp in kpts:
            xi = int(np.clip(kp.pt[0], 0, w - 1))
            yi = int(np.clip(kp.pt[1], 0, h - 1))
            scores.append(kp.response * float(saliency[yi, xi]))
        order = np.argsort(scores)[::-1][:top_k]
        kpts = [kpts[i] for i in order]
    else:
        kpts = sorted(kpts, key=lambda k: k.response, reverse=True)[:top_k]

    # --- 5. SIFT description ---
    kpts, descs = sift.compute(enhanced, kpts)
    if descs is None:
        return [], np.empty((0, 128), np.float32)

    # --- 6. RootSIFT: L1-normalise then sqrt ---
    #   Proven by Arandjelovic & Zisserman 2012 to improve NN matching accuracy.
    descs = descs.astype(np.float32)
    descs /= (descs.sum(axis=1, keepdims=True) + 1e-8)   # L1 norm
    descs = np.sqrt(descs)                                 # element-wise sqrt

    return kpts, descs


# ---------------------------------------------------------------------------
# 6. Matching & Evaluation
# ---------------------------------------------------------------------------
def match_knn_ratio(descs1, descs2, ratio=0.80):
    """
    KNN ratio-test matching (Lowe 2004) at 0.80.
    Returns (idx1, idx2) arrays of matched indices.
    """
    bf = cv2.BFMatcher(cv2.NORM_L2)
    raw = bf.knnMatch(descs1, descs2, k=2)
    good1, good2 = [], []
    for pair in raw:
        if len(pair) == 2:
            m, n = pair
            if m.distance < ratio * n.distance:
                good1.append(m.queryIdx)
                good2.append(m.trainIdx)
    return np.array(good1), np.array(good2)


def evaluate_pair(img1_gray, img2_gray, H_np, top_k=2000,
                  px_threshold=3.0, ratio=0.80):
    """
    Extract, match, RANSAC pre-filter, then geometrically verify.
    Returns accuracy in [0, 100].
    """
    kpts1, desc1 = extract_features(img1_gray, top_k)
    kpts2, desc2 = extract_features(img2_gray, top_k)

    if len(kpts1) < 10 or len(kpts2) < 10:
        return None

    idx1, idx2 = match_knn_ratio(desc1, desc2, ratio)
    if len(idx1) < 4:
        return 0.0

    pts1 = np.float32([kpts1[i].pt for i in idx1])
    pts2 = np.float32([kpts2[i].pt for i in idx2])

    # --- RANSAC geometric pre-filter ---
    # Estimate homography from matches; keep only inliers.
    # This removes structurally inconsistent correspondences before scoring.
    if len(pts1) >= 4:
        _, mask = cv2.findHomography(pts1, pts2,
                                     cv2.RANSAC, ransacReprojThreshold=4.0,
                                     confidence=0.995)
        if mask is not None:
            mask = mask.ravel().astype(bool)
            pts1, pts2 = pts1[mask], pts2[mask]

    if len(pts1) == 0:
        return 0.0

    # --- Ground-truth homography verification ---
    pts1_h = np.hstack([pts1, np.ones((len(pts1), 1))])
    pts1_w = (H_np @ pts1_h.T).T
    pts1_w = pts1_w[:, :2] / pts1_w[:, 2:3]

    dist = np.linalg.norm(pts1_w - pts2, axis=1)
    correct = dist < px_threshold
    return float(correct.mean()) * 100.0


# ---------------------------------------------------------------------------
# 7. CLI: Single-pair visualisation  OR  dataset evaluation
# ---------------------------------------------------------------------------
def cmd_pair(args):
    img1 = cv2.imread(args.img1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(args.img2, cv2.IMREAD_GRAYSCALE)
    if img1 is None or img2 is None:
        raise FileNotFoundError("Could not read one or both images.")

    img1 = cv2.resize(img1, (800, 600))
    img2 = cv2.resize(img2, (800, 600))
    h, w = img1.shape[:2]

    kpts1, desc1 = extract_features(img1, top_k=args.top_k)
    kpts2, desc2 = extract_features(img2, top_k=args.top_k)

    idx1, idx2 = match_knn_ratio(desc1.astype(np.float32),
                                  desc2.astype(np.float32))

    print(f"Keypoints: {len(kpts1)} | {len(kpts2)}")
    print(f"Good matches: {len(idx1)}")

    out = cv2.drawMatches(cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR),
                          kpts1,
                          cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR),
                          kpts2,
                          [cv2.DMatch(i, j, 0)
                           for i, j in zip(idx1[:200], idx2[:200])],
                          None,
                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    out_path = "dft_matches.jpg"
    cv2.imwrite(out_path, out)
    print(f"Match visualisation saved to {out_path}")


def cmd_eval(args):
    img_files = [os.path.join(args.dataset, f)
                 for f in os.listdir(args.dataset)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not img_files:
        raise RuntimeError(f"No images found in {args.dataset}")

    img_files = img_files[:args.limit]
    print(f"DFT Pipeline — Evaluating {len(img_files)} images from {args.dataset}")

    accs, latencies = [], []
    for img_path in tqdm.tqdm(img_files, desc="Eval"):
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, (800, 600))
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        H_np = generateRandomHomography((h, w), GLOBAL_MULTIPLIER=args.difficulty)
        img_warped = cv2.warpPerspective(gray, H_np, (w, h),
                                         borderMode=cv2.BORDER_CONSTANT)
        t0 = time.perf_counter()
        acc = evaluate_pair(gray, img_warped, H_np,
                            top_k=args.top_k, ratio=args.ratio)
        t1 = time.perf_counter()

        if acc is not None:
            accs.append(acc)
            latencies.append((t1 - t0) * 1000)

    if accs:
        print("\n" + "=" * 40)
        print("  DFT PIPELINE — ACCURACY REPORT")
        print("=" * 40)
        print(f"  Images processed : {len(accs)}")
        print(f"  Mean Accuracy    : {np.mean(accs):.2f} %")
        print(f"  Mean Latency     : {np.mean(latencies):.1f} ms")
        print("=" * 40 + "\n")
    else:
        print("No valid pairs evaluated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DFT-based traditional feature matching pipeline")

    sub = parser.add_subparsers(dest="cmd")

    # --- pair mode ---
    p = sub.add_parser("pair", help="Match two specific images and visualise")
    p.add_argument("--img1", required=True, help="First image path")
    p.add_argument("--img2", required=True, help="Second image path")
    p.add_argument("--top_k", type=int, default=2000)
    p.add_argument("--ratio", type=float, default=0.80)

    # --- eval mode ---
    e = sub.add_parser("eval", help="Evaluate on a dataset with synthetic warps")
    e.add_argument("--dataset", required=True, help="Image directory")
    e.add_argument("--top_k", type=int, default=2000, help="Keypoints per image")
    e.add_argument("--limit", type=int, default=50, help="Max images to use")
    e.add_argument("--difficulty", type=float, default=0.3, help="Warp strength")
    e.add_argument("--ratio", type=float, default=0.80, help="KNN ratio threshold")

    args = parser.parse_args()

    if args.cmd == "pair":
        cmd_pair(args)
    elif args.cmd == "eval":
        cmd_eval(args)
    else:
        parser.print_help()
