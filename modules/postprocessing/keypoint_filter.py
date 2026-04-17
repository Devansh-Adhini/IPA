import numpy as np

def distribute_keypoints_anms(kps, desc, num_points):
    """Adaptive Non-Maximal Suppression to distribute keypoints evenly."""
    if len(kps) <= num_points:
        return kps, desc

    pts = np.array([kp.pt for kp in kps])
    responses = np.array([kp.response for kp in kps])
    
    indices = np.argsort(-responses)
    pts_sorted = pts[indices]
    
    radii = np.full(len(kps), np.inf)
    
    for i in range(1, len(kps)):
        diffs = pts_sorted[:i] - pts_sorted[i]
        dists = diffs[:, 0]**2 + diffs[:, 1]**2
        radii[i] = np.min(dists)
        
    top_indices = np.argsort(-radii)[:num_points]
    
    filtered_kps = [kps[indices[i]] for i in top_indices]
    filtered_desc = desc[indices[top_indices]]
    
    return filtered_kps, filtered_desc

def filter_keypoints(kps, desc, score_threshold=0.5, num_points=None, use_anms=True):
    """
    Removes weak detections using the keypoint response threshold
    while preserving descriptor alignment, or optionally keeps the 
    exact top N keypoints by response. If use_anms is True, applies
    spatial distribution to prevent clustering.
    """
    if num_points is not None:
        if len(kps) == 0:
            return kps, desc
        
        if use_anms:
            return distribute_keypoints_anms(kps, desc, num_points)
            
        indices = np.argsort([-kp.response for kp in kps])[:num_points]
        filtered_kps = [kps[i] for i in indices]
        filtered_desc = desc[indices] if isinstance(desc, np.ndarray) else np.array([desc[i] for i in indices])
        return filtered_kps, filtered_desc

    filtered_kps = []
    filtered_desc = []

    for i, kp in enumerate(kps):
        if kp.response > score_threshold:
            filtered_kps.append(kp)
            filtered_desc.append(desc[i])

    if len(filtered_desc) > 0:
        filtered_desc = np.stack(filtered_desc)
    else:
        filtered_desc = np.array([])
        
    return filtered_kps, filtered_desc
