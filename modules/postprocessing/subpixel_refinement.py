import numpy as np

def subpixel_refine(kps, heatmap, window_size=3):
    """
    Refines keypoint coordinates to subpixel accuracy using a 2D Taylor polynomial fit
    (quadratic peak interpolation) over the corresponding heatmap output from DALF.
    """
    if heatmap is None or len(kps) == 0:
        return kps

    H, W = heatmap.shape
    d = window_size // 2

    if heatmap.max() > 0:
        hmap_norm = heatmap / heatmap.max()
    else:
        hmap_norm = heatmap

    for kp in kps:
        x, y = int(round(kp.pt[0])), int(round(kp.pt[1]))

        if y - d < 0 or y + d >= H or x - d < 0 or x + d >= W:
            continue
            
        try:
            p11 = hmap_norm[y, x]
            p01 = hmap_norm[y-1, x]
            p21 = hmap_norm[y+1, x]
            p10 = hmap_norm[y, x-1]
            p12 = hmap_norm[y, x+1]
            
            dx = (p12 - p10) / 2.0
            dy = (p21 - p01) / 2.0

            dxx = p12 - 2.0 * p11 + p10
            dyy = p21 - 2.0 * p11 + p01

            offset_x = -dx / dxx if dxx != 0 else 0
            offset_y = -dy / dyy if dyy != 0 else 0

            offset_x = np.clip(offset_x, -0.5, 0.5)
            offset_y = np.clip(offset_y, -0.5, 0.5)

            kp.pt = (kp.pt[0] + offset_x, kp.pt[1] + offset_y)
        except Exception:
            pass

    return kps
