import cv2
import numpy as np

def multiscale_detect_dalf(dalf, img, top_k=2048, return_map=True):
    """
    Calls DALF's detectAndCompute with MS=True.
    This natively handles multi-scale pyramids inside DALF
    and returns properly merged keypoints, descriptors, and heatmap.
    """
    # Ensure color image
    if len(img.shape) == 2:
        img_input = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img_input = img
        
    outputs = dalf.detectAndCompute(img_input, MS=True, return_map=return_map, top_k=top_k)
    
    if return_map:
        return outputs[0], outputs[1], outputs[2]
    else:
        return outputs[0], outputs[1]
