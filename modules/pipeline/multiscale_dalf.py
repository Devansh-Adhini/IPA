import cv2
import numpy as np

def multiscale_detect(dalf, img, scales=[0.5, 0.75, 1.0, 1.25], top_k=2048):

    all_kps = []
    all_desc = []

    for s in scales:
        resized = cv2.resize(img, None, fx=s, fy=s)

        kps, desc = dalf.detectAndCompute(resized, top_k=top_k)

        for kp in kps:
            kp.pt = (kp.pt[0] / s, kp.pt[1] / s)

        all_kps.extend(kps)
        all_desc.append(desc)

    all_desc = np.vstack(all_desc)

    return all_kps, all_desc
