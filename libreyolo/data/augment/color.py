"""Color-space augmentations shared by the numpy pipelines.

Moved verbatim from ``libreyolo/training/augment.py`` (originally adapted
from the official YOLOX repository).
"""

import cv2
import numpy as np


def augment_hsv(img, hgain=5, sgain=30, vgain=30):
    """Random HSV jitter (in-place)."""
    hsv_augs = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain]
    hsv_augs *= np.random.randint(0, 2, 3)  # randomly zero-out each channel
    hsv_augs = hsv_augs.astype(np.int16)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)

    img_hsv[..., 0] = (img_hsv[..., 0] + hsv_augs[0]) % 180
    img_hsv[..., 1] = np.clip(img_hsv[..., 1] + hsv_augs[1], 0, 255)
    img_hsv[..., 2] = np.clip(img_hsv[..., 2] + hsv_augs[2], 0, 255)

    cv2.cvtColor(img_hsv.astype(img.dtype), cv2.COLOR_HSV2BGR, dst=img)
