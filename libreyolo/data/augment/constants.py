"""Shared constants for augmentation pipelines.

Note there is deliberately no universal pad value here: most detection
pipelines fill with 114, but YOLO-NAS pose fills with 127
(``YOLO_NAS_POSE_PAD_VALUE`` in ``libreyolo/models/yolonas/utils.py``).
Recipes pass their own fill values explicitly.
"""

import numpy as np

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
