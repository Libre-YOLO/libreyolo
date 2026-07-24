# Copyright 2026 SenseTime Group Inc. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
#
# Task prompt templates ported for LibreYOLO from OpenSenseNova/SenseNova-Vision
# (commit 12ccd96e32b32967a11cacb6c5bd5fe3a555fc0c, Apache-2.0), files
# inference/inference_demo.py and data/prompts.py. The prompts are the exact
# instructions the released checkpoint was trained/evaluated with; paraphrasing
# them degrades output quality, so they are kept verbatim.

DEPTH_PROMPT = (
    "Estimate relative depth for each pixel in the image, with closer objects "
    "appearing brighter and distant objects appearing darker. Output is a "
    "grayscale image with pixel values ranging from 0-255."
)

NORMAL_PROMPT = (
    "Estimate surface normals and encode as an RGB image. Each channel "
    "corresponds to a direction component (X, Y, Z) with continuous value "
    "variations, creating smooth color gradients distinct from other task outputs."
)

OCR_PROMPT = (
    "Perform word-level text detection and recognition on the entire image. "
    "Output a structured text list containing every detected word, its bounding "
    "box coordinates with <bbox> format, and the recognized text content."
)

BBOX_DETECTION_PROMPT_TEMPLATE = (
    "Detect all instances of {categories} in the image. Output the results "
    "as a structured text list with each detection including category and "
    "bounding box coordinates in <bbox> format."
)

POINT_DETECTION_PROMPT_TEMPLATE = (
    "Locate and identify {categories} within the scene. Output detection "
    "results as text entries, each containing the object class and pixel "
    "coordinates defining the object point location."
)

KEYPOINT_PROMPT_TEMPLATES = {
    "human": (
        "Detect all instances of {categories} in the image. For each instance, "
        "output a bounding box in <bbox> format and the coordinates of its "
        "nose, left eye, right eye, left ear, right ear, left shoulder, right "
        "shoulder, left elbow, right elbow, left wrist, right wrist, left hip, "
        "right hip, left knee, right knee, left ankle, right ankle in "
        "<kpt>[x,y]</kpt> format. Return results as a structured list."
    ),
    "animal": (
        "Detect all instances of {categories} in the image. For each instance, "
        "output a bounding box in <bbox> format and the coordinates of its "
        "left eye, right eye, nose, neck, root of tail, left shoulder, left "
        "elbow, left front paw, right shoulder, right elbow, right front paw, "
        "left hip, left knee, left back paw, right hip, right knee, right back "
        "paw in <kpt>[x,y]</kpt> format. Return results as a structured list."
    ),
    "mixed": (
        "Detect all instances of {categories} in the image. For human "
        "instances, output a bounding box in <bbox> format and the COCO human "
        "keypoints: nose, left eye, right eye, left ear, right ear, left "
        "shoulder, right shoulder, left elbow, right elbow, left wrist, right "
        "wrist, left hip, right hip, left knee, right knee, left ankle, right "
        "ankle. For animal instances, output a bounding box in <bbox> format "
        "and the AP-10K animal keypoints: left eye, right eye, nose, neck, "
        "root of tail, left shoulder, left elbow, left front paw, right "
        "shoulder, right elbow, right front paw, left hip, left knee, left back "
        "paw, right hip, right knee, right back paw. Use <kpt>[x,y]</kpt> "
        "format for every keypoint and return results as a structured list."
    ),
}

HUMAN_KEYPOINT_CATEGORIES = {"human", "person", "people", "pedestrian"}

MASK_QUESTION_TEMPLATE = 'Can you segment the image based on the following categories: {categories}? Please output the {task_type} segmentation masks.'

CAPTION_QUESTION_TEMPLATE = 'Please find all instances in the image and assign color to each instance in the EXACT format: <p>instance-no<color>(R,G,B)</color></p>, then respond with {task_type} segmentation masks.'

# COCO panoptic category names (things + stuff, dataset-suffixes stripped),
# the default vocabulary for panoptic segmentation when set_classes() was
# not called. Factual dataset metadata from data/prompts.py.
COCO_PANOPTIC_NAMES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane',
    'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
    'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
    'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
    'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
    'cake', 'chair', 'couch', 'potted plant', 'bed',
    'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
    'toaster', 'sink', 'refrigerator', 'book', 'clock',
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush',
    'banner', 'blanket', 'bridge', 'cardboard', 'counter',
    'curtain', 'door', 'floor-wood', 'flower', 'fruit',
    'gravel', 'house', 'light', 'mirror', 'net',
    'pillow', 'platform', 'playingfield', 'railroad', 'river',
    'road', 'roof', 'sand', 'sea', 'shelf',
    'snow', 'stairs', 'tent', 'towel', 'wall-brick',
    'wall-stone', 'wall-tile', 'wall-wood', 'water', 'window-blind',
    'window', 'tree', 'fence', 'ceiling', 'sky',
    'cabinet', 'table', 'floor', 'pavement', 'mountain',
    'grass', 'dirt', 'paper', 'food', 'building',
    'rock', 'wall', 'rug',
]
