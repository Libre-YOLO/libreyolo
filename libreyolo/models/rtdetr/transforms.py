"""RT-DETR training transform: stretch-to-square resize (matches eval/inference).

RT-DETR validation (:class:`RTDETRValPreprocessor` /
:class:`RTDETRv2ValPreprocessor`) and inference
(:func:`libreyolo.models.rtdetr.utils.preprocess_numpy`) resize each image with
a plain **stretch** to the square model input — no aspect preservation, no
letterbox padding. Upstream RT-DETR does the same for both train and eval.

The trainer previously reused ``YOLO9TrainTransform``, which *letterboxes*
(aspect-preserving + gray padding). That made training see a different image
distribution than validation/inference (issue: RT-DETR trains letterboxed but
validates/infers stretched). This transform keeps the YOLO9 photometric + flip
augmentations but swaps the letterbox resize for the same stretch-to-square
geometry as eval, so train and eval agree.

Output contract is unchanged from ``YOLO9TrainTransform``:
``(image_chw_float01_rgb, padded_labels)`` where ``padded_labels`` is
``[class, x1, y1, x2, y2]`` normalized to ``[0, 1]`` and padded to
``max_labels`` rows with ``class = -1`` marking empty slots.
"""

from __future__ import annotations

import random

import cv2
import numpy as np

from libreyolo.data.augment.color import augment_hsv


class RTDETRTrainTransform:
    """Stretch-to-square RT-DETR training transform."""

    # Take the un-letterboxed original image + original-coordinate labels from
    # the dataset and own the (single) resize here — the same opt-in the stretch
    # val preprocessors use. This avoids a letterbox-then-stretch double resize
    # and makes the net geometry a pure stretch to the square input.
    wants_unresized_image = True

    def __init__(
        self,
        max_labels=300,
        flip_prob=0.5,
        vertical_flip_prob=0.0,
        hsv_prob=1.0,
    ):
        self.max_labels = max_labels
        self.flip_prob = flip_prob
        self.vertical_flip_prob = vertical_flip_prob
        self.hsv_prob = hsv_prob

    @staticmethod
    def _finalize_image(image, input_dim):
        """Stretch-resize to (H, W) then BGR->RGB, HWC->CHW, /255 float32.

        Matches ``letterbox_preproc(to_rgb=True, scale=True)``'s finalize
        (RGB, 0-1) and ``RTDETRValPreprocessor``'s resize/normalize — only the
        resize geometry differs (stretch vs letterbox).
        """
        target_h, target_w = input_dim
        resized = cv2.resize(
            image, (target_w, target_h), interpolation=cv2.INTER_LINEAR
        )
        resized = resized[:, :, ::-1]  # BGR -> RGB
        resized = resized.transpose(2, 0, 1)  # HWC -> CHW
        return np.ascontiguousarray(resized, dtype=np.float32) / 255.0

    def __call__(self, image, targets, input_dim, segments=None):
        if segments is not None:
            # RT-DETR is detection-only; the mask/segment path is never wired.
            raise NotImplementedError(
                "RTDETRTrainTransform does not support segment masks"
            )

        target_h, target_w = input_dim
        boxes = targets[:, :4].copy()
        labels = targets[:, 4].copy()

        if len(boxes) == 0:
            padded_labels = np.zeros((self.max_labels, 5), dtype=np.float32)
            padded_labels[:, 0] = -1  # -1 marks empty slots
            # Background images get the same photometric/flip augmentation as
            # labeled ones (no labels to sync), matching YOLO9TrainTransform.
            if random.random() < self.hsv_prob:
                augment_hsv(image)
            if random.random() < self.flip_prob:
                image = image[:, ::-1]
            if random.random() < self.vertical_flip_prob:
                image = image[::-1, :]
            return self._finalize_image(image, input_dim), padded_labels

        # Keep an un-augmented copy for the all-boxes-filtered fallback.
        image_o = image.copy()
        boxes_o = boxes.copy()
        labels_o = labels.copy()

        if random.random() < self.hsv_prob:
            augment_hsv(image)

        # Flips operate in the current image's pixel space (before resize).
        height, width = image.shape[:2]
        if random.random() < self.flip_prob:
            image_t = image[:, ::-1]
            boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
        else:
            image_t = image

        height = image_t.shape[0]
        if random.random() < self.vertical_flip_prob:
            image_t = image_t[::-1, :]
            boxes[:, [1, 3]] = height - boxes[:, [3, 1]]

        # Stretch resize: independent x/y scale (no aspect preservation).
        cur_h, cur_w = image_t.shape[:2]
        rx = target_w / cur_w
        ry = target_h / cur_h
        image_out = self._finalize_image(image_t, input_dim)
        boxes[:, [0, 2]] *= rx
        boxes[:, [1, 3]] *= ry

        # Drop degenerate boxes after resize.
        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]
        mask = (w > 1) & (h > 1)
        boxes_t = boxes[mask]
        labels_t = labels[mask]

        # Fallback to the original image if everything was filtered out.
        if len(boxes_t) == 0:
            o_h, o_w = image_o.shape[:2]
            rx_o = target_w / o_w
            ry_o = target_h / o_h
            image_out = self._finalize_image(image_o, input_dim)
            boxes_t = boxes_o.copy()
            boxes_t[:, [0, 2]] *= rx_o
            boxes_t[:, [1, 3]] *= ry_o
            labels_t = labels_o

        # Normalize to [0, 1] by the (square) target size and clip.
        boxes_norm = boxes_t.copy()
        boxes_norm[:, [0, 2]] /= target_w
        boxes_norm[:, [1, 3]] /= target_h
        boxes_norm = np.clip(boxes_norm, 0, 1)

        # Format: [class, x1, y1, x2, y2].
        targets_t = np.hstack((np.expand_dims(labels_t, 1), boxes_norm))

        padded_labels = np.zeros((self.max_labels, 5), dtype=np.float32)
        padded_labels[:, 0] = -1
        n = min(len(targets_t), self.max_labels)
        padded_labels[:n] = targets_t[:n]

        return image_out, padded_labels
