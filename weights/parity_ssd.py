"""Verify LibreSSD300 against the pinned torchvision reference graph.

Reference: https://github.com/pytorch/vision/tree/v0.26.0
Commit: 336d36e8db990a905498c73933e35231876e28bc
License: BSD-3-Clause

This probe checks the exact published checkpoint hash, fixed-size preprocessing,
default boxes, raw regression/classification heads, and mapped public COCO-80
detections. It never downloads or redistributes the upstream artifact.

Usage:
    python weights/parity_ssd.py upstream.pth --image tests/fixtures/dog.jpg
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.models.detection import ssd300_vgg16
from torchvision.models.detection.image_list import ImageList

from _conversion_utils import add_repo_root_to_path


EXPECTED_SHA256 = "b556d3b43ab6c3f63d81bfb8835fe8756ac22da664357da100dccf96b6a6b42d"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapped_reference(
    detection: dict[str, torch.Tensor],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from libreyolo.utils.coco import COCO91_TO_COCO80

    labels = detection["labels"]
    valid = torch.as_tensor(
        [int(label) in COCO91_TO_COCO80 for label in labels],
        dtype=torch.bool,
        device=labels.device,
    )
    boxes = detection["boxes"][valid].detach().cpu().numpy()
    scores = detection["scores"][valid].detach().cpu().numpy()
    classes = np.asarray(
        [COCO91_TO_COCO80[int(label)] for label in labels[valid].cpu()],
        dtype=np.int64,
    )
    return boxes, scores, classes


def verify(checkpoint_path: str, image_path: str, device: str = "auto") -> dict:
    """Run all parity gates and return their measured errors."""
    add_repo_root_to_path()
    from libreyolo.models.ssd.nn import LibreSSDModel
    from libreyolo.models.ssd.utils import preprocess_image
    from libreyolo.postprocess.ssd import _default_boxes, postprocess
    from libreyolo.utils.coco import COCO91_TO_COCO80

    checkpoint = Path(checkpoint_path)
    actual_hash = _sha256(checkpoint)
    if actual_hash != EXPECTED_SHA256:
        raise ValueError(
            f"Unexpected SSD checkpoint SHA-256 {actual_hash}; "
            f"expected {EXPECTED_SHA256}."
        )

    selected_device = torch.device(
        "cuda"
        if device == "auto" and torch.cuda.is_available()
        else device
        if device != "auto"
        else "cpu"
    )
    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    reference = ssd300_vgg16(weights=None, weights_backbone=None).eval()
    candidate = LibreSSDModel(num_classes=91).eval()
    reference.load_state_dict(state_dict, strict=True)
    candidate.load_state_dict(state_dict, strict=True)

    image = Image.open(image_path).convert("RGB")
    candidate_input = preprocess_image(image)[0]
    source = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1)
    source = source.to(dtype=torch.float32) / 255.0
    reference_input = reference.transform([source])[0].tensors
    preprocess_error = float((reference_input - candidate_input).abs().max())
    if preprocess_error != 0.0:
        raise AssertionError(f"preprocess max_abs_diff={preprocess_error}, expected 0")

    reference.to(selected_device)
    candidate.to(selected_device)
    candidate_input = candidate_input.to(selected_device)
    with torch.inference_mode():
        reference_features = list(reference.backbone(candidate_input).values())
        reference_raw = reference.head(reference_features)
        candidate_raw = candidate(candidate_input)

    raw_errors = {
        key: float((reference_raw[key] - candidate_raw[key]).abs().max())
        for key in reference_raw
    }
    if any(error != 0.0 for error in raw_errors.values()):
        raise AssertionError(f"raw head parity failed: {raw_errors}")

    image_list = ImageList(candidate_input, [(300, 300)])
    reference_anchors = reference.anchor_generator(image_list, reference_features)[0]
    candidate_anchors = _default_boxes(
        device=selected_device,
        dtype=candidate_input.dtype,
    )
    anchor_error = float((reference_anchors - candidate_anchors).abs().max())
    if anchor_error != 0.0:
        raise AssertionError(f"default-box max_abs_diff={anchor_error}, expected 0")

    with torch.inference_mode():
        reference_detection = reference.postprocess_detections(
            reference_raw,
            [reference_anchors],
            [(300, 300)],
        )[0]
    candidate_detection = postprocess(
        candidate_raw,
        conf_thres=reference.score_thresh,
        iou_thres=reference.nms_thresh,
        original_size=(300, 300),
        max_det=reference.detections_per_img,
        class_map=COCO91_TO_COCO80,
        topk_candidates=reference.topk_candidates,
    )
    reference_boxes, reference_scores, reference_classes = _mapped_reference(
        reference_detection
    )
    if len(reference_boxes) != candidate_detection["num_detections"]:
        raise AssertionError(
            "detection count mismatch: "
            f"reference={len(reference_boxes)}, "
            f"LibreYOLO={candidate_detection['num_detections']}"
        )
    if not np.array_equal(reference_classes, candidate_detection["classes"]):
        raise AssertionError("mapped COCO-80 labels differ")

    box_error = (
        float(np.max(np.abs(reference_boxes - candidate_detection["boxes"])))
        if len(reference_boxes)
        else 0.0
    )
    score_error = (
        float(np.max(np.abs(reference_scores - candidate_detection["scores"])))
        if len(reference_scores)
        else 0.0
    )
    if box_error > 1e-5 or score_error != 0.0:
        raise AssertionError(
            f"detection parity failed: boxes={box_error}, scores={score_error}"
        )

    return {
        "device": str(selected_device),
        "sha256": actual_hash,
        "preprocess_max_abs": preprocess_error,
        "raw_max_abs": raw_errors,
        "anchors_max_abs": anchor_error,
        "detections": int(candidate_detection["num_detections"]),
        "boxes_max_abs": box_error,
        "scores_max_abs": score_error,
        "labels_equal": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", help="Official SSD300-VGG16 .pth file")
    parser.add_argument(
        "--image",
        default="tests/fixtures/dog.jpg",
        help="RGB parity image",
    )
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    print(json.dumps(verify(args.checkpoint, args.image, args.device), indent=2))


if __name__ == "__main__":
    main()
