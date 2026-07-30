"""YOLO-NAS detect/pose Core ML graph parity on Apple hardware.

The fixtures use deterministic random parameters with amplified synthetic
heads.  They test conversion fidelity and input sensitivity without loading
the proprietary official YOLO-NAS checkpoints.  They do not establish model
accuracy or native-640 preprocessing accuracy.
"""

from __future__ import annotations

import gc
import sys

import numpy as np
import pytest
import torch

pytestmark = [
    pytest.mark.coreml,
    pytest.mark.e2e,
    pytest.mark.export_backend,
    pytest.mark.experimental_backend,
]

if sys.platform != "darwin":  # pragma: no cover - platform gate
    pytest.skip("Core ML artifacts only run on macOS", allow_module_level=True)

pytest.importorskip(
    "coremltools",
    reason="Core ML parity requires the coremltools runtime",
)

from .test_coreml_roundtrip import (  # noqa: E402
    _assert_model_artifact_parity,
    _match_public_detections,
    _pose_payload,
    _public_non_square_source,
)


@pytest.mark.parametrize(
    ("task", "size", "num_classes"),
    [
        pytest.param("detect", "s", 2, id="detect"),
        pytest.param("pose", "n", 1, id="pose"),
    ],
)
def test_coreml_yolonas_synthetic_named_output_parity(
    task,
    size,
    num_classes,
    tmp_path,
):
    """Compare both saved artifacts with the exact prepared PyTorch graph."""
    from libreyolo import LibreYOLONAS, LibreYOLO

    torch.manual_seed(20260729)
    model = LibreYOLONAS(
        None,
        size=size,
        nb_classes=num_classes,
        task=task,
        device="cpu",
    )
    network = model.model.eval()

    # Initial decoded boxes are dominated by the fixed anchor grid.  Amplify
    # only this generated fixture's prediction layers so every named output
    # changes well above conversion noise for the two-probe sensitivity gate.
    with torch.no_grad():
        for head in (
            network.heads.head1,
            network.heads.head2,
            network.heads.head3,
        ):
            head.reg_pred.weight.mul_(200.0)
            head.cls_pred.weight.mul_(200.0)
            if hasattr(head, "pose_pred"):
                head.pose_pred.weight.mul_(200.0)

    source = _public_non_square_source()
    predict_kwargs = {
        "conf": 1e-5,
        "iou": 0.6 if task == "pose" else 0.45,
        "max_det": 5,
        "verbose": False,
    }
    native = model.predict(source, imgsz=96, **predict_kwargs)
    artifact = _assert_model_artifact_parity(
        model,
        "yolonas",
        task,
        96,
        tmp_path,
    )
    del model
    gc.collect()
    deployed_model = LibreYOLO(artifact)
    deployed = deployed_model.predict(source, **predict_kwargs)
    repeated = deployed_model.predict(source, **predict_kwargs)
    order = _match_public_detections(
        native,
        deployed,
        repeated,
        minimum_iou=0.9999,
        maximum_score_error=1e-5,
    )
    if task == "pose":
        native_keypoints = _pose_payload(native)
        deployed_keypoints = _pose_payload(deployed)
        repeated_keypoints = _pose_payload(repeated)
        np.testing.assert_array_equal(deployed_keypoints, repeated_keypoints)
        aligned = deployed_keypoints[order]
        assert (
            float(
                np.abs(
                    native_keypoints[..., :2] - aligned[..., :2]
                ).max()
            )
            <= 1e-3
        )
        if native_keypoints.shape[-1] == 3:
            assert (
                float(
                    np.abs(
                        native_keypoints[..., 2] - aligned[..., 2]
                    ).max()
                )
                <= 1e-5
            )
