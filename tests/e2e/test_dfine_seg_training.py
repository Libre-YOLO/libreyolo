"""
E2E: D-FINE segmentation training and inference tests.

Trains D-FINE-seg from the published detect checkpoint (explicit
detect-to-segment transfer: the mask head starts untrained) for 2 epochs on
LibreYOLO/fire-smoke-seg (HuggingFace, public, 141 train / 40 valid / 20 test
images, 2 classes: fire & smoke, YOLO segmentation format with polygon
annotations), then verifies the checkpoint round-trips as task='segment' and
that inference and mask-mAP validation work.

COCO-pretrained LibreDFINE*-seg weights exist on HF; this test deliberately
exercises the detect-to-segment transfer path (untrained mask head), which is
how users fine-tune on custom data starting from detect weights.

The dataset auto-downloads from HuggingFace — no API keys needed.

Usage:
    pytest tests/e2e/test_dfine_seg_training.py -v -m e2e
"""

from pathlib import Path

import pytest

from .conftest import requires_cuda, run_in_subprocess
from .test_rfdetr_seg_training import download_fire_smoke_dataset, patch_data_yaml

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.dfine,
    pytest.mark.slow,
    pytest.mark.flagship_nightly,
]


@pytest.fixture(scope="module")
def dataset():
    """Download fire-smoke-seg dataset and patch data.yaml."""
    from .test_rfdetr_seg_training import DATASET_ROOT

    download_fire_smoke_dataset()
    patch_data_yaml()
    return DATASET_ROOT


@requires_cuda
@pytest.mark.parametrize("size", ["n"])
def test_dfine_seg_transfer_training(dataset, tmp_path, size):
    """Detect-to-segment transfer train, then verify ckpt task + masks + mAP."""
    data_yaml_py = repr(str(dataset / "data.yaml"))
    project_py = repr(str(tmp_path / "dfine_seg"))

    # The body runs behind a __main__ guard: D-FINE training uses dataloader
    # workers, and on Windows the multiprocessing spawn start method re-imports
    # the script module — flat top-level code would re-execute in every worker
    # and deadlock the run.
    run_in_subprocess(
        f"""
        from pathlib import Path

        def _main():
            import torch

            from libreyolo import LibreYOLO, SAMPLE_IMAGE
            from libreyolo.models.dfine.model import LibreDFINE

            # 1. Resolve the published detect checkpoint (auto-download), then
            # re-open it as an explicit detect-to-segment transfer.
            det = LibreYOLO("LibreDFINE{size}.pt")
            det_path = Path("weights") / "LibreDFINE{size}.pt"
            assert det_path.exists(), f"expected auto-downloaded weights at {{det_path}}"

            model = LibreDFINE(
                str(det_path),
                size="{size}",
                task="segment",
                allow_detect_to_segment_transfer=True,
            )
            assert model.task == "segment"

            # 2. Short transfer-training run — smoke that the seg train loop
            # runs end-to-end and writes a LibreYOLO-style checkpoint.
            model.train(
                data={data_yaml_py},
                epochs=2,
                batch=4,
                imgsz=640,
                project={project_py},
                name="run",
            )

            ckpt_path = Path({project_py}) / "run" / "weights" / "last.pt"
            assert ckpt_path.exists(), f"missing {{ckpt_path}}"

            # 3. Checkpoint must round-trip as a segment checkpoint: task
            # metadata plus mask-head weights.
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            assert ckpt.get("task") == "segment", f"task metadata: {{ckpt.get('task')}}"
            state = ckpt.get("model", ckpt)
            mask_keys = [k for k in state if "mask_decoder" in k or "mask_head" in k]
            assert mask_keys, "checkpoint missing mask head keys"

            trained = LibreYOLO(str(ckpt_path))
            assert trained.task == "segment"

            # 4. Inference produces masks on the original canvas.
            result = trained.predict(SAMPLE_IMAGE, conf=0.05)
            result = result[0] if isinstance(result, list) else result
            print(f"detections: {{0 if result.boxes is None else len(result.boxes.xyxy)}}")
            if result.masks is not None:
                print(f"mask shape: {{tuple(result.masks.data.shape)}}")

            # 5. Mask mAP is computed and finite — a sanity floor, not an
            # improvement check (2 epochs is too short to guarantee
            # improvement).
            post = trained.val(
                data={data_yaml_py}, split="test", batch=4, conf=0.001, iou=0.6
            )
            assert "metrics/mAP50-95(M)" in post, "validation did not return mask mAP"
            post_map = post["metrics/mAP50-95(M)"]
            assert post_map == post_map, "mask mAP is NaN"
            print(f"post-training mask mAP50-95(M): {{post_map:.4f}}")

            print("PASSED")

        if __name__ == "__main__":
            _main()
        """,
        timeout=1200,
    )
