"""E2E overfit gate for YOLO-NAS multi-class (shared-skeleton) pose (issue #484).

Fine-tunes ``yolonas-pose-n`` from the pretrained COCO pose checkpoint on a tiny
synthetic 2-class, 1-keypoint dataset (circle vs square, each with a single
center keypoint) and asserts the model:

1. writes a multi-class pose checkpoint (``nc=2`` + names + ``num_keypoints=1``)
   that reloads into a fresh model with an auto-rebuilt head, and
2. actually discriminates the two classes on the val split.

Follows the tier-0 overfit-gate precedent: a small, deterministic synthetic
dataset the model must clearly overfit, with an assertion on class
discrimination (not just mAP, which can pass with class collapse).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
import yaml

from .conftest import cuda_cleanup, requires_cuda

pytestmark = [pytest.mark.e2e, pytest.mark.yolonas, pytest.mark.slow]


def _make_dataset(root: Path, n_per_class: int = 40, size: int = 320, seed: int = 0) -> str:
    """Circle (class 0) vs square (class 1); each has one center keypoint.

    kpt_shape [1, 2] mirrors the meter-reading use case from issue #484.
    """
    rng = np.random.default_rng(seed)
    for split in ("train", "val"):
        img_dir = root / "images" / split
        lbl_dir = root / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        npc = n_per_class if split == "train" else max(6, n_per_class // 3)
        idx = 0
        for cls in (0, 1):
            for _ in range(npc):
                img = np.full((size, size, 3), 40, dtype=np.uint8)
                r = int(rng.integers(size // 8, size // 6))
                cx = int(rng.integers(r + 4, size - r - 4))
                cy = int(rng.integers(r + 4, size - r - 4))
                color = (0, 200, 255) if cls == 0 else (255, 120, 0)
                if cls == 0:
                    cv2.circle(img, (cx, cy), r, color, -1)
                else:
                    cv2.rectangle(img, (cx - r, cy - r), (cx + r, cy + r), color, -1)
                img = cv2.GaussianBlur(img, (3, 3), 0)
                name = f"{split}_{cls}_{idx:03d}"
                cv2.imwrite(str(img_dir / f"{name}.jpg"), img)
                bw = bh = (2 * r + 6) / size
                ncx, ncy = cx / size, cy / size
                (lbl_dir / f"{name}.txt").write_text(
                    f"{cls} {ncx:.6f} {ncy:.6f} {bw:.6f} {bh:.6f} {ncx:.6f} {ncy:.6f}\n"
                )
                idx += 1

    yaml_path = root / "data.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "path": str(root),
                "train": "images/train",
                "val": "images/val",
                "names": {0: "circle", 1: "square"},
                "nc": 2,
                "kpt_shape": [1, 2],
            }
        )
    )
    return str(yaml_path)


@requires_cuda
def test_yolonas_multiclass_pose_overfit(tmp_path):
    from libreyolo.models.yolonas.model import LibreYOLONAS

    data = _make_dataset(tmp_path)

    model = LibreYOLONAS("yolo_nas_pose_n_coco_pose.pth", size="n", task="pose")
    try:
        results = model.train(
            data=data,
            epochs=40,
            batch=16,
            imgsz=320,
            device="0",
            workers=0,
            seed=0,
            project=str(tmp_path / "runs"),
            name="mc_pose",
            exist_ok=True,
            optimizer="AdamW",
            lr0=2e-3,
        )
        best = results.get("best_checkpoint")
        assert best and Path(best).exists(), "no best checkpoint written"

        # 1. Checkpoint carries multi-class pose metadata.
        ckpt = torch.load(best, map_location="cpu", weights_only=False)
        assert ckpt["task"] == "pose"
        assert ckpt["nc"] == 2, f"expected nc=2, got {ckpt['nc']}"
        assert set(ckpt["names"].values()) == {"circle", "square"}
        assert ckpt["num_keypoints"] == 1, ckpt["num_keypoints"]
        assert ckpt["model_family"] == "yolonas"

        # 2. Reload into a fresh model: head auto-rebuilt to nc=2, K=1.
        del model
        cuda_cleanup()
        fresh = LibreYOLONAS(best, size="n", task="pose", device="0")
        assert fresh.nb_classes == 2
        assert fresh.num_keypoints == 1
        assert fresh.model.num_classes == 2

        # 3. Class discrimination on the val split (the point of the test:
        #    mAP alone can pass under class collapse).
        val_imgs = sorted((tmp_path / "images" / "val").glob("*.jpg"))
        correct = total = 0
        for img in val_imgs:
            true_cls = 0 if "_0_" in img.name else 1
            out = fresh.predict(str(img), conf=0.3)
            r0 = out[0] if isinstance(out, list) else out
            if len(r0.boxes) == 0:
                continue
            top = int(torch.as_tensor(r0.boxes.conf).argmax())
            pred_cls = int(r0.boxes.cls[top])
            total += 1
            correct += int(pred_cls == true_cls)
            # every detection also carries the single keypoint
            assert r0.keypoints is not None and r0.keypoints.xy.shape[-2] == 1

        assert total >= 0.5 * len(val_imgs), (
            f"model detected only {total}/{len(val_imgs)} val objects; "
            "fine-tune did not converge"
        )
        acc = correct / max(total, 1)
        assert acc >= 0.9, f"class-discrimination accuracy {acc:.2f} < 0.9"
    finally:
        cuda_cleanup()
