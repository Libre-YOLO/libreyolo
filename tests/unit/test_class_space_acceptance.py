"""Acceptance coverage for names-only detection dataset class spaces."""

from pathlib import Path

import pytest
import torch

from libreyolo.utils.serialization import load_trusted_torch_file

pytestmark = pytest.mark.unit


def _write_names_only_data_yaml(tmp_path: Path) -> Path:
    yaml_path = tmp_path / "data.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {tmp_path.as_posix()}",
                "train: train/images",
                "val: valid/images",
                "names: [cat, dog]",
            ]
        ),
        encoding="utf-8",
    )
    return yaml_path


def test_names_only_class_space_reaches_yolo9_head_loss_and_checkpoint(
    tmp_path, monkeypatch
):
    from libreyolo.models.yolo9.model import LibreYOLO9
    from libreyolo.models.yolo9.trainer import YOLO9Trainer

    data_yaml = _write_names_only_data_yaml(tmp_path)
    captured = {}

    class RecordingTrainer(YOLO9Trainer):
        def train(self):
            head = self.model.head
            loss = head._get_loss_fn(self.device)
            captured.update(
                head_nc=head.nc,
                head_outputs=[tower[-1].out_channels for tower in head.cv3],
                loss_nc=loss.num_classes,
            )

            # Exercise the real checkpoint writer without serializing the full
            # acceptance model a second time.
            self.model = torch.nn.Linear(head.cv3[0][-1].in_channels, head.nc)
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
            self.save_dir = tmp_path / "yolo9-run"
            self.save_dir.mkdir()
            self._save_checkpoint(epoch=0, loss=1.0, is_best=False)
            captured["checkpoint"] = self.save_dir / "weights" / "last.pt"
            return {}

    model = LibreYOLO9(model_path=None, size="t", device="cpu")
    monkeypatch.setattr(model, "_trainer_class", lambda: RecordingTrainer)
    monkeypatch.setattr(model, "_restore_after_training", lambda _result: None)

    model.train(
        data=str(data_yaml),
        epochs=1,
        batch=1,
        imgsz=64,
        device="cpu",
        amp=False,
    )

    checkpoint = load_trusted_torch_file(
        captured["checkpoint"], map_location="cpu", context="class-space acceptance"
    )
    assert model.nb_classes == 2
    assert model.names == {0: "cat", 1: "dog"}
    assert captured["head_nc"] == captured["loss_nc"] == 2
    assert captured["head_outputs"] == [2, 2, 2]
    assert checkpoint["nc"] == checkpoint["config"]["num_classes"] == 2
    assert checkpoint["names"] == {0: "cat", 1: "dog"}


def test_names_only_class_space_reaches_rfdetr_head_criterion_and_checkpoint(
    tmp_path, monkeypatch
):
    from libreyolo.models.rfdetr import model as rfdetr_model
    from libreyolo.models.rfdetr.trainer import RFDETRTrainer

    data_yaml = _write_names_only_data_yaml(tmp_path)
    captured = {}

    class RecordingTrainer(RFDETRTrainer):
        def train(self):
            self.on_setup()
            head = self.model.model.class_embed
            captured.update(
                head_outputs=head.out_features,
                criterion_nc=self.criterion.num_classes,
            )

            # The live head is sufficient to verify checkpoint class width and
            # keeps this no-download acceptance test compact.
            self.model = head
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
            self.save_dir = tmp_path / "rfdetr-run"
            self.save_dir.mkdir()
            self._save_checkpoint(epoch=0, loss=1.0, is_best=False)
            captured["checkpoint"] = self.save_dir / "weights" / "last.pt"
            return {}

    monkeypatch.setattr(rfdetr_model, "RFDETRTrainer", RecordingTrainer)
    model = rfdetr_model.LibreRFDETR(model_path={}, size="n", device="cpu")
    monkeypatch.setattr(model, "_restore_after_training", lambda _result: None)

    model.train(
        data=str(data_yaml),
        epochs=1,
        batch_size=1,
        imgsz=384,
        device="cpu",
        amp=False,
        ema=False,
        eval_interval=-1,
        output_dir=str(tmp_path / "requested-rfdetr-run"),
    )

    checkpoint = load_trusted_torch_file(
        captured["checkpoint"], map_location="cpu", context="class-space acceptance"
    )
    assert model.nb_classes == 2
    assert model.names == {0: "cat", 1: "dog"}
    # RF-DETR reserves one additional logit for the background/no-object class.
    assert captured["head_outputs"] == captured["criterion_nc"] == 3
    assert checkpoint["model"]["weight"].shape[0] == 3
    assert checkpoint["nc"] == checkpoint["config"]["num_classes"] == 2
    assert checkpoint["names"] == {0: "cat", 1: "dog"}
