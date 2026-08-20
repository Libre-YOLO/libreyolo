"""Behavior tests for the train command.

These verify observable CLI behavior (dry-run config resolution).
Real training is covered in e2e/test_rf1_training.py.
"""

import json
from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from libreyolo.cli.commands import train as train_module
from libreyolo.cli.commands.train import train_cmd
from libreyolo.cli.parsing import KeyValueCommand

pytestmark = pytest.mark.unit

runner = CliRunner()


def _make_app() -> typer.Typer:
    app = typer.Typer()
    app.command("train", cls=KeyValueCommand)(train_cmd)
    return app


def _write_vlm_checkpoint(directory: Path, *, size: str = "2b") -> Path:
    from libreyolo.models.vlm.qwen3vl import LibreQwen3VL

    directory.mkdir()
    contract = {
        "schema": 1,
        "family": "qwen3vl",
        "size": size,
        "base_repo": f"Qwen/Qwen3-VL-{size.upper()}-Instruct",
        "base_revision": LibreQwen3VL.HF_REVISIONS[size],
        "names": ["person"],
        "bbox_key": "bbox_2d",
        "coord_divisor": 1000.0,
        "box_format": "xyxy",
        "prompt": LibreQwen3VL._format_detection_prompt(None, "person"),
        "task": "detect",
    }
    (directory / "libreyolo_vlm.json").write_text(
        json.dumps(contract), encoding="utf-8"
    )
    (directory / "adapter_config.json").write_text(
        '{"peft_type":"LORA"}', encoding="utf-8"
    )
    (directory / "adapter_model.safetensors").write_bytes(b"adapter")
    return directory


def test_vlm_dry_run_uses_recipe_defaults_without_loading(monkeypatch):
    monkeypatch.setattr(
        "libreyolo.LibreVLM",
        lambda *_args, **_kwargs: pytest.fail("dry-run must not load VLM weights"),
    )

    result = runner.invoke(
        _make_app(),
        [
            "data=coco8.yaml",
            "model=qwen3-vl-2b",
            "--dry-run",
            "--json",
        ],
    )

    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert data["model_family"] == "qwen3vl"
    assert data["resolved_config"] == {
        "model": "qwen3-vl-2b",
        "data": "coco8.yaml",
        "family": "qwen3vl",
        "size": "2b",
        "task": "detect",
        "epochs": 10,
        "batch": 1,
        "accumulate": 8,
        "workers": 0,
        "seed": 0,
        "device": "auto",
        "lr0": 1e-4,
        "lora": True,
        "gradient_checkpointing": True,
        "vram_check": True,
        "hflip": 0.5,
        "project": "runs/vlm",
        "name": "train",
        "exist_ok": True,
        "resume": None,
        "allow_download_scripts": False,
    }


def test_vlm_dry_run_honors_compatible_overrides():
    result = runner.invoke(
        _make_app(),
        [
            "data=coco8.yaml",
            "model=qwen3-vl-4b",
            "epochs=2",
            "batch=2",
            "workers=1",
            "seed=9",
            "lr0=0.00002",
            "lora=false",
            "flip_prob=0.25",
            "project=runs/custom",
            "name=trial",
            "exist_ok=false",
            "--dry-run",
            "--json",
        ],
    )

    assert result.exit_code == 0
    config = json.loads(result.stdout)["resolved_config"]
    assert config["epochs"] == 2
    assert config["batch"] == 2
    assert config["workers"] == 1
    assert config["seed"] == 9
    assert config["lr0"] == 2e-5
    assert config["lora"] is False
    assert config["hflip"] == 0.25
    assert config["project"] == "runs/custom"
    assert config["name"] == "trial"
    assert config["exist_ok"] is False


def test_vlm_dry_run_accepts_required_true_flags_and_full_ft_defaults(
    monkeypatch,
):
    monkeypatch.setattr(
        train_module,
        "load_model_or_exit",
        lambda *_args, **_kwargs: pytest.fail("dry-run must not load VLM weights"),
    )

    result = runner.invoke(
        _make_app(),
        [
            "data=coco8.yaml",
            "model=qwen3-vl-2b",
            "task=detect",
            "pretrained=true",
            "val=true",
            "epochs=1",
            "batch=1",
            "workers=0",
            "flip_prob=0",
            "lora=false",
            "--dry-run",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    config = json.loads(result.stdout)["resolved_config"]
    assert config["task"] == "detect"
    assert config["epochs"] == 1
    assert config["batch"] == 1
    assert config["accumulate"] == 8
    assert config["device"] == "auto"
    assert config["gradient_checkpointing"] is True
    assert config["vram_check"] is True
    assert config["workers"] == 0
    assert config["hflip"] == 0.0
    assert config["lora"] is False
    assert config["lr0"] == 2e-5


@pytest.mark.parametrize(
    ("option", "message"),
    [
        ("epochs=0", "epochs must be an integer >= 1"),
        ("batch=0", "batch must be an integer >= 1"),
        ("workers=-1", "workers must be an integer >= 0"),
        ("flip_prob=-0.1", "flip_prob must be finite and within [0, 1]"),
        ("flip_prob=1.1", "flip_prob must be finite and within [0, 1]"),
        ("lr0=0", "lr0 must be a finite positive number"),
        ("device=not-a-device", "Invalid VLM training device"),
    ],
)
def test_vlm_dry_run_rejects_invalid_recipe_values_before_loading(
    option, message, monkeypatch
):
    monkeypatch.setattr(
        train_module,
        "load_model_or_exit",
        lambda *_args, **_kwargs: pytest.fail("invalid VLM config loaded weights"),
    )

    result = runner.invoke(
        _make_app(),
        [
            "data=coco8.yaml",
            "model=qwen3-vl-2b",
            option,
            "--dry-run",
            "--json",
        ],
    )

    assert result.exit_code != 0
    data = json.loads(result.stdout)
    assert data["error"] == "config_type_error"
    assert message in data["message"]


def test_vlm_rejects_non_bf16_cuda_before_loading(monkeypatch):
    import torch

    class DeviceContext:
        def __init__(self, _device):
            pass

        def __enter__(self):
            return None

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device", DeviceContext)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    monkeypatch.setattr(
        train_module,
        "load_model_or_exit",
        lambda *_args, **_kwargs: pytest.fail("unsupported GPU loaded VLM weights"),
    )

    result = runner.invoke(
        _make_app(),
        ["data=coco8.yaml", "model=qwen3-vl-2b", "device=auto", "--json"],
    )

    assert result.exit_code == 2, result.output
    payload = json.loads(result.stdout)
    assert payload["error"] == "config_unsupported"
    assert "BF16-capable" in payload["message"]


@pytest.mark.parametrize("data", ["", "missing-vlm-dataset.yaml"])
def test_vlm_rejects_invalid_dataset_reference_before_loading(data, monkeypatch):
    monkeypatch.setattr(
        train_module,
        "load_model_or_exit",
        lambda *_args, **_kwargs: pytest.fail("invalid dataset loaded VLM weights"),
    )

    result = runner.invoke(
        _make_app(),
        [f"data={data}", "model=qwen3-vl-2b", "--dry-run", "--json"],
    )

    payload = json.loads(result.stdout)
    assert result.exit_code in {2, 3}, result.output
    assert payload["error"] in {
        "config_type_error",
        "data_not_found",
    }


@pytest.mark.parametrize(
    ("yaml_text", "expected_error"),
    [
        ("names: [cat\ntrain: images/train\n", "config_type_error"),
        (
            "path: .\ntrain: images/train\n"
            "annotations:\n  train: annotations/train.json\n"
            "names: [cat]\n",
            "data_not_found",
        ),
        ("path: .\ntrain: missing/images\nnames: [cat]\n", "data_not_found"),
        (
            "path: .\ntrain: missing/images\nnames: [cat]\n"
            "download: |\n  print('not allowed')\n",
            "data_not_found",
        ),
        (
            "path: .\ntrain: missing/images\nnames: [cat]\ndownload: [bad]\n",
            "config_type_error",
        ),
        (
            "path: .\ntrain: images/train\nnc: 1\nnames: [cat, dog]\n",
            "config_type_error",
        ),
    ],
)
def test_vlm_dataset_contract_errors_are_structured_before_loading(
    yaml_text, expected_error, monkeypatch, tmp_path
):
    dataset = tmp_path / "invalid.yaml"
    dataset.write_text(yaml_text, encoding="utf-8")
    monkeypatch.setattr(
        train_module,
        "load_model_or_exit",
        lambda *_args, **_kwargs: pytest.fail("invalid dataset loaded VLM weights"),
    )

    result = runner.invoke(
        _make_app(),
        [f"data={dataset}", "model=qwen3-vl-2b", "--dry-run", "--json"],
    )

    assert result.exit_code != 0, result.output
    assert json.loads(result.stdout)["error"] == expected_error


def _write_native_coco_cli_dataset(tmp_path, *, annotation_text=None):
    from PIL import Image

    images = tmp_path / "images" / "train"
    annotations = tmp_path / "annotations"
    images.mkdir(parents=True)
    annotations.mkdir()
    Image.new("RGB", (32, 24), (20, 30, 40)).save(images / "sample.png")
    payload = {
        "images": [{"id": 1, "file_name": "sample.png", "width": 32, "height": 24}],
        "categories": [{"id": 5, "name": "cat"}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 5,
                "bbox": [2, 3, 10, 8],
                "area": 80,
                "iscrowd": 0,
            }
        ],
    }
    (annotations / "train.json").write_text(
        json.dumps(payload) if annotation_text is None else annotation_text,
        encoding="utf-8",
    )
    dataset = tmp_path / "native-coco.yaml"
    dataset.write_text(
        f"path: {tmp_path.as_posix()}\n"
        "train: images/train\n"
        "annotations:\n  train: annotations/train.json\n"
        "names: [cat]\n",
        encoding="utf-8",
    )
    return dataset


def test_vlm_native_coco_dry_run_validates_without_loading(monkeypatch, tmp_path):
    dataset = _write_native_coco_cli_dataset(tmp_path)
    monkeypatch.setattr(
        train_module,
        "load_model_or_exit",
        lambda *_args, **_kwargs: pytest.fail("dry-run loaded VLM weights"),
    )

    result = runner.invoke(
        _make_app(),
        [f"data={dataset}", "model=qwen3-vl-2b", "--dry-run", "--json"],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["valid"] is True


def test_vlm_malformed_native_coco_is_rejected_before_loading(monkeypatch, tmp_path):
    dataset = _write_native_coco_cli_dataset(tmp_path, annotation_text="{")
    monkeypatch.setattr(
        train_module,
        "load_model_or_exit",
        lambda *_args, **_kwargs: pytest.fail("invalid COCO loaded VLM weights"),
    )

    result = runner.invoke(
        _make_app(),
        [f"data={dataset}", "model=qwen3-vl-2b", "--dry-run", "--json"],
    )

    assert result.exit_code != 0, result.output
    assert json.loads(result.stdout)["error"] == "config_type_error"


@pytest.mark.parametrize("invalid", ["category", "bbox"])
def test_vlm_invalid_native_coco_annotations_are_rejected_before_loading(
    invalid, monkeypatch, tmp_path
):
    dataset = _write_native_coco_cli_dataset(tmp_path)
    annotation_file = tmp_path / "annotations" / "train.json"
    payload = json.loads(annotation_file.read_text(encoding="utf-8"))
    if invalid == "category":
        payload["annotations"][0]["category_id"] = 999
    else:
        payload["annotations"][0]["bbox"][0] = float("nan")
    annotation_file.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        train_module,
        "load_model_or_exit",
        lambda *_args, **_kwargs: pytest.fail("invalid COCO loaded VLM weights"),
    )

    result = runner.invoke(
        _make_app(),
        [f"data={dataset}", "model=qwen3-vl-2b", "--dry-run", "--json"],
    )

    assert result.exit_code != 0, result.output
    assert json.loads(result.stdout)["error"] == "config_type_error"


def test_vlm_orphan_coco_validation_annotations_are_rejected(monkeypatch, tmp_path):
    annotations = tmp_path / "annotations"
    annotations.mkdir()
    (annotations / "val.json").write_text("{}", encoding="utf-8")
    dataset = tmp_path / "orphan-val.yaml"
    dataset.write_text(
        f"path: {tmp_path.as_posix()}\ntrain: missing/train\n"
        "annotations:\n  val: annotations/val.json\n"
        "names: [cat]\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        train_module,
        "load_model_or_exit",
        lambda *_args, **_kwargs: pytest.fail("orphan COCO loaded VLM weights"),
    )

    result = runner.invoke(
        _make_app(),
        [f"data={dataset}", "model=qwen3-vl-2b", "--dry-run", "--json"],
    )

    assert result.exit_code != 0, result.output
    assert json.loads(result.stdout)["error"] == "config_type_error"


def test_vlm_downloadable_native_coco_may_defer_all_missing_artifacts(
    monkeypatch, tmp_path
):
    dataset = tmp_path / "downloadable.yaml"
    dataset.write_text(
        "path: missing\ntrain: images/train\n"
        "annotations:\n  train: annotations/train.json\n"
        "names: [cat]\ndownload: https://example.invalid/dataset.zip\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        train_module,
        "load_model_or_exit",
        lambda *_args, **_kwargs: pytest.fail("dry-run loaded VLM weights"),
    )

    result = runner.invoke(
        _make_app(),
        [f"data={dataset}", "model=qwen3-vl-2b", "--dry-run", "--json"],
    )

    assert result.exit_code == 0, result.output


def test_vlm_downloadable_native_coco_may_defer_when_only_annotation_exists(
    monkeypatch, tmp_path
):
    root = tmp_path / "partial"
    annotations = root / "annotations"
    annotations.mkdir(parents=True)
    (annotations / "train.json").write_text("{}", encoding="utf-8")
    dataset = tmp_path / "downloadable-annotation.yaml"
    dataset.write_text(
        f"path: {root.as_posix()}\ntrain: images/train\n"
        "annotations:\n  train: annotations/train.json\n"
        "names: [cat]\ndownload: https://example.invalid/dataset.zip\n",
        encoding="utf-8",
    )

    result = runner.invoke(
        _make_app(),
        [f"data={dataset}", "model=qwen3-vl-2b", "--dry-run", "--json"],
    )

    assert result.exit_code == 0, result.output


def test_vlm_downloadable_native_coco_rejects_images_without_annotation(
    monkeypatch, tmp_path
):
    from PIL import Image

    images = tmp_path / "images" / "train"
    images.mkdir(parents=True)
    Image.new("RGB", (16, 16)).save(images / "sample.png")
    dataset = tmp_path / "downloadable-images.yaml"
    dataset.write_text(
        f"path: {tmp_path.as_posix()}\ntrain: images/train\n"
        "annotations:\n  train: annotations/train.json\n"
        "names: [cat]\ndownload: https://example.invalid/dataset.zip\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        train_module,
        "load_model_or_exit",
        lambda *_args, **_kwargs: pytest.fail("partial COCO loaded VLM weights"),
    )

    result = runner.invoke(
        _make_app(),
        [f"data={dataset}", "model=qwen3-vl-2b", "--dry-run", "--json"],
    )

    assert result.exit_code != 0, result.output
    assert json.loads(result.stdout)["error"] == "data_not_found"


def test_vlm_missing_val_split_is_rejected_before_loading(monkeypatch, tmp_path):
    train_images = tmp_path / "images" / "train"
    train_images.mkdir(parents=True)
    (train_images / "sample.jpg").write_bytes(b"fixture")
    dataset = tmp_path / "partial.yaml"
    dataset.write_text(
        f"path: {tmp_path.as_posix()}\n"
        "train: images/train\nval: images/missing\nnames: [cat]\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        train_module,
        "load_model_or_exit",
        lambda *_args, **_kwargs: pytest.fail("partial dataset loaded VLM weights"),
    )

    result = runner.invoke(
        _make_app(),
        [f"data={dataset}", "model=qwen3-vl-2b", "--dry-run", "--json"],
    )

    assert result.exit_code != 0, result.output
    assert json.loads(result.stdout)["error"] == "data_not_found"


@pytest.mark.parametrize("broken_split", ["train", "val"])
def test_vlm_image_lists_are_checked_before_loading(
    broken_split, monkeypatch, tmp_path
):
    images = tmp_path / "images"
    images.mkdir()
    (images / "sample.jpg").write_bytes(b"fixture")
    valid_list = tmp_path / "valid.txt"
    valid_list.write_text("images/sample.jpg\n", encoding="utf-8")
    missing_list = tmp_path / "missing.txt"
    missing_list.write_text("images/does-not-exist.jpg\n", encoding="utf-8")
    train_list = missing_list if broken_split == "train" else valid_list
    val_list = missing_list if broken_split == "val" else valid_list
    dataset = tmp_path / "listed.yaml"
    dataset.write_text(
        f"path: {tmp_path.as_posix()}\n"
        f"train: {train_list.name}\n"
        f"val: {val_list.name}\n"
        "names: [cat]\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        train_module,
        "load_model_or_exit",
        lambda *_args, **_kwargs: pytest.fail("broken image list loaded VLM weights"),
    )

    result = runner.invoke(
        _make_app(),
        [f"data={dataset}", "model=qwen3-vl-2b", "--dry-run", "--json"],
    )

    assert result.exit_code != 0, result.output
    payload = json.loads(result.stdout)
    assert payload["error"] == "data_not_found"
    assert broken_split in payload["message"]
    assert "does-not-exist.jpg" in payload["message"]


def test_vlm_resume_is_preflighted_before_loading(tmp_path, monkeypatch):
    checkpoint = _write_vlm_checkpoint(tmp_path / "resume")
    monkeypatch.setattr(
        train_module,
        "load_model_or_exit",
        lambda *_args, **_kwargs: pytest.fail("dry-run must not load VLM weights"),
    )

    result = runner.invoke(
        _make_app(),
        [
            "data=coco8.yaml",
            "model=qwen3-vl-2b",
            f"resume={checkpoint}",
            "--dry-run",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["resolved_config"]["resume"] == str(checkpoint)


@pytest.mark.parametrize("resume", ["missing", "wrong-size"])
def test_vlm_invalid_resume_is_rejected_before_loading(resume, tmp_path, monkeypatch):
    checkpoint = tmp_path / "missing"
    if resume == "wrong-size":
        checkpoint = _write_vlm_checkpoint(tmp_path / "wrong-size", size="4b")
    monkeypatch.setattr(
        train_module,
        "load_model_or_exit",
        lambda *_args, **_kwargs: pytest.fail("invalid resume loaded VLM weights"),
    )

    result = runner.invoke(
        _make_app(),
        [
            "data=coco8.yaml",
            "model=qwen3-vl-2b",
            f"resume={checkpoint}",
            "--dry-run",
            "--json",
        ],
    )

    assert result.exit_code == 2, result.output
    data = json.loads(result.stdout)
    assert data["error"] == "config_unsupported"


def test_vlm_checkpoint_train_is_rejected_before_loading(tmp_path, monkeypatch):
    checkpoint = _write_vlm_checkpoint(tmp_path / "checkpoint")
    monkeypatch.setattr(
        train_module,
        "load_model_or_exit",
        lambda *_args, **_kwargs: pytest.fail("rejected VLM checkpoint loaded weights"),
    )

    result = runner.invoke(
        _make_app(),
        [
            "data=coco8.yaml",
            f"model={checkpoint}",
            "--dry-run",
            "--json",
        ],
    )

    assert result.exit_code == 2, result.output
    data = json.loads(result.stdout)
    assert data["error"] == "config_unsupported"
    assert "adapter is merged for inference" in data["message"]
    assert "resume=<checkpoint directory>" in data["suggestion"]


def test_vlm_checkpoint_unverified_size_is_rejected_before_loading(
    tmp_path, monkeypatch
):
    checkpoint = _write_vlm_checkpoint(tmp_path / "checkpoint-8b", size="8b")
    monkeypatch.setattr(
        train_module,
        "load_model_or_exit",
        lambda *_args, **_kwargs: pytest.fail("unsupported VLM must not load"),
    )

    result = runner.invoke(
        _make_app(),
        ["data=coco8.yaml", f"model={checkpoint}", "--json"],
    )

    assert result.exit_code == 2, result.output
    data = json.loads(result.stdout)
    assert data["error"] == "config_unsupported"
    assert "Verified sizes: 2b, 4b" in data["message"]


@pytest.mark.parametrize(
    ("model", "message"),
    [
        ("qwen3-vl-8b", "Verified sizes: 2b, 4b"),
        ("smolvlm2-500m", "not supported for family 'smolvlm2'"),
    ],
)
def test_vlm_dry_run_rejects_unverified_training_before_loading(
    model, message, monkeypatch
):
    monkeypatch.setattr(
        "libreyolo.LibreVLM",
        lambda *_args, **_kwargs: pytest.fail("unsupported VLM must not load"),
    )

    result = runner.invoke(
        _make_app(),
        ["data=coco8.yaml", f"model={model}", "--dry-run", "--json"],
    )

    assert result.exit_code != 0
    data = json.loads(result.stdout)
    assert data["error"] == "config_unsupported"
    assert message in data["message"]


def test_vlm_train_rejects_explicit_detector_only_option():
    result = runner.invoke(
        _make_app(),
        [
            "data=coco8.yaml",
            "model=qwen3-vl-2b",
            "optimizer=adamw",
            "--dry-run",
            "--json",
        ],
    )

    assert result.exit_code != 0
    data = json.loads(result.stdout)
    assert data["error"] == "config_unsupported"
    assert "optimizer" in data["message"]


@pytest.mark.parametrize(
    ("option", "message"),
    [
        ("pretrained=false", "pretrained=false"),
        ("val=false", "val=false"),
        ("task=segment", "task='detect'"),
    ],
)
def test_vlm_train_rejects_unsupported_contract_options_before_loading(
    option, message, monkeypatch
):
    monkeypatch.setattr(
        train_module,
        "load_model_or_exit",
        lambda *_args, **_kwargs: pytest.fail("rejected VLM request loaded weights"),
    )

    result = runner.invoke(
        _make_app(),
        ["data=coco8.yaml", "model=qwen3-vl-2b", option, "--json"],
    )

    assert result.exit_code == 2, result.output
    data = json.loads(result.stdout)
    assert data["error"] == "config_unsupported"
    assert message in data["message"]


def test_vlm_train_routes_only_vlm_kwargs_and_reports_loss_truthfully(
    monkeypatch, tmp_path
):
    from libreyolo.models.vlm.training import trainer as trainer_module

    captured = {}
    images = tmp_path / "images" / "train"
    images.mkdir(parents=True)
    (images / "sample.jpg").write_bytes(b"fixture")
    dataset = tmp_path / "dataset.yaml"
    dataset.write_text(
        f"path: {tmp_path.as_posix()}\ntrain: images/train\nnames: [cat]\n",
        encoding="utf-8",
    )

    class FakeVLM:
        FAMILY = "qwen3vl"
        device = "cpu"

        def train(self, *, data, **kwargs):
            captured["data"] = data
            captured["kwargs"] = kwargs
            return {
                "save_dir": "runs/vlm/train",
                "best": "runs/vlm/train/weights/best",
                "last": "runs/vlm/train/weights/last",
                "epochs": 2,
                "final_loss": 0.4,
                "best_metric": 0.35,
                "best_epoch": 2,
                "metric_name": "val/loss",
            }

    monkeypatch.setattr(
        train_module,
        "load_model_or_exit",
        lambda *args, **kwargs: FakeVLM(),
    )
    monkeypatch.setattr(trainer_module, "require_vlm_lora_dependencies", lambda: None)

    result = runner.invoke(
        _make_app(),
        [
            f"data={dataset}",
            "model=qwen3-vl-2b",
            "epochs=2",
            "allow_download_scripts=true",
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert captured["data"] == str(dataset)
    assert captured["kwargs"] == {
        "epochs": 2,
        "batch": 1,
        "accumulate": 8,
        "lr0": None,
        "lora": True,
        "project": "runs/vlm",
        "name": "train",
        "exist_ok": True,
        "workers": 0,
        "seed": 0,
        "device": "auto",
        "gradient_checkpointing": True,
        "vram_check": True,
        "resume": None,
        "hflip": 0.5,
        "allow_download_scripts": True,
    }
    data = json.loads(result.stdout)
    assert data["metric_name"] == "val/loss"
    assert data["best_metric"] == 0.35
    assert data["best"] == "runs/vlm/train/weights/best"
    assert data["last"] == "runs/vlm/train/weights/last"
    assert "best_metrics" not in data


def test_vlm_lora_dependency_failure_precedes_dataset_preparation(monkeypatch):
    from libreyolo.models.vlm.training import trainer as trainer_module

    events = []
    monkeypatch.setattr(
        train_module,
        "_preflight_vlm_dataset",
        lambda *_args, prepare, **_kwargs: events.append(f"dataset:{prepare}"),
    )

    def missing_dependency():
        events.append("dependency")
        raise ImportError("PEFT unavailable")

    monkeypatch.setattr(
        trainer_module, "require_vlm_lora_dependencies", missing_dependency
    )
    monkeypatch.setattr(
        train_module,
        "load_model_or_exit",
        lambda *_args, **_kwargs: pytest.fail("dependency failure loaded VLM weights"),
    )

    result = runner.invoke(
        _make_app(),
        ["data=coco8.yaml", "model=qwen3-vl-2b", "--json"],
    )

    assert result.exit_code != 0, result.output
    assert json.loads(result.stdout)["error"] == "config_unsupported"
    assert events == ["dataset:False", "dependency"]


def test_vlm_download_script_warning_precedes_dataset_preparation(monkeypatch):
    from libreyolo.cli.output import OutputHandler
    from libreyolo.models.vlm.training import trainer as trainer_module

    events = []
    monkeypatch.setattr(
        train_module,
        "_preflight_vlm_dataset",
        lambda *_args, prepare, **_kwargs: events.append(f"dataset:{prepare}"),
    )
    monkeypatch.setattr(
        trainer_module,
        "require_vlm_lora_dependencies",
        lambda: events.append("dependency"),
    )
    monkeypatch.setattr(
        OutputHandler,
        "warning",
        lambda _self, _message: events.append("warning"),
    )

    class FakeVLM:
        device = "cpu"

        def train(self, **_kwargs):
            events.append("train")
            return {"epochs": 1, "final_loss": 0.5}

    def load_model(*_args, **_kwargs):
        events.append("load")
        return FakeVLM()

    monkeypatch.setattr(train_module, "load_model_or_exit", load_model)

    result = runner.invoke(
        _make_app(),
        [
            "data=coco8.yaml",
            "model=qwen3-vl-2b",
            "allow_download_scripts=true",
            "epochs=1",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert events[:5] == [
        "dataset:False",
        "dependency",
        "warning",
        "dataset:True",
        "load",
    ]


def test_train_dry_run_uses_rtdetr_defaults():
    """Dry-run shows correct family-specific defaults for RT-DETR."""
    app = _make_app()
    result = runner.invoke(
        app,
        [
            "data=coco8.yaml",
            "model=rtdetr-r18",
            "--dry-run",
            "--json",
        ],
    )

    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert data["model_family"] == "rtdetr"
    assert data["resolved_config"]["epochs"] == 72
    assert data["resolved_config"]["batch"] == 4
    assert data["resolved_config"]["optimizer"] == "adamw"
    assert data["resolved_config"]["lr0"] == 0.0001
    assert data["resolved_config"]["scheduler"] == "constant"


def test_train_dry_run_uses_rtdetr_defaults_for_weight_filename():
    """Dry-run detects family defaults from supported weight filenames."""
    app = _make_app()
    result = runner.invoke(
        app,
        [
            "data=coco8.yaml",
            "model=LibreRTDETRr18.pt",
            "--dry-run",
            "--json",
        ],
    )

    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert data["model_family"] == "rtdetr"
    assert data["resolved_config"]["epochs"] == 72
    assert data["resolved_config"]["batch"] == 4
    assert data["resolved_config"]["optimizer"] == "adamw"
    assert data["resolved_config"]["lr0"] == 0.0001
    assert data["resolved_config"]["scheduler"] == "constant"


def test_train_dry_run_uses_rfdetr_defaults():
    """Dry-run shows native RF-DETR defaults instead of generic YOLO defaults."""
    app = _make_app()
    result = runner.invoke(
        app,
        [
            "data=coco8.yaml",
            "model=rfdetr-m",
            "--dry-run",
            "--json",
        ],
    )

    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert data["model_family"] == "rfdetr"
    cfg = data["resolved_config"]
    assert cfg["epochs"] == 100
    assert cfg["batch"] == 4
    assert cfg["lr0"] == 0.0001
    assert cfg["workers"] == 0
    assert cfg["weight_decay"] == 0.0001
    assert cfg["eval_interval"] == 1
    assert cfg["warmup_epochs"] == 0
    assert cfg["lr_drop"] == 100
    assert cfg["ema_decay"] == 0.993
    assert cfg["amp_dtype"] == "float16"
    assert cfg["max_det"] == 300
    assert "eval_max_det" not in cfg
    from libreyolo.models.rfdetr.config import RFDETRConfig

    assert RFDETRConfig().ema_tau == 100
    assert "optimizer" not in cfg
    assert "scheduler" not in cfg


def test_train_dry_run_rfdetr_user_override_wins():
    app = _make_app()
    result = runner.invoke(
        app,
        [
            "data=coco8.yaml",
            "model=LibreRFDETRm.pt",
            "epochs=3",
            "batch=2",
            "lr0=0.001",
            "lr_drop=7",
            "amp_dtype=bf16",
            "max_det=500",
            "eval_max_det=500",
            "--dry-run",
            "--json",
        ],
    )

    assert result.exit_code == 0
    data = json.loads(result.stdout)
    cfg = data["resolved_config"]
    assert cfg["epochs"] == 3
    assert cfg["batch"] == 2
    assert cfg["lr0"] == 0.001
    assert cfg["lr_drop"] == 7
    assert cfg["amp_dtype"] == "bfloat16"
    assert cfg["max_det"] == 500
    assert cfg["eval_max_det"] == 500


def test_train_dry_run_rejects_invalid_max_det():
    app = _make_app()
    result = runner.invoke(
        app,
        [
            "data=coco8.yaml",
            "model=LibreYOLO9t.pt",
            "max_det=0",
            "--dry-run",
            "--json",
        ],
    )

    assert result.exit_code == 2
    data = json.loads(result.stdout)
    assert data["error"] == "config_type_error"
    assert "max_det must be >= 1" in data["message"]


def test_train_dry_run_rejects_invalid_eval_max_det():
    app = _make_app()
    result = runner.invoke(
        app,
        [
            "data=coco8.yaml",
            "model=LibreYOLO9t.pt",
            "eval_max_det=0",
            "--dry-run",
            "--json",
        ],
    )

    assert result.exit_code == 2
    data = json.loads(result.stdout)
    assert data["error"] == "config_type_error"
    assert "eval_max_det must be >= 1" in data["message"]


def test_train_dry_run_rfdetr_lora_flag_is_visible():
    app = _make_app()
    result = runner.invoke(
        app,
        [
            "data=coco8.yaml",
            "model=LibreRFDETRm.pt",
            "--lora",
            "--dry-run",
            "--json",
        ],
    )

    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert data["model_family"] == "rfdetr"
    assert data["resolved_config"]["lora"] is True


def test_train_dry_run_rfdetr_freeze_flag_is_visible():
    app = _make_app()
    result = runner.invoke(
        app,
        [
            "data=coco8.yaml",
            "model=LibreRFDETRm.pt",
            "--freeze",
            "backbone",
            "--dry-run",
            "--json",
        ],
    )

    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert data["model_family"] == "rfdetr"
    assert data["resolved_config"]["freeze"] == "backbone"


def test_train_dry_run_rejects_ambiguous_freeze_true():
    app = _make_app()
    result = runner.invoke(
        app,
        [
            "data=coco8.yaml",
            "model=LibreYOLO9t.pt",
            "--freeze",
            "true",
            "--dry-run",
            "--json",
        ],
    )

    assert result.exit_code == 2
    data = json.loads(result.stdout)
    assert data["error"] == "config_type_error"
    assert "freeze=True is ambiguous" in data["message"]


@pytest.mark.parametrize("model_name", ["LibreDFINEs.pt"])
def test_train_dry_run_rejects_class_balanced_on_custom_loaders(model_name):
    app = _make_app()
    result = runner.invoke(
        app,
        [
            "data=coco8.yaml",
            f"model={model_name}",
            "class_balanced=true",
            "--dry-run",
            "--json",
        ],
    )
    assert result.exit_code == 2, result.stdout
    data = json.loads(result.stdout)
    assert data["error"] == "config_unsupported"
    assert "class_balanced" in data["message"]


def test_train_dry_run_accepts_class_balanced_for_rfdetr_detection(monkeypatch):
    monkeypatch.setattr(
        "libreyolo.cli.commands.train._create_explicit_task_train_model",
        lambda **_kwargs: None,
    )
    app = _make_app()
    result = runner.invoke(
        app,
        [
            "data=coco8.yaml",
            "model=rfdetr-n",
            "task=detect",
            "class_balanced=true",
            "--dry-run",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.stdout
    config = json.loads(result.stdout)["resolved_config"]
    assert config["class_balanced"] is True


def test_train_dry_run_optin_helpers_both_grammars():
    """New #768 knobs parse in both CLI grammars and stay off by default."""
    app = _make_app()

    default = runner.invoke(
        app,
        ["data=coco8.yaml", "model=LibreYOLO9t.pt", "--dry-run", "--json"],
    )
    assert default.exit_code == 0, default.stdout
    default_cfg = json.loads(default.stdout)["resolved_config"]
    assert default_cfg["class_balanced"] is False
    assert default_cfg["average_best"] == 0
    assert default_cfg["export_check"] is False
    assert default_cfg["precise_bn"] == 0

    kv = runner.invoke(
        app,
        [
            "data=coco8.yaml",
            "model=LibreYOLO9t.pt",
            "class_balanced=true",
            "average_best=5",
            "export_check=true",
            "precise_bn=128",
            "--dry-run",
            "--json",
        ],
    )
    assert kv.exit_code == 0, kv.stdout
    kv_cfg = json.loads(kv.stdout)["resolved_config"]
    assert kv_cfg["class_balanced"] is True
    assert kv_cfg["average_best"] == 5
    assert kv_cfg["export_check"] is True
    assert kv_cfg["precise_bn"] == 128

    dashed = runner.invoke(
        app,
        [
            "data=coco8.yaml",
            "model=LibreYOLO9t.pt",
            "--class-balanced",
            "--average-best",
            "5",
            "--export-check",
            "--precise-bn",
            "128",
            "--dry-run",
            "--json",
        ],
    )
    assert dashed.exit_code == 0, dashed.stdout
    dashed_cfg = json.loads(dashed.stdout)["resolved_config"]
    assert dashed_cfg["class_balanced"] is True
    assert dashed_cfg["average_best"] == 5
    assert dashed_cfg["export_check"] is True
    assert dashed_cfg["precise_bn"] == 128


def test_train_dry_run_distill_model_is_visible():
    """A distillation teacher resolves into the config without error."""
    app = _make_app()
    result = runner.invoke(
        app,
        [
            "data=coco8.yaml",
            "model=LibreYOLO9t.pt",
            "distill_model=LibreYOLO9m.pt",
            "dis=2.0",
            "--dry-run",
            "--json",
        ],
    )

    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert data["resolved_config"]["distill_model"] == "LibreYOLO9m.pt"
    assert data["resolved_config"]["dis"] == 2.0


def test_train_dry_run_accepts_lora_for_dfine_and_deim():
    app = _make_app()
    for model_name, family in (
        ("LibreDFINEs.pt", "dfine"),
        ("LibreDEIMs.pt", "deim"),
        ("LibreDEIMv2s.pt", "deimv2"),
        ("LibreRTDETRr18.pt", "rtdetr"),
        ("LibreRTDETRv2r18.pt", "rtdetrv2"),
        ("LibreRTDETRv4s.pt", "rtdetrv4"),
        ("LibreECs.pt", "ec"),
    ):
        result = runner.invoke(
            app,
            [
                "data=coco8.yaml",
                f"model={model_name}",
                "--lora",
                "--dry-run",
                "--json",
            ],
        )

        assert result.exit_code == 0, result.stdout
        data = json.loads(result.stdout)
        assert data["model_family"] == family
        assert data["resolved_config"]["lora"] is True


def test_train_dry_run_rejects_lora_for_unsupported_family():
    app = _make_app()
    result = runner.invoke(
        app,
        [
            "data=coco8.yaml",
            "model=LibreYOLO9t.pt",
            "--lora",
            "--dry-run",
            "--json",
        ],
    )

    assert result.exit_code == 2
    data = json.loads(result.stdout)
    assert data["error"] == "config_unsupported"
    assert "not supported for yolo9" in data["message"]


def test_train_rfdetr_actual_call_uses_reported_defaults(monkeypatch, tmp_path):
    """RF-DETR train should receive the same defaults shown by dry-run."""
    app = _make_app()
    captured = {}

    class _RFDETRLike:
        FAMILY = "rfdetr"
        device = "cpu"

        def train(self, data, **kwargs):
            captured["data"] = data
            captured["kwargs"] = kwargs
            return {"output_dir": str(tmp_path / "rfdetr_exp")}

    monkeypatch.setattr(
        "libreyolo.cli.commands.train.load_model_or_exit",
        lambda out, model, model_path, device: _RFDETRLike(),
    )

    result = runner.invoke(
        app,
        [
            "data=dummy.yaml",
            "model=LibreRFDETRm.pt",
            f"project={tmp_path}",
            "exist_ok=true",
            "save_plots=true",
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert captured["data"] == "dummy.yaml"
    kwargs = captured["kwargs"]
    assert kwargs["epochs"] == 100
    assert kwargs["batch"] == 4
    assert kwargs["lr0"] == 0.0001
    assert kwargs["num_workers"] == 0
    assert kwargs["weight_decay"] == 0.0001
    assert kwargs["eval_interval"] == 1
    assert kwargs["warmup_epochs"] == 0
    assert kwargs["scheduler"] == "step"
    assert kwargs["lr_drop"] == 100
    assert kwargs["use_ema"] is True
    assert kwargs["ema_decay"] == 0.993
    assert kwargs["amp_dtype"] == "float16"
    assert kwargs["max_det"] == 300
    assert kwargs["eval_max_det"] is None
    assert kwargs["save_plots"] is True
    assert kwargs["early_stopping"] is False

    data = json.loads(result.stdout)
    assert data["model_family"] == "rfdetr"
    assert data["epochs_completed"] == 100


def test_train_rfdetr_scheduler_override_reaches_trainer(monkeypatch, tmp_path):
    app = _make_app()
    captured = {}

    class _RFDETRLike:
        FAMILY = "rfdetr"
        device = "cpu"

        def train(self, data, **kwargs):
            captured["kwargs"] = kwargs
            return {"output_dir": str(tmp_path / "rfdetr_exp")}

    monkeypatch.setattr(
        "libreyolo.cli.commands.train.load_model_or_exit",
        lambda out, model, model_path, device: _RFDETRLike(),
    )

    result = runner.invoke(
        app,
        [
            "data=dummy.yaml",
            "model=LibreRFDETRm.pt",
            "scheduler=cosine",
            f"project={tmp_path}",
            "exist_ok=true",
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert captured["kwargs"]["scheduler"] == "cosine"
    assert "ignores these parameters" not in result.output


def test_train_rfdetr_lora_flag_reaches_trainer(monkeypatch, tmp_path):
    app = _make_app()
    captured = {}

    class _RFDETRLike:
        FAMILY = "rfdetr"
        device = "cpu"

        def train(self, data, **kwargs):
            captured["kwargs"] = kwargs
            return {"output_dir": str(tmp_path / "rfdetr_exp")}

    monkeypatch.setattr(
        "libreyolo.cli.commands.train.load_model_or_exit",
        lambda out, model, model_path, device: _RFDETRLike(),
    )

    result = runner.invoke(
        app,
        [
            "data=dummy.yaml",
            "model=LibreRFDETRm.pt",
            "--lora",
            f"project={tmp_path}",
            "exist_ok=true",
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert captured["kwargs"]["lora"] is True
    assert "ignores these parameters" not in result.output


def test_train_rfdetr_lr_drop_override_reaches_trainer(monkeypatch, tmp_path):
    app = _make_app()
    captured = {}

    class _RFDETRLike:
        FAMILY = "rfdetr"
        device = "cpu"

        def train(self, data, **kwargs):
            captured["kwargs"] = kwargs
            return {"output_dir": str(tmp_path / "rfdetr_exp")}

    monkeypatch.setattr(
        "libreyolo.cli.commands.train.load_model_or_exit",
        lambda out, model, model_path, device: _RFDETRLike(),
    )

    result = runner.invoke(
        app,
        [
            "data=dummy.yaml",
            "model=LibreRFDETRm.pt",
            "lr_drop=12",
            f"project={tmp_path}",
            "exist_ok=true",
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert captured["kwargs"]["lr_drop"] == 12
    assert "ignores these parameters" not in result.output


def test_train_rfdetr_obb_uses_task_architecture_without_generic_load(
    monkeypatch, tmp_path
):
    app = _make_app()
    captured = {}

    class _RFDETROBBLike:
        FAMILY = "rfdetr"
        device = "cpu"

        def __init__(
            self,
            model_path=None,
            size=None,
            task=None,
            device="auto",
            allow_detect_to_obb_transfer=False,
        ):
            captured["init"] = {
                "model_path": model_path,
                "size": size,
                "task": task,
                "device": device,
                "allow_detect_to_obb_transfer": allow_detect_to_obb_transfer,
            }
            self.size = size
            self.task = task
            self.device = device

        @classmethod
        def detect_task_from_filename(cls, filename):
            return "obb" if filename.lower().endswith("-obb.pt") else None

        @classmethod
        def detect_size_from_filename(cls, filename):
            return "n" if "rfdetrn" in filename.lower() else None

        def train(self, data, **kwargs):
            captured["data"] = data
            captured["kwargs"] = kwargs
            return {"output_dir": str(tmp_path / "rfdetr_obb_exp")}

    def fail_load(*_args, **_kwargs):
        raise AssertionError(
            "RF-DETR OBB training should instantiate the task architecture"
        )

    import libreyolo.models.rfdetr.model as rfdetr_model

    monkeypatch.setattr("libreyolo.cli.commands.train.load_model_or_exit", fail_load)
    monkeypatch.setattr(
        "libreyolo.cli.commands.train._model_ref_exists", lambda _: False
    )
    monkeypatch.setattr(rfdetr_model, "LibreRFDETR", _RFDETROBBLike)

    result = runner.invoke(
        app,
        [
            "data=uav-obb.yaml",
            "model=LibreRFDETRn.pt",
            "task=obb",
            "epochs=1",
            "pretrained=true",
            f"project={tmp_path}",
            "exist_ok=true",
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert captured["init"] == {
        "model_path": None,
        "size": "n",
        "task": "obb",
        "device": "auto",
        "allow_detect_to_obb_transfer": True,
    }
    assert captured["data"] == "uav-obb.yaml"
    assert "pretrained" not in captured["kwargs"]
    data = json.loads(result.stdout)
    assert data["model_family"] == "rfdetr"
    assert data["epochs_completed"] == 1


def test_train_rfdetr_pose_uses_explicit_detect_transfer_flag(monkeypatch, tmp_path):
    app = _make_app()
    captured = {}

    class _RFDETRPoseLike:
        FAMILY = "rfdetr"
        device = "cpu"

        def __init__(
            self,
            model_path=None,
            size=None,
            task=None,
            device="auto",
            allow_detect_to_obb_transfer=False,
            allow_detect_to_pose_transfer=False,
        ):
            captured["init"] = {
                "model_path": model_path,
                "size": size,
                "task": task,
                "device": device,
                "allow_detect_to_obb_transfer": allow_detect_to_obb_transfer,
                "allow_detect_to_pose_transfer": allow_detect_to_pose_transfer,
            }
            self.size = size
            self.task = task
            self.device = device

        @classmethod
        def detect_task_from_filename(cls, filename):
            return "pose" if filename.lower().endswith("-pose.pt") else None

        @classmethod
        def detect_size_from_filename(cls, filename):
            return "n" if "rfdetrn" in filename.lower() else None

        def train(self, data, **kwargs):
            captured["data"] = data
            captured["kwargs"] = kwargs
            return {"output_dir": str(tmp_path / "rfdetr_pose_exp")}

    def fail_load(*_args, **_kwargs):
        raise AssertionError(
            "RF-DETR pose training should instantiate the task architecture"
        )

    import libreyolo.models.rfdetr.model as rfdetr_model

    monkeypatch.setattr("libreyolo.cli.commands.train.load_model_or_exit", fail_load)
    monkeypatch.setattr(
        "libreyolo.cli.commands.train._model_ref_exists", lambda _: True
    )
    monkeypatch.setattr(rfdetr_model, "LibreRFDETR", _RFDETRPoseLike)

    result = runner.invoke(
        app,
        [
            "data=coco-pose.yaml",
            "model=LibreRFDETRn.pt",
            "task=pose",
            "epochs=1",
            "pretrained=true",
            f"project={tmp_path}",
            "exist_ok=true",
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert captured["init"] == {
        "model_path": "LibreRFDETRn.pt",
        "size": "n",
        "task": "pose",
        "device": "auto",
        "allow_detect_to_obb_transfer": False,
        "allow_detect_to_pose_transfer": True,
    }
    assert captured["data"] == "coco-pose.yaml"
    assert "pretrained" not in captured["kwargs"]
    data = json.loads(result.stdout)
    assert data["model_family"] == "rfdetr"
    assert data["epochs_completed"] == 1


def test_train_rfdetr_detect_checkpoint_switches_to_obb_architecture(
    monkeypatch, tmp_path
):
    app = _make_app()
    detect_path = tmp_path / "custom-rfdetr.pt"
    detect_path.write_bytes(b"placeholder")
    captured = {}

    class _LoadedRFDETRDetect:
        FAMILY = "rfdetr"
        task = "detect"
        size = "n"
        device = "cpu"

    class _RFDETROBBLike:
        FAMILY = "rfdetr"
        device = "cpu"

        def __init__(
            self,
            model_path=None,
            size=None,
            task=None,
            device="auto",
            allow_detect_to_obb_transfer=False,
        ):
            captured["init"] = {
                "model_path": model_path,
                "size": size,
                "task": task,
                "device": device,
                "allow_detect_to_obb_transfer": allow_detect_to_obb_transfer,
            }
            self.size = size
            self.task = task
            self.device = device

        def train(self, data, **kwargs):
            captured["data"] = data
            captured["kwargs"] = kwargs
            return {"output_dir": str(tmp_path / "rfdetr_obb_custom_transfer")}

    import libreyolo.models.rfdetr.model as rfdetr_model

    monkeypatch.setattr(
        "libreyolo.cli.commands.train.load_model_or_exit",
        lambda out, model, model_path, device: _LoadedRFDETRDetect(),
    )
    monkeypatch.setattr(rfdetr_model, "LibreRFDETR", _RFDETROBBLike)

    result = runner.invoke(
        app,
        [
            "data=uav-obb.yaml",
            f"model={detect_path}",
            "task=obb",
            "epochs=1",
            "pretrained=true",
            f"project={tmp_path}",
            "exist_ok=true",
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert captured["init"] == {
        "model_path": str(detect_path),
        "size": "n",
        "task": "obb",
        "device": "auto",
        "allow_detect_to_obb_transfer": True,
    }
    assert captured["data"] == "uav-obb.yaml"
    assert "pretrained" not in captured["kwargs"]
    data = json.loads(result.stdout)
    assert data["model_family"] == "rfdetr"
    assert data["epochs_completed"] == 1
