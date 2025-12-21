import sys
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.integration


def setup_module(module):
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root / "temporary"))


MODELS = [
    ("8", "n"),
    ("8", "s"),
    ("8", "m"),
    ("8", "l"),
    ("8", "x"),
    ("11", "n"),
    ("11", "s"),
    ("11", "m"),
    ("11", "l"),
    ("11", "x"),
]


@pytest.mark.parametrize("version, size", MODELS)
def test_convert_weights(version, size, temp_weights_dir, weights_dir):
    if version == "8":
        try:
            from convert_yolo8_weights import convert_weights
        except ImportError:
            pytest.skip(
                "Could not import convert_yolo8_weights. Make sure conversion scripts are in temporary/ directory."
            )

        source_name = f"yolov8{size}.pt"
        dest_name = f"libreyolo8{size}.pt"
    else:
        try:
            from convert_yolo11_weights import convert_weights
        except ImportError:
            pytest.skip(
                "Could not import convert_yolo11_weights. Make sure conversion scripts are in temporary/ directory."
            )

        source_name = f"yolo11{size}.pt"
        dest_name = f"libreyolo11{size}.pt"

    source_path = temp_weights_dir / source_name
    dest_path = weights_dir / dest_name

    if not source_path.exists():
        pytest.skip(f"Source weight {source_name} not found. Run download script first.")

    convert_weights(str(source_path), str(dest_path))

    assert dest_path.exists(), "Converted file was not created"

    try:
        state_dict = torch.load(dest_path, map_location="cpu", weights_only=True)
    except Exception as e:
        pytest.fail(f"Failed to load converted weights: {e}")

    assert len(state_dict) > 0, "State dict is empty"
    assert any(k.startswith("backbone") for k in state_dict), "Missing backbone keys"
    assert any(k.startswith("head") for k in state_dict), "Missing head keys"

