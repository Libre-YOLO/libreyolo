"""Unit tests for LibreFOMO core architecture and registration."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

pytestmark = [pytest.mark.unit, pytest.mark.fomo]


# ===========================================================================
# Helpers
# ===========================================================================


def _make_random_fomo(size: str = "s", nc: int = 1) -> "LibreFOMO":
    from libreyolo.models.fomo.model import LibreFOMO

    return LibreFOMO(model_path=None, size=size, nb_classes=nc, device="cpu")


# ===========================================================================
# nn.py — architecture contracts
# ===========================================================================


class TestFOMOBackboneShapes:
    """Verify spatial resolution contracts for all three sizes."""

    @pytest.mark.parametrize(
        "size,imgsz,expected_hw",
        [
            ("s", 96,  12),  # 96  / 8 = 12
            ("m", 192, 24),  # 192 / 8 = 24
            ("l", 224, 28),  # 224 / 8 = 28
        ],
    )
    def test_backbone_spatial_downsample(self, size: str, imgsz: int, expected_hw: int) -> None:
        from libreyolo.models.fomo.nn import FOMOBackbone

        backbone = FOMOBackbone(size).eval()
        x = torch.zeros(1, 3, imgsz, imgsz)
        with torch.no_grad():
            out = backbone(x)
        assert out.shape[-2] == expected_hw, f"size={size}: expected H={expected_hw}, got {out.shape[-2]}"
        assert out.shape[-1] == expected_hw, f"size={size}: expected W={expected_hw}, got {out.shape[-1]}"

    @pytest.mark.parametrize("size", ["s", "m", "l"])
    @pytest.mark.parametrize("nc", [1, 3])
    def test_model_output_channels(self, size: str, nc: int) -> None:
        from libreyolo.models.fomo.nn import LibreFOMOModel

        model = LibreFOMOModel(size=size, nc=nc).eval()
        cfg = model.CONFIGS[size]
        x = torch.zeros(1, 3, cfg["imgsz"], cfg["imgsz"])
        with torch.no_grad():
            out = model(x)
        assert out.shape[1] == nc + 1, f"Expected {nc+1} channels, got {out.shape[1]}"

    @pytest.mark.parametrize("size", ["s", "m", "l"])
    def test_model_batch_dimension_preserved(self, size: str) -> None:
        from libreyolo.models.fomo.nn import LibreFOMOModel, CONFIGS

        model = LibreFOMOModel(size=size, nc=1).eval()
        imgsz = CONFIGS[size]["imgsz"]
        x = torch.zeros(2, 3, imgsz, imgsz)
        with torch.no_grad():
            out = model(x)
        assert out.shape[0] == 2


class TestDetectSizeFromStateDict:
    """Verify size detection heuristic for all three variants."""

    @pytest.mark.parametrize("size", ["s", "m", "l"])
    def test_detect_roundtrip(self, size: str) -> None:
        from libreyolo.models.fomo.nn import LibreFOMOModel, detect_size_from_state_dict

        model = LibreFOMOModel(size=size, nc=1)
        sd = model.state_dict()
        detected = detect_size_from_state_dict(sd)
        assert detected == size, f"Expected size={size!r}, detected={detected!r}"

    def test_detect_empty_dict_returns_none(self) -> None:
        from libreyolo.models.fomo.nn import detect_size_from_state_dict

        assert detect_size_from_state_dict({}) is None

    def test_detect_unrelated_dict_returns_none(self) -> None:
        from libreyolo.models.fomo.nn import detect_size_from_state_dict

        assert detect_size_from_state_dict({"unrelated.weight": torch.zeros(3, 3)}) is None

    @pytest.mark.parametrize("size", ["s", "m", "l"])
    def test_detect_roundtrip_ddp(self, size: str) -> None:
        from libreyolo.models.fomo.nn import LibreFOMOModel, detect_size_from_state_dict

        model = LibreFOMOModel(size=size, nc=1)
        sd = {f"module.{k}": v for k, v in model.state_dict().items()}
        detected = detect_size_from_state_dict(sd)
        assert detected == size, f"Expected size={size!r} under DDP, detected={detected!r}"


class TestFOMONNInvalidSize:
    def test_raises_on_invalid_size(self) -> None:
        from libreyolo.models.fomo.nn import LibreFOMOModel

        with pytest.raises(ValueError, match="Unsupported LibreFOMO size"):
            LibreFOMOModel(size="xl", nc=1)


# ===========================================================================
# model.py — LibreFOMO class
# ===========================================================================


class TestLibreFOMORegistration:
    def test_family_attribute(self) -> None:
        from libreyolo.models.fomo.model import LibreFOMO

        assert LibreFOMO.FAMILY == "fomo"

    def test_supported_tasks(self) -> None:
        from libreyolo.models.fomo.model import LibreFOMO

        assert "point" in LibreFOMO.SUPPORTED_TASKS

    def test_default_task(self) -> None:
        from libreyolo.models.fomo.model import LibreFOMO

        assert LibreFOMO.DEFAULT_TASK == "point"

    def test_in_models_registry(self) -> None:
        import libreyolo.models  # triggers registration  # noqa: F401
        from libreyolo.models.base.model import BaseModel
        from libreyolo.models.fomo.model import LibreFOMO

        assert LibreFOMO in BaseModel._registry

    def test_input_sizes_populated(self) -> None:
        from libreyolo.models.fomo.model import LibreFOMO

        for size in ("s", "m", "l"):
            assert size in LibreFOMO.INPUT_SIZES
            assert LibreFOMO.INPUT_SIZES[size] > 0


class TestLibreFOMOCanLoad:
    @pytest.mark.parametrize("size", ["s", "m", "l"])
    def test_can_load_own_state_dict(self, size: str) -> None:
        from libreyolo.models.fomo.nn import LibreFOMOModel
        from libreyolo.models.fomo.model import LibreFOMO

        sd = LibreFOMOModel(size=size, nc=1).state_dict()
        assert LibreFOMO.can_load(sd)

    def test_cannot_load_unrelated_dict(self) -> None:
        from libreyolo.models.fomo.model import LibreFOMO

        assert not LibreFOMO.can_load({"some.unrelated.key": torch.zeros(1)})

    def test_cannot_load_missing_head(self) -> None:
        from libreyolo.models.fomo.model import LibreFOMO
        from libreyolo.models.fomo.nn import LibreFOMOModel

        sd = LibreFOMOModel(size="s", nc=1).state_dict()
        sd_no_head = {k: v for k, v in sd.items() if k != "head.weight"}
        assert not LibreFOMO.can_load(sd_no_head)

    def test_can_load_ddp_state_dict(self) -> None:
        from libreyolo.models.fomo.model import LibreFOMO
        from libreyolo.models.fomo.nn import LibreFOMOModel

        sd = LibreFOMOModel(size="s", nc=1).state_dict()
        sd_ddp = {f"module.{k}": v for k, v in sd.items()}
        assert LibreFOMO.can_load(sd_ddp)


class TestLibreFOMODetectNbClasses:
    @pytest.mark.parametrize("nc", [1, 2, 5])
    def test_detect_nb_classes(self, nc: int) -> None:
        from libreyolo.models.fomo.nn import LibreFOMOModel
        from libreyolo.models.fomo.model import LibreFOMO

        sd = LibreFOMOModel(size="s", nc=nc).state_dict()
        assert LibreFOMO.detect_nb_classes(sd) == nc

    def test_detect_nb_classes_ddp(self) -> None:
        from libreyolo.models.fomo.model import LibreFOMO
        from libreyolo.models.fomo.nn import LibreFOMOModel

        sd = LibreFOMOModel(size="s", nc=3).state_dict()
        sd_ddp = {f"module.{k}": v for k, v in sd.items()}
        assert LibreFOMO.detect_nb_classes(sd_ddp) == 3


class TestLibreFOMOFactory:
    def test_factory_loads_fomo_model(self, tmp_path: Path) -> None:
        """Verify that LibreYOLO factory auto-detects and loads LibreFOMO from a checkpoint."""
        from libreyolo import LibreYOLO
        from libreyolo.models.fomo.nn import LibreFOMOModel
        from libreyolo.utils.serialization import wrap_libreyolo_checkpoint

        model_module = LibreFOMOModel(size="s", nc=1)
        state_dict = model_module.state_dict()

        wrapped_ckpt = wrap_libreyolo_checkpoint(
            state_dict=state_dict,
            model_family="fomo",
            size="s",
            nc=1,
            names={0: "point"},
            task="point",
        )

        weights_file = tmp_path / "LibreFOMOs.pt"
        torch.save(wrapped_ckpt, weights_file)

        model = LibreYOLO(str(weights_file))

        assert model.family == "fomo"
        assert model.size == "s"
        assert model.task == "point"
        assert model.nb_classes == 1


class TestLibreFOMORandomInit:
    """Random-weight models instantiate without errors and have correct defaults."""

    @pytest.mark.parametrize("size", ["s", "m", "l"])
    def test_random_init_all_sizes(self, size: str) -> None:
        model = _make_random_fomo(size=size)
        assert model.size == size
        assert model.task == "point"
        assert model.device.type == "cpu"

    def test_random_init_sets_nb_classes(self) -> None:
        model = _make_random_fomo(size="s", nc=3)
        assert model.nb_classes == 3

    def test_model_attribute_is_nn_module(self) -> None:
        model = _make_random_fomo()
        assert isinstance(model.model, torch.nn.Module)


class TestLibreFOMODownloadURL:
    def test_valid_filename_returns_url(self) -> None:
        from libreyolo.models.fomo.model import LibreFOMO

        url = LibreFOMO.get_download_url("LibreFOMOs-point.pt")
        assert url == "https://huggingface.co/fomo-edge-ai/FOMO/resolve/main/weights/FOMOs.pt"

    def test_unknown_filename_returns_none(self) -> None:
        from libreyolo.models.fomo.model import LibreFOMO

        url = LibreFOMO.get_download_url("SomeOtherModel.pt")
        assert url is None
