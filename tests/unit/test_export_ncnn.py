"""Unit tests for the ncnn export module."""

import tempfile
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
import yaml

from libreyolo.export.exporter import BaseExporter, NcnnExporter

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _TinyModel(nn.Module):
    """Minimal model for export tests (no real weights needed)."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(8, 4)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def _make_wrapper(nb_classes=4, model_name="TESTYOLO", size="s", input_size=32):
    """Build a mock BaseModel-like wrapper around _TinyModel."""
    wrapper = MagicMock()
    wrapper.model = _TinyModel()
    wrapper.model.eval()
    wrapper.size = size
    wrapper.nb_classes = nb_classes
    wrapper.names = {i: f"class_{i}" for i in range(nb_classes)}
    wrapper.device = torch.device("cpu")
    wrapper._get_model_name.return_value = model_name
    wrapper._get_input_size.return_value = input_size
    return wrapper


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestNCNNFormatRegistration:
    """Test ncnn format is properly registered."""

    def test_ncnn_format_registered(self):
        """Verify ncnn is in registry."""
        assert "ncnn" in BaseExporter._registry

    def test_ncnn_format_config(self):
        """Verify ncnn format configuration."""
        assert NcnnExporter.suffix == "_ncnn"
        assert NcnnExporter.requires_onnx is False


class TestNCNNAvailabilityCheck:
    """Test ncnn/pnnx availability checking."""

    def test_check_ncnn_export_raises_helpful_error(self):
        """Verify helpful error message when pnnx not installed."""
        try:
            import pnnx  # noqa: F401

            pytest.skip("pnnx is installed, skipping missing pnnx test")
        except ImportError:
            pass

        from libreyolo.export.ncnn import check_ncnn_export_available

        with pytest.raises(ImportError) as exc_info:
            check_ncnn_export_available()

        error_msg = str(exc_info.value)
        assert "pnnx" in error_msg.lower()
        assert "pip install" in error_msg


class TestNCNNOutputPathGeneration:
    """Test output path generation for ncnn format."""

    def test_auto_path_ncnn(self):
        """ncnn export should generate path with _ncnn suffix."""
        wrapper = _make_wrapper(model_name="yolo9", size="t")
        exporter = NcnnExporter(wrapper)
        assert exporter.suffix == "_ncnn"

    def test_fp16_suffix_in_auto_path(self):
        """FP16 ncnn export should include _fp16 in auto-generated filename."""
        wrapper = _make_wrapper(model_name="TESTYOLO", size="s")
        exporter = NcnnExporter(wrapper)
        path = exporter._auto_output_path(half=True, int8=False)
        assert "_fp16_ncnn" in path


class TestNCNNMetadataYAML:
    """Test metadata YAML write/read round-trip."""

    def test_metadata_roundtrip(self):
        """Test that metadata can be written and read back correctly."""
        from libreyolo.export.ncnn import _save_metadata

        metadata = {
            "libreyolo_version": "0.1.5",
            "model_family": "yolo9",
            "model_size": "t",
            "nb_classes": 80,
            "names": {"0": "person", "1": "bicycle"},
            "imgsz": 640,
            "precision": "fp32",
            "dynamic": False,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            _save_metadata(output_dir, metadata)

            metadata_path = output_dir / "metadata.yaml"
            assert metadata_path.exists()

            with open(metadata_path) as f:
                loaded = yaml.safe_load(f)

            assert loaded["model_family"] == "yolo9"
            assert loaded["model_size"] == "t"
            assert loaded["nb_classes"] == 80
            assert loaded["precision"] == "fp32"
            assert loaded["imgsz"] == 640
            assert loaded["dynamic"] is False
            assert loaded["names"]["0"] == "person"
            assert loaded["names"]["1"] == "bicycle"

    def test_backend_reads_rectangular_metadata(self, tmp_path):
        """ncnn backend metadata must preserve non-square YOLO9 export shape."""
        from libreyolo.backends.ncnn import NcnnBackend

        metadata_path = tmp_path / "metadata.yaml"
        metadata_path.write_text(
            "\n".join(
                [
                    "model_family: yolo9",
                    "imgsz: 640",
                    "imgsz_h: 320",
                    "imgsz_w: 640",
                    "nc: 2",
                ]
            )
        )

        parsed = NcnnBackend._read_metadata(metadata_path)

        assert parsed[5] == (320, 640)


class TestNCNNBlobDiscovery:
    def test_param_parser_preserves_all_terminal_output_names(self, tmp_path):
        from libreyolo.backends.ncnn import NcnnBackend

        param_path = tmp_path / "model.ncnn.param"
        param_path.write_text(
            "\n".join(
                [
                    "7767517",
                    "4 5",
                    "Input input 0 1 images",
                    "Split split 1 2 images left right",
                    "Convolution scores_head 1 1 left scores 0=4",
                    "Convolution boxes_head 1 1 right boxes 0=4",
                ]
            )
        )

        inputs, outputs = NcnnBackend._discover_blob_names(param_path)

        assert inputs == ["images"]
        assert outputs == ["scores", "boxes"]


class TestNCNNExportValidation:
    """Test ncnn export validation in exporter."""

    def test_ncnn_export_fails_without_pnnx(self):
        """ncnn export without pnnx should raise ImportError."""
        try:
            import pnnx  # noqa: F401

            pytest.skip("pnnx is installed, skipping missing pnnx test")
        except ImportError:
            pass

        wrapper = _make_wrapper()
        exporter = NcnnExporter(wrapper)

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ImportError, match="pnnx"):
                exporter(output_path=str(Path(tmpdir) / "model_ncnn"))

    def test_low_level_ncnn_rejects_half_before_dependency_check(self, tmp_path):
        from libreyolo.export.ncnn import export_ncnn

        output = tmp_path / "model_ncnn"
        with pytest.raises(NotImplementedError, match="NCNN FP16"):
            export_ncnn(
                _TinyModel().eval(),
                torch.zeros(1, 3, 8, 8),
                output_path=str(output),
                half=True,
            )
        assert not output.exists()


class TestPNNXOptionalLoaderFailure:
    """Only a completed conversion may tolerate a missing PNNX helper."""

    @staticmethod
    def _run_direct_export(monkeypatch, tmp_path, export):
        from libreyolo.export.ncnn import _export_pnnx_direct

        monkeypatch.setitem(sys.modules, "pnnx", SimpleNamespace(export=export))
        model = _TinyModel().eval()
        dummy = torch.zeros(1, 3, 8, 8)
        return _export_pnnx_direct(model, dummy, tmp_path, half=False)

    def test_accepts_missing_matching_optional_loader_after_artifacts(
        self, monkeypatch, tmp_path
    ):
        def fake_export(model, traced_path, dummy, fp16):
            trace_dir = Path(traced_path).parent
            (trace_dir / "model.ncnn.param").write_text("param", encoding="utf-8")
            (trace_dir / "model.ncnn.bin").write_bytes(b"bin")
            raise FileNotFoundError("model_pnnx.py")

        param_path, bin_path = self._run_direct_export(
            monkeypatch, tmp_path, fake_export
        )

        assert param_path.read_text(encoding="utf-8") == "param"
        assert bin_path.read_bytes() == b"bin"

    def test_reraises_unrelated_missing_file_even_after_artifacts(
        self, monkeypatch, tmp_path
    ):
        def fake_export(model, traced_path, dummy, fp16):
            trace_dir = Path(traced_path).parent
            (trace_dir / "model.ncnn.param").write_text("param", encoding="utf-8")
            (trace_dir / "model.ncnn.bin").write_bytes(b"bin")
            raise FileNotFoundError("labels.txt")

        with pytest.raises(FileNotFoundError, match="labels.txt"):
            self._run_direct_export(monkeypatch, tmp_path, fake_export)

    def test_reraises_missing_loader_when_artifacts_are_incomplete(
        self, monkeypatch, tmp_path
    ):
        def fake_export(model, traced_path, dummy, fp16):
            trace_dir = Path(traced_path).parent
            (trace_dir / "model.ncnn.param").write_text("param", encoding="utf-8")
            raise FileNotFoundError("model_pnnx.py")

        with pytest.raises(FileNotFoundError, match="model_pnnx.py"):
            self._run_direct_export(monkeypatch, tmp_path, fake_export)


def test_ncnn_metadata_is_safe_loadable_with_classifier_sequences(tmp_path):
    import yaml

    from libreyolo.export.ncnn import _save_metadata

    _save_metadata(
        tmp_path,
        {
            "task": "classify",
            "classification_mean": (0.485, 0.456, 0.406),
            "classification_std": [0.229, 0.224, 0.225],
        },
    )

    loaded = yaml.safe_load((tmp_path / "metadata.yaml").read_text(encoding="utf-8"))
    assert loaded["classification_mean"] == [0.485, 0.456, 0.406]
    assert loaded["classification_std"] == [0.229, 0.224, 0.225]
