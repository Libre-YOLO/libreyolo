"""Unit tests for CoreML export. Mocks coremltools so it runs on every platform."""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

# Install a fake `coremltools` module so the import inside libreyolo.export.coreml
# succeeds even on machines without coremltools installed. Only do this if the
# real coremltools is genuinely unavailable, so we don't pollute sys.modules
# for any e2e test that runs in the same pytest session.
try:  # pragma: no cover - environment-dependent
    import coremltools  # noqa: F401
except ImportError:
    _fake_ct = MagicMock()
    _fake_ct.ComputeUnit.ALL = "ALL"
    _fake_ct.ComputeUnit.CPU_AND_GPU = "CPU_AND_GPU"
    _fake_ct.ComputeUnit.CPU_AND_NE = "CPU_AND_NE"
    _fake_ct.ComputeUnit.CPU_ONLY = "CPU_ONLY"
    _fake_ct.precision.FLOAT32 = "FLOAT32"
    _fake_ct.precision.FLOAT16 = "FLOAT16"
    _fake_ct.target.iOS15 = "iOS15"
    sys.modules["coremltools"] = _fake_ct

from libreyolo.export.coreml import _stringify_metadata, _to_compute_unit  # noqa: E402


pytestmark = pytest.mark.unit


class TestToComputeUnit:
    def test_all(self):
        import coremltools as ct

        assert _to_compute_unit("all") == ct.ComputeUnit.ALL

    def test_cpu_and_gpu(self):
        import coremltools as ct

        assert _to_compute_unit("cpu_and_gpu") == ct.ComputeUnit.CPU_AND_GPU

    def test_cpu_and_ne(self):
        import coremltools as ct

        assert _to_compute_unit("cpu_and_ne") == ct.ComputeUnit.CPU_AND_NE

    def test_cpu_only(self):
        import coremltools as ct

        assert _to_compute_unit("cpu_only") == ct.ComputeUnit.CPU_ONLY

    def test_case_insensitive(self):
        import coremltools as ct

        assert _to_compute_unit("ALL") == ct.ComputeUnit.ALL
        assert _to_compute_unit("Cpu_And_Ne") == ct.ComputeUnit.CPU_AND_NE

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="compute_units"):
            _to_compute_unit("tpu")


class _DummyModel(torch.nn.Module):
    def forward(self, x):
        return x.mean(dim=(2, 3))


class _DummyYoloxExportModel(torch.nn.Module):
    def __init__(self, nc: int = 80):
        super().__init__()
        self.nc = int(nc)

    def forward(self, x):
        batch = x.shape[0]
        height, width = int(x.shape[-2]), int(x.shape[-1])
        candidates = sum(
            ((height + stride - 1) // stride)
            * ((width + stride - 1) // stride)
            for stride in (8, 16, 32)
        )
        return torch.zeros(
            batch,
            candidates,
            5 + self.nc,
            dtype=x.dtype,
            device=x.device,
        )


class _DummyYolo9ExportModel(torch.nn.Module):
    def __init__(self, nc: int = 80):
        super().__init__()
        self.nc = int(nc)

    def forward(self, x):
        return torch.zeros(
            x.shape[0],
            4 + self.nc,
            8400,
            dtype=x.dtype,
            device=x.device,
        )


class _NoGradYoloxExportModel(_DummyYoloxExportModel):
    """Model whose trace must use the same inference state as JIT's check."""

    def forward(self, x):
        assert not torch.is_grad_enabled()
        if torch.jit.is_tracing():
            assert not torch.backends.mha.get_fastpath_enabled()
        return super().forward(x)


class _DummyRtdetrExportModel(torch.nn.Module):
    def forward(self, x):
        batch = x.shape[0]
        return {
            "pred_logits": torch.zeros(batch, 300, 80, dtype=x.dtype, device=x.device),
            "pred_boxes": torch.zeros(batch, 300, 4, dtype=x.dtype, device=x.device),
        }


class _DummyPicoSAM3ExportModel(torch.nn.Module):
    def forward(self, x):
        return x[:, :1]


class _DummyRFDETRPoseExportModel(torch.nn.Module):
    def forward(self, x):
        batch = x.shape[0]
        return {
            "pred_boxes": torch.zeros(batch, 10, 4, dtype=x.dtype, device=x.device),
            "pred_logits": torch.zeros(batch, 10, 1, dtype=x.dtype, device=x.device),
            "pred_keypoints": torch.zeros(
                batch,
                10,
                17 * 3,
                dtype=x.dtype,
                device=x.device,
            ),
        }


def _strict_metadata(
    family: str,
    task: str,
    size: str,
    *,
    names: dict[str, str] | None = None,
    imgsz: int = 640,
) -> dict:
    names = names or {"0": "class_0"}
    metadata = {
        "schema_version": "1",
        "libreyolo_version": "0.0.1",
        "model_family": family,
        "size": size,
        "model_size": size,
        "task": task,
        "supported_tasks": [task],
        "default_task": task,
        "names": names,
        "nc": len(names),
        "imgsz": imgsz,
        "imgsz_h": imgsz,
        "imgsz_w": imgsz,
    }
    if task == "pose":
        metadata.update({"num_keypoints": 17, "keypoint_dim": 3})
    return metadata


def _patch_ct(monkeypatch):
    """Reset the fake coremltools module and return the mock for assertions."""
    # Create the main coremltools mock
    fake = MagicMock()
    fake.ComputeUnit.ALL = "ALL"
    fake.ComputeUnit.CPU_AND_GPU = "CPU_AND_GPU"
    fake.ComputeUnit.CPU_AND_NE = "CPU_AND_NE"
    fake.ComputeUnit.CPU_ONLY = "CPU_ONLY"
    fake.precision.FLOAT32 = "FLOAT32"
    fake.precision.FLOAT16 = "FLOAT16"
    fake.target.iOS15 = "iOS15"
    fake.ImageType = MagicMock(side_effect=lambda **kw: ("ImageType", kw))
    fake.TensorType = MagicMock(side_effect=lambda **kw: ("TensorType", kw))

    # Create models submodule mock
    fake_models = MagicMock()
    fake_models.pipeline = MagicMock()
    fake.models = fake_models
    # MLModel is in the models submodule
    fake.models.MLModel = MagicMock()

    # Create the MLModel mock that gets returned by convert
    mlmodel = MagicMock()
    mlmodel.user_defined_metadata = {
        "com.github.apple.coremltools.conversion_date": "2026-07-29",
        "com.github.apple.coremltools.source": "torch==2.7.0",
        "com.github.apple.coremltools.source_dialect": "TorchScript",
        "com.github.apple.coremltools.version": "9.0",
    }
    mlmodel.save.side_effect = lambda path: Path(path).mkdir(parents=True)
    fake.convert = MagicMock(return_value=mlmodel)
    fake.utils.load_spec.side_effect = lambda _path: SimpleNamespace(
        description=SimpleNamespace(
            metadata=SimpleNamespace(
                userDefined=dict(mlmodel.user_defined_metadata)
            ),
        ),
    )

    from libreyolo.export import coreml as coreml_module
    from libreyolo.export import coreml_identity

    def bind_test_abi(metadata, _spec):
        return {
            **metadata,
            "coreml_profile_abi_schema": "coreml-deployment-abi-v2",
            "coreml_profile_abi_sha256": "3" * 64,
        }

    monkeypatch.setattr(
        coreml_identity,
        "bind_coreml_deployment_abi",
        bind_test_abi,
    )
    monkeypatch.setattr(
        coreml_identity,
        "validate_coreml_deployment_abi",
        lambda _spec, _metadata: "3" * 64,
    )
    monkeypatch.setattr(
        coreml_module,
        "_validate_coreml_deployment_spec",
        lambda *_args, **_kwargs: None,
    )

    # Patch the module and submodules
    monkeypatch.setitem(sys.modules, "coremltools", fake)
    monkeypatch.setitem(sys.modules, "coremltools.models", fake_models)
    monkeypatch.setitem(
        sys.modules, "coremltools.models.pipeline", fake_models.pipeline
    )

    return fake, mlmodel


class TestExportCoreML:
    def test_trace_runs_with_gradients_disabled(self, tmp_path, monkeypatch):
        _patch_ct(monkeypatch)
        from libreyolo.export.coreml import export_coreml

        fastpath_before = torch.backends.mha.get_fastpath_enabled()
        result = export_coreml(
            _NoGradYoloxExportModel(nc=1).eval(),
            torch.randn(1, 3, 32, 32),
            output_path=str(tmp_path / "no-grad.mlpackage"),
            precision="fp32",
            metadata=_strict_metadata(
                "yolox",
                "detect",
                "n",
                names={"0": "person"},
                imgsz=32,
            ),
            model_family="yolox",
            model_task="detect",
            model_size="n",
            compute_units="cpu_only",
        )

        assert result.endswith("no-grad.mlpackage")
        assert torch.backends.mha.get_fastpath_enabled() is fastpath_before

    def test_fp32_basic_call(self, tmp_path, monkeypatch):
        fake, mlmodel = _patch_ct(monkeypatch)
        from libreyolo.export.coreml import export_coreml

        nn_model = _DummyYoloxExportModel(nc=1).eval()
        dummy = torch.randn(1, 3, 640, 640)
        out = tmp_path / "model.mlpackage"

        result = export_coreml(
            nn_model,
            dummy,
            output_path=str(out),
            precision="fp32",
            compute_units="all",
            nms=False,
            metadata=_strict_metadata("yolox", "detect", "n", names={"0": "person"}),
            model_family="yolox",
            model_task="detect",
            model_size="n",
        )

        assert result == str(out)
        # ct.convert called with mlprogram + FLOAT32 + iOS15 + ImageType input
        kwargs = fake.convert.call_args.kwargs
        assert kwargs["convert_to"] == "mlprogram"
        assert kwargs["compute_precision"] == "FLOAT32"
        assert kwargs["minimum_deployment_target"] == "iOS15"
        # ImageType called with scale=1/255 and image input name 'image'
        img_kwargs = fake.ImageType.call_args.kwargs
        assert img_kwargs["name"] == "image"
        assert img_kwargs["scale"] == pytest.approx(1.0 / 255.0)
        assert img_kwargs["bias"] == [0.0, 0.0, 0.0]
        assert kwargs["compute_units"] == "ALL"
        assert kwargs["outputs"] == [
            ("TensorType", {"name": "prediction"}),
        ]
        # Metadata was stringified and stored
        assert all(isinstance(v, str) for v in mlmodel.user_defined_metadata.values())
        assert mlmodel.user_defined_metadata["model_family"] == "yolox"
        assert mlmodel.user_defined_metadata["libreyolo_producer"] == "libreyolo"
        assert mlmodel.user_defined_metadata["artifact_format"] == "coreml"
        assert mlmodel.user_defined_metadata["coreml_io_schema_version"] == "2"
        coreml_io = __import__("json").loads(mlmodel.user_defined_metadata["coreml_io"])
        assert coreml_io["input"]["geometry"] == "letterbox_top_left"
        assert coreml_io["input"]["resize_backend"] == "pillow"
        assert coreml_io["validation"] == {"color": "rgb", "range": "0_255"}
        assert coreml_io["outputs"][0]["name"] == "prediction"
        assert coreml_io["outputs"][0]["shape"] == [1, 8400, 6]
        assert out.is_dir()
        staged_path = Path(mlmodel.save.call_args.args[0])
        assert staged_path.name == "candidate.mlpackage"
        assert staged_path != out

    def test_unpromoted_exact_profile_defaults_to_cpu_only(
        self,
        tmp_path,
        monkeypatch,
    ):
        fake, _ = _patch_ct(monkeypatch)
        from libreyolo.export.coreml import export_coreml

        output = tmp_path / "yolo9-profile.mlpackage"
        export_coreml(
            _DummyYolo9ExportModel().eval(),
            torch.randn(1, 3, 640, 640),
            output_path=str(output),
            metadata=_strict_metadata(
                "yolo9",
                "detect",
                "t",
                names={
                    str(index): f"class_{index}"
                    for index in range(80)
                },
            ),
            model_family="yolo9",
            model_task="detect",
            model_size="t",
        )
        assert fake.convert.call_args.kwargs["compute_units"] == "CPU_ONLY"
        assert output.is_dir()

    def test_picosam3_component_converts_saves_and_embeds_scope(
        self,
        tmp_path,
        monkeypatch,
    ):
        fake, mlmodel = _patch_ct(monkeypatch)
        from libreyolo.export.coreml import export_coreml

        output = tmp_path / "picosam3.mlpackage"
        result = export_coreml(
            _DummyPicoSAM3ExportModel().eval(),
            torch.zeros(1, 3, 96, 96),
            output_path=str(output),
            metadata=_strict_metadata(
                "picosam3",
                "segment",
                "pico",
                names={"0": "object"},
                imgsz=96,
            ),
            model_family="picosam3",
            model_task="segment",
            model_size="pico",
            compute_units="cpu_only",
        )

        assert result == str(output)
        assert output.is_dir()
        assert fake.ImageType.call_args.kwargs["name"] == "roi_image"
        assert fake.convert.call_args.kwargs["outputs"] == [
            ("TensorType", {"name": "mask_logits"})
        ]
        metadata = mlmodel.user_defined_metadata
        assert metadata["artifact_scope"] == "roi_component"
        assert metadata["component_contract"] == "picosam3_roi_v1"
        assert metadata["roi_input_size"] == "96"
        assert metadata["roi_padding"] == "0.1"
        io = __import__("json").loads(metadata["coreml_io"])
        assert io["outputs"] == [
            {
                "name": "mask_logits",
                "role": "mask_logits",
                "encoding": "raw_logits",
                "rank": 4,
                "dtype": "float32",
                "shape": [1, 1, 96, 96],
            }
        ]

    def test_eomt_compact_query_component_converts_with_tensor_boundary(
        self,
        tmp_path,
        monkeypatch,
    ):
        fake, mlmodel = _patch_ct(monkeypatch)
        from libreyolo.export.coreml import export_coreml
        from libreyolo.models.eomt.nn import LibreEoMTNet

        torch.manual_seed(31)
        model = LibreEoMTNet(
            config="s",
            nb_classes=2,
            image_size=32,
            num_queries=4,
        ).eval()
        output = tmp_path / "eomt-semantic.mlpackage"
        result = export_coreml(
            model,
            torch.zeros(1, 3, 32, 32),
            output_path=str(output),
            metadata=_strict_metadata(
                "eomt",
                "semantic",
                "s",
                names={"0": "a", "1": "b"},
                imgsz=32,
            ),
            model_family="eomt",
            model_task="semantic",
            model_size="s",
            compute_units="cpu_only",
        )

        assert result == str(output)
        assert output.is_dir()
        fake.ImageType.assert_not_called()
        assert fake.TensorType.call_args_list[0].kwargs == {
            "name": "image",
            "shape": (1, 3, 32, 32),
        }
        assert fake.convert.call_args.kwargs["outputs"] == [
            ("TensorType", {"name": "class_queries_logits"}),
            ("TensorType", {"name": "masks_queries_logits"}),
        ]
        metadata = mlmodel.user_defined_metadata
        assert metadata["artifact_scope"] == "patch_component"
        assert metadata["eomt_contract"] == "eomt_raw_queries_v1"
        assert metadata["eomt_num_queries"] == "4"
        assert metadata["eomt_attention_mask"] == "functional_concat_v1"
        io = __import__("json").loads(metadata["coreml_io"])
        assert io["input"]["geometry"] == "eomt_split"
        assert [item["shape"] for item in io["outputs"]] == [
            [1, 4, 3],
            [1, 4, 8, 8],
        ]

    def test_rfdetr_pose_uses_float_tensor_boundary(self, tmp_path, monkeypatch):
        fake, mlmodel = _patch_ct(monkeypatch)
        from libreyolo.export.coreml import export_coreml

        output = tmp_path / "rfdetr-pose.mlpackage"
        result = export_coreml(
            _DummyRFDETRPoseExportModel().eval(),
            torch.rand(1, 3, 64, 64),
            output_path=str(output),
            compute_units="cpu_only",
            model_family="rfdetr",
            model_task="pose",
            model_size="n",
            metadata=_strict_metadata(
                "rfdetr",
                "pose",
                "n",
                names={"0": "person"},
                imgsz=64,
            ),
        )

        assert result == str(output)
        pipeline = fake.PassPipeline.return_value
        fake.PassPipeline.assert_called_once_with()
        pipeline.remove_passes.assert_called_once_with(
            {"common::divide_to_multiply"}
        )

        assert fake.convert.call_args.kwargs["pass_pipeline"] is pipeline
        fake.ImageType.assert_not_called()
        tensor_kwargs = fake.TensorType.call_args_list[0].kwargs
        assert tensor_kwargs == {"name": "image", "shape": (1, 3, 64, 64)}
        metadata = mlmodel.user_defined_metadata
        assert metadata["coreml_required_compute_units"] == "cpu_only"
        assert (
            metadata["coreml_conversion_pass_profile"]
            == "rfdetr_pose_preserve_division_v1"
        )
        assert __import__("json").loads(metadata["coreml_disabled_passes"]) == [
            "common::divide_to_multiply"
        ]
        io = __import__("json").loads(metadata["coreml_io"])
        assert io["input"] == {
            "name": "image",
            "kind": "tensor",
            "layout": "NCHW",
            "color": "rgb",
            "range": "0_1",
            "geometry": "stretch",
            "interpolation": "bilinear",
            "resize_backend": "torchvision",
            "pad_value": 0,
        }

    def test_pass_pipeline_preserves_rfdetr_division(self):
        from libreyolo.export.coreml import _coreml_pass_pipeline

        fake_ct = SimpleNamespace(PassPipeline=MagicMock())
        assert _coreml_pass_pipeline(fake_ct, "ec", "pose") is None
        fake_ct.PassPipeline.assert_not_called()

        for task in ("detect", "obb", "pose"):
            pipeline = _coreml_pass_pipeline(fake_ct, "rfdetr", task)
            assert pipeline is fake_ct.PassPipeline.return_value
            pipeline.remove_passes.assert_called_once_with(
                {"common::divide_to_multiply"}
            )
            fake_ct.PassPipeline.reset_mock()
            pipeline.remove_passes.reset_mock()

        nafnet_pipeline = _coreml_pass_pipeline(fake_ct, "nafnet", "restore")
        assert nafnet_pipeline is fake_ct.PassPipeline.return_value
        fake_ct.PassPipeline.assert_called_once_with()
        nafnet_pipeline.remove_passes.assert_called_once_with(
            {"common::fuse_elementwise_to_batchnorm"}
        )

    @pytest.mark.parametrize(
        "compute_units",
        ["all", "cpu_and_gpu", "cpu_and_ne"],
    )
    def test_direct_rfdetr_pose_rejects_non_cpu_before_graph_or_destination(
        self,
        tmp_path,
        monkeypatch,
        compute_units,
    ):
        fake, _mlmodel = _patch_ct(monkeypatch)
        from libreyolo.export.coreml import export_coreml

        output = tmp_path / "rejected.mlpackage"
        with pytest.raises(NotImplementedError, match="requires.*cpu_only"):
            export_coreml(
                _DummyRFDETRPoseExportModel().eval(),
                torch.rand(1, 3, 64, 64),
                output_path=str(output),
                compute_units=compute_units,
                model_family="rfdetr",
                model_task="pose",
                model_size="n",
                metadata=_strict_metadata(
                    "rfdetr",
                    "pose",
                    "n",
                    names={"0": "person"},
                    imgsz=64,
                ),
            )

        fake.convert.assert_not_called()
        assert not output.exists()

    def test_direct_rfdetr_pose_rejects_fp16_before_graph_or_destination(
        self,
        tmp_path,
        monkeypatch,
    ):
        fake, _mlmodel = _patch_ct(monkeypatch)
        from libreyolo.export.coreml import export_coreml

        output = tmp_path / "rejected-fp16.mlpackage"
        with pytest.raises(NotImplementedError, match="requires precision='fp32'"):
            export_coreml(
                _DummyRFDETRPoseExportModel().eval(),
                torch.rand(1, 3, 64, 64),
                output_path=str(output),
                precision="fp16",
                compute_units="cpu_only",
                model_family="rfdetr",
                model_task="pose",
                model_size="n",
                metadata=_strict_metadata(
                    "rfdetr",
                    "pose",
                    "n",
                    names={"0": "person"},
                    imgsz=64,
                ),
            )

        fake.convert.assert_not_called()
        assert not output.exists()

    def test_direct_depth_anything3_rejects_fp16_before_conversion(
        self,
        tmp_path,
        monkeypatch,
    ):
        fake, _mlmodel = _patch_ct(monkeypatch)
        from libreyolo.export.coreml import export_coreml

        output = tmp_path / "rejected-da3-fp16.mlpackage"
        with pytest.raises(
            NotImplementedError,
            match="Depth Anything 3.*precision='fp32'",
        ):
            export_coreml(
                _DummyYoloxExportModel(nc=1).eval(),
                torch.rand(1, 3, 504, 504),
                output_path=str(output),
                precision="fp16",
                compute_units="cpu_only",
                model_family="depth_anything3",
                model_task="depth",
                model_size="l",
                metadata=_strict_metadata(
                    "depth_anything3",
                    "depth",
                    "l",
                    imgsz=504,
                ),
            )

        fake.convert.assert_not_called()
        assert not output.exists()

    def test_fp16_uses_float16_precision(self, tmp_path, monkeypatch):
        fake, mlmodel = _patch_ct(monkeypatch)
        from libreyolo.export.coreml import export_coreml

        export_coreml(
            _DummyYoloxExportModel(nc=1).eval(),
            torch.randn(1, 3, 640, 640),
            output_path=str(tmp_path / "m.mlpackage"),
            precision="fp16",
            compute_units="cpu_and_ne",
            nms=False,
            metadata=_strict_metadata("yolox", "detect", "n"),
            model_family="yolox",
            model_task="detect",
            model_size="n",
        )
        assert fake.convert.call_args.kwargs["compute_precision"] == "FLOAT16"
        assert fake.convert.call_args.kwargs["compute_units"] == "CPU_AND_NE"

    def test_metadata_names_json_encoded(self, tmp_path, monkeypatch):
        import json

        fake, mlmodel = _patch_ct(monkeypatch)
        from libreyolo.export.coreml import export_coreml

        export_coreml(
            _DummyYoloxExportModel(nc=2).eval(),
            torch.randn(1, 3, 640, 640),
            output_path=str(tmp_path / "m.mlpackage"),
            precision="fp32",
            compute_units="all",
            nms=False,
            metadata=_strict_metadata(
                "yolox",
                "detect",
                "n",
                names={"0": "person", "1": "cat"},
            ),
            model_family="yolox",
            model_task="detect",
            model_size="n",
        )
        decoded = json.loads(mlmodel.user_defined_metadata["names"])
        assert decoded == {"0": "person", "1": "cat"}

    def test_direct_export_normalizes_destination_suffix(self, tmp_path, monkeypatch):
        _fake, _mlmodel = _patch_ct(monkeypatch)
        from libreyolo.export.coreml import export_coreml

        requested = tmp_path / "detector.onnx"
        result = export_coreml(
            _DummyYoloxExportModel(nc=1).eval(),
            torch.randn(1, 3, 16, 16),
            output_path=str(requested),
            metadata=_strict_metadata(
                "yolox",
                "detect",
                "n",
                names={"0": "person"},
                imgsz=16,
            ),
            model_family="yolox",
            model_task="detect",
            model_size="n",
            compute_units="cpu_only",
        )

        assert result == str(tmp_path / "detector.mlpackage")
        assert Path(result).is_dir()
        assert not requested.exists()

    def test_parser_incompatible_detector_shape_fails_before_conversion(
        self,
        tmp_path,
        monkeypatch,
    ):
        fake, _mlmodel = _patch_ct(monkeypatch)
        from libreyolo.export.coreml import export_coreml

        output = tmp_path / "invalid.mlpackage"
        with pytest.raises(RuntimeError, match="detector prediction must be rank three"):
            export_coreml(
                _DummyModel().eval(),
                torch.randn(1, 3, 16, 16),
                output_path=str(output),
                metadata=_strict_metadata(
                    "yolox",
                    "detect",
                    "n",
                    names={"0": "person"},
                    imgsz=16,
                ),
                model_family="yolox",
                model_task="detect",
                model_size="n",
                compute_units="cpu_only",
            )

        fake.convert.assert_not_called()
        assert not output.exists()

    def test_supported_tasks_json_encoded(self):
        import json

        metadata = _stringify_metadata(
            {
                "supported_tasks": ["detect", "segment"],
                "default_task": "detect",
            }
        )

        assert json.loads(metadata["supported_tasks"]) == ["detect", "segment"]

    def test_rtdetr_dict_output_is_flattened_for_trace(self):
        from libreyolo.export.coreml import _wrap_for_family

        wrapped = _wrap_for_family(_DummyRtdetrExportModel().eval(), "rtdetr")
        logits, boxes = wrapped(torch.randn(1, 3, 640, 640))

        assert logits.shape == (1, 300, 80)
        assert boxes.shape == (1, 300, 4)


class TestUnsupportedFamily:
    def test_rfdetr_s_and_l_obb_are_rejected_after_m4_parity_failures(self):
        from libreyolo.export.coreml import _validate_export_profile

        with pytest.raises(NotImplementedError, match="2.66%"):
            _validate_export_profile("rfdetr", "obb", "l")
        with pytest.raises(NotImplementedError, match="0.52%"):
            _validate_export_profile("rfdetr", "obb", "s")
        _validate_export_profile("rfdetr", "obb", "n")
        _validate_export_profile("rfdetr", "obb", "m")

    def test_unknown_family_raises(self, tmp_path, monkeypatch):
        _patch_ct(monkeypatch)
        from libreyolo.export.coreml import export_coreml

        with pytest.raises(NotImplementedError, match="not supported"):
            export_coreml(
                _DummyModel().eval(),
                torch.randn(1, 3, 640, 640),
                output_path=str(tmp_path / "m.mlpackage"),
                precision="fp32",
                compute_units="all",
                nms=False,
                metadata={"model_family": "unknown_family"},
                model_family="unknown_family",
            )

    def test_task_contract_is_not_a_family_only_allowlist(self, tmp_path, monkeypatch):
        _patch_ct(monkeypatch)
        from libreyolo.export.coreml import export_coreml

        with pytest.raises(NotImplementedError, match="task 'segment'"):
            export_coreml(
                _DummyModel().eval(),
                torch.randn(1, 3, 16, 16),
                output_path=str(tmp_path / "m.mlpackage"),
                model_family="yolo9",
                model_task="segment",
            )

    def test_rfdetr_segment_direct_export_fails_before_graph_or_destination(
        self,
        tmp_path,
        monkeypatch,
    ):
        _patch_ct(monkeypatch)
        from libreyolo.export import coreml

        prepare = MagicMock()
        monkeypatch.setattr(coreml, "_wrap_coreml_contract", prepare)
        destination = tmp_path / "rfdetr-segment.mlpackage"

        with pytest.raises(
            NotImplementedError,
            match="not supported for 'rfdetr' task 'segment'",
        ):
            coreml.export_coreml(
                _DummyModel().eval(),
                torch.randn(1, 3, 16, 16),
                output_path=str(destination),
                model_family="rfdetr",
                model_task="segment",
                model_size="n",
            )

        prepare.assert_not_called()
        assert not destination.exists()

    @pytest.mark.parametrize("size", [None, "s", "m", "l", "x"])
    def test_deimv2_license_boundary_is_size_gated(self, size, tmp_path, monkeypatch):
        _patch_ct(monkeypatch)
        from libreyolo.export.coreml import export_coreml

        with pytest.raises(NotImplementedError, match="DINOv3 licensing boundary"):
            export_coreml(
                _DummyRtdetrExportModel().eval(),
                torch.randn(1, 3, 16, 16),
                output_path=str(tmp_path / "m.mlpackage"),
                model_family="deimv2",
                model_task="detect",
                model_size=size,
            )

    def test_batch_rejected_before_graph_preparation(self, tmp_path, monkeypatch):
        _patch_ct(monkeypatch)
        from libreyolo.export import coreml

        prepare = MagicMock()
        monkeypatch.setattr(coreml, "_wrap_coreml_contract", prepare)
        with pytest.raises(ValueError, match="batch=1"):
            coreml.export_coreml(
                _DummyModel().eval(),
                torch.randn(2, 3, 16, 16),
                output_path=str(tmp_path / "m.mlpackage"),
                model_family="resnet",
                model_task="classify",
                model_size="18",
                metadata=_strict_metadata("resnet", "classify", "18"),
            )
        prepare.assert_not_called()

    def test_dynamic_rejected_before_graph_preparation(self, tmp_path, monkeypatch):
        _patch_ct(monkeypatch)
        from libreyolo.export import coreml

        prepare = MagicMock()
        monkeypatch.setattr(coreml, "_wrap_coreml_contract", prepare)
        with pytest.raises(NotImplementedError, match="dynamic=True"):
            coreml.export_coreml(
                _DummyModel().eval(),
                torch.randn(1, 3, 16, 16),
                output_path=str(tmp_path / "m.mlpackage"),
                model_family="resnet",
                model_task="classify",
                model_size="18",
                dynamic=True,
            )
        prepare.assert_not_called()

    def test_sam_dynamic_flag_rejected_before_component_wrapping(
        self,
        tmp_path,
        monkeypatch,
    ):
        _patch_ct(monkeypatch)
        from libreyolo.export import coreml

        wrap = MagicMock()
        monkeypatch.setattr(coreml, "_export_sam_coreml_impl", wrap)
        with pytest.raises(
            NotImplementedError,
            match="prompt_max_points.*dynamic=True",
        ):
            coreml.export_coreml(
                _DummyModel().eval(),
                torch.randn(1, 3, 16, 16),
                output_path=str(tmp_path / "sam.mlpackage"),
                model_family="mobilesam",
                model_task="segment",
                model_size="tiny",
                dynamic=True,
            )
        wrap.assert_not_called()

    def test_sam_prompt_bound_controls_execution_profile_label(
        self,
        tmp_path,
        monkeypatch,
    ):
        _patch_ct(monkeypatch)
        from libreyolo.export import coreml

        dispatch = MagicMock(return_value=str(tmp_path / "edgetam.mlpackage"))
        monkeypatch.setattr(coreml, "_export_sam_coreml_impl", dispatch)
        metadata = _strict_metadata(
            "edgetam",
            "segment",
            "edge",
            imgsz=1024,
        )
        with pytest.raises(NotImplementedError, match="not yet been promoted"):
            coreml.export_coreml(
                _DummyModel().eval(),
                torch.zeros(1, 3, 1024, 1024),
                output_path=str(tmp_path / "edgetam.mlpackage"),
                model_family="edgetam",
                model_task="segment",
                model_size="edge",
                metadata=metadata,
                prompt_max_points=4,
                compute_units="validated",
            )
        dispatch.assert_not_called()

        with pytest.warns(RuntimeWarning, match="awaiting"):
            coreml.export_coreml(
                _DummyModel().eval(),
                torch.zeros(1, 3, 1024, 1024),
                output_path=str(tmp_path / "edgetam-p16.mlpackage"),
                model_family="edgetam",
                model_task="segment",
                model_size="edge",
                metadata=metadata,
                prompt_max_points=4,
                compute_units="cpu_only",
            )
        assert not dispatch.call_args.kwargs[
            "has_candidate_execution_profile"
        ]
        assert dispatch.call_args.kwargs["requested_compute_units"] == (
            "cpu_only"
        )

    def test_incomplete_metadata_rejected_before_graph_preparation(
        self, tmp_path, monkeypatch
    ):
        _patch_ct(monkeypatch)
        from libreyolo.export import coreml

        prepare = MagicMock()
        monkeypatch.setattr(coreml, "_wrap_coreml_contract", prepare)
        with pytest.raises(ValueError, match=r"missing .*schema_version"):
            coreml.export_coreml(
                _DummyModel().eval(),
                torch.randn(1, 3, 16, 16),
                output_path=str(tmp_path / "m.mlpackage"),
                model_family="resnet",
                model_task="classify",
                model_size="18",
                metadata={"names": {"0": "class_0"}, "nc": 1},
                compute_units="cpu_only",
            )
        prepare.assert_not_called()


class TestCoreMLContracts:
    def test_supported_registry_is_task_granular(self):
        from libreyolo.export.coreml import supported_coreml_exports

        supported = supported_coreml_exports()
        assert len(supported) == 59
        assert ("grounding_dino", "detect") not in supported
        assert ("clip", "classify") in supported
        assert ("siglip2", "classify") in supported
        assert ("rfdetr", "detect") in supported
        assert ("rfdetr", "segment") not in supported
        assert ("ec", "pose") in supported
        assert ("yolo9", "segment") not in supported
        assert ("yolonas", "detect") in supported
        assert ("yolonas", "pose") in supported
        assert ("picosam3", "segment") in supported
        assert ("rtmdet", "segment") in supported
        assert ("eomt", "semantic") in supported
        assert ("eomt", "segment") in supported
        assert ("eomt", "panoptic") in supported

    def test_family_photometric_wrappers_are_exact(self):
        from libreyolo.export.coreml import _wrap_coreml_contract

        x = torch.tensor(
            [
                [
                    [[0.1, 0.2]],
                    [[0.3, 0.4]],
                    [[0.5, 0.6]],
                ]
            ],
            dtype=torch.float32,
        )

        picodet = _wrap_coreml_contract(torch.nn.Identity(), "picodet", "detect")
        picodet_mean = torch.tensor((123.675, 116.28, 103.53)).view(1, 3, 1, 1)
        picodet_std = torch.tensor((58.395, 57.12, 57.375)).view(1, 3, 1, 1)
        torch.testing.assert_close(picodet(x), (x * 255.0 - picodet_mean) / picodet_std)

        rtmdet = _wrap_coreml_contract(torch.nn.Identity(), "rtmdet", "detect")
        bgr_mean = torch.tensor((103.53, 116.28, 123.675)).view(1, 3, 1, 1)
        bgr_std = torch.tensor((57.375, 57.12, 58.395)).view(1, 3, 1, 1)
        torch.testing.assert_close(
            rtmdet(x),
            (x[:, [2, 1, 0]] * 255.0 - bgr_mean) / bgr_std,
        )

        fomo = _wrap_coreml_contract(torch.nn.Identity(), "fomo", "point")
        torch.testing.assert_close(fomo(x), (x - 0.5) / 0.5)

    def test_resize_backend_and_kernel_are_explicit(self):
        from libreyolo.export.coreml import _input_contract

        opencv_families = {
            "depth_anything",
            "lingbotvision",
            "picodet",
            "pidnet",
            "rtdetr",
            "yolo9",
            "yolo9_e2e",
            "yolo9_p2",
            "zipdepth",
        }
        for family in opencv_families:
            assert _input_contract(family, "detect", "n")["resize_backend"] == "opencv"

        for family in {
            "dfine",
            "rfdetr",
            "rtdetrv2",
            "rtmdet",
            "yolo7",
            "yolox",
        }:
            assert _input_contract(family, "detect", "n")["resize_backend"] == "pillow"

        assert (
            _input_contract("depth_anything", "depth", "s")["interpolation"]
            == "bicubic"
        )

    def test_multi_output_semantic_order(self):
        from libreyolo.export.coreml import _output_contract

        assert [
            item["name"] for item in _output_contract("dfine", "segment", nms=False)
        ] == ["pred_logits", "pred_boxes", "pred_masks"]
        assert [item["name"] for item in _output_contract("ec", "pose", nms=False)] == [
            "pred_logits",
            "pred_keypoints",
        ]

    def test_artifact_metadata_is_pinned_to_exported_task(self):
        from libreyolo.export.coreml import _prepare_strict_metadata

        metadata = _strict_metadata("dfine", "segment", "n")
        metadata["supported_tasks"] = ["detect", "segment"]
        metadata["default_task"] = "detect"
        prepared = _prepare_strict_metadata(
            metadata,
            family="dfine",
            task="segment",
            size="n",
            height=640,
            width=640,
        )
        assert prepared["supported_tasks"] == ["segment"]
        assert prepared["default_task"] == "segment"

    @pytest.mark.parametrize("nb_classes", [2, "two"])
    def test_conflicting_nb_classes_metadata_is_rejected(self, nb_classes):
        from libreyolo.export.coreml import _prepare_strict_metadata

        metadata = _strict_metadata("yolox", "detect", "n")
        metadata["nb_classes"] = nb_classes
        with pytest.raises(ValueError, match="nb_classes"):
            _prepare_strict_metadata(
                metadata,
                family="yolox",
                task="detect",
                size="n",
                height=640,
                width=640,
            )

    @pytest.mark.parametrize(
        "invalid_schema",
        [
            [0, True],
            [0, 17.5],
            [0, {"count": 17}],
            [0, [17]],
        ],
    )
    def test_rfdetr_grouppose_rejects_noninteger_schema_items(
        self,
        invalid_schema,
    ):
        from libreyolo.export.coreml import _prepare_strict_metadata

        metadata = _strict_metadata("rfdetr", "pose", "n")
        metadata.update(
            {
                "keypoint_dim": 8,
                "num_keypoints_per_class": invalid_schema,
            }
        )
        with pytest.raises(ValueError, match="nonnegative integers"):
            _prepare_strict_metadata(
                metadata,
                family="rfdetr",
                task="pose",
                size="n",
                height=640,
                width=640,
            )

    def test_classification_metadata_defaults_to_softmax(self):
        from libreyolo.export.coreml import _prepare_strict_metadata

        prepared = _prepare_strict_metadata(
            _strict_metadata("resnet", "classify", "18", imgsz=224),
            family="resnet",
            task="classify",
            size="18",
            height=224,
            width=224,
        )

        assert prepared["classification_activation"] == "softmax"

    def test_invalid_classification_activation_is_rejected(self):
        from libreyolo.export.coreml import _prepare_strict_metadata

        metadata = _strict_metadata("resnet", "classify", "18", imgsz=224)
        metadata["classification_activation"] = "identity"
        with pytest.raises(ValueError, match="classification_activation"):
            _prepare_strict_metadata(
                metadata,
                family="resnet",
                task="classify",
                size="18",
                height=224,
                width=224,
            )

    @pytest.mark.parametrize("family", ["nafnet", "realesrgan"])
    def test_fixed_restore_contract_requires_exact_native_canvas(self, family):
        from libreyolo.export.coreml import _input_contract

        contract = _input_contract(family, "restore", "s")

        assert contract["geometry"] == "native"


class TestAtomicPackageSave:
    def test_save_failure_preserves_existing_package(self, tmp_path):
        from libreyolo.export.coreml import _save_mlpackage_atomic

        destination = tmp_path / "model.mlpackage"
        destination.mkdir()
        sentinel = destination / "known-good.txt"
        sentinel.write_text("old", encoding="utf-8")

        model = MagicMock()
        model.save.side_effect = RuntimeError("save failed")
        with pytest.raises(RuntimeError, match="save failed"):
            _save_mlpackage_atomic(model, destination)

        assert sentinel.read_text(encoding="utf-8") == "old"
        assert not list(tmp_path.glob(".*.staging"))

    def test_successful_save_replaces_only_after_staging(self, tmp_path):
        from libreyolo.export.coreml import _save_mlpackage_atomic

        destination = tmp_path / "model.mlpackage"
        destination.mkdir()
        (destination / "old.txt").write_text("old", encoding="utf-8")

        model = MagicMock()

        def save(path):
            candidate = Path(path)
            candidate.mkdir(parents=True)
            (candidate / "new.txt").write_text("new", encoding="utf-8")

        model.save.side_effect = save
        _save_mlpackage_atomic(model, destination)

        assert (destination / "new.txt").read_text(encoding="utf-8") == "new"
        assert not (destination / "old.txt").exists()
        assert not list(tmp_path.glob(".*.staging"))

    def test_candidate_validation_failure_preserves_existing_package(
        self,
        tmp_path,
    ):
        from libreyolo.export.coreml import _save_mlpackage_atomic

        destination = tmp_path / "model.mlpackage"
        destination.mkdir()
        sentinel = destination / "known-good.txt"
        sentinel.write_text("old", encoding="utf-8")
        model = MagicMock()

        def save(path):
            Path(path).mkdir(parents=True)

        model.save.side_effect = save

        def reject(_candidate):
            raise ValueError("staged ABI mismatch")

        with pytest.raises(ValueError, match="staged ABI mismatch"):
            _save_mlpackage_atomic(
                model,
                destination,
                validate_candidate=reject,
            )

        assert sentinel.read_text(encoding="utf-8") == "old"
        assert not list(tmp_path.glob(".*.staging"))


class TestFinalDeploymentSpec:
    @staticmethod
    def _tensor_feature(name, shape, *, dtype=65568):
        array = SimpleNamespace(shape=list(shape), dataType=dtype)
        feature_type = SimpleNamespace(
            isOptional=False,
            multiArrayType=array,
            WhichOneof=lambda _name: "multiArrayType",
        )
        return SimpleNamespace(name=name, type=feature_type)

    def _spec(self):
        return SimpleNamespace(
            description=SimpleNamespace(
                input=[
                    self._tensor_feature(
                        "image",
                        (1, 3, 32, 32),
                    )
                ],
                output=[
                    self._tensor_feature(
                        "prediction",
                        (1, 6, 21),
                    )
                ],
            )
        )

    def test_exact_tensor_boundary_passes(self):
        from libreyolo.export.coreml import (
            _validate_coreml_deployment_spec,
        )

        _validate_coreml_deployment_spec(
            self._spec(),
            input_contract={"name": "image", "kind": "tensor"},
            output_contract=[
                {"name": "prediction", "shape": [1, 6, 21]}
            ],
            input_shape=(1, 3, 32, 32),
            nms=False,
        )

    @pytest.mark.parametrize(
        ("mutation", "message"),
        [
            (
                lambda spec: setattr(
                    spec.description.input[0],
                    "name",
                    "other",
                ),
                "input name",
            ),
            (
                lambda spec: setattr(
                    spec.description.input[0].type.multiArrayType,
                    "shape",
                    [1, 3, 16, 16],
                ),
                "TensorType shape",
            ),
            (
                lambda spec: setattr(
                    spec.description.output[0].type.multiArrayType,
                    "shape",
                    [1, 6, 20],
                ),
                "changed shape",
            ),
            (
                lambda spec: setattr(
                    spec.description.output[0].type.multiArrayType,
                    "dataType",
                    65552,
                ),
                "must be FP32",
            ),
        ],
    )
    def test_tampered_boundary_fails(self, mutation, message):
        from libreyolo.export.coreml import (
            _validate_coreml_deployment_spec,
        )

        spec = self._spec()
        mutation(spec)
        with pytest.raises(RuntimeError, match=message):
            _validate_coreml_deployment_spec(
                spec,
                input_contract={"name": "image", "kind": "tensor"},
                output_contract=[
                    {
                        "name": "prediction",
                        "shape": [1, 6, 21],
                    }
                ],
                input_shape=(1, 3, 32, 32),
                nms=False,
            )


class TestTransactionalPreparation:
    def test_pool_is_restored_when_trace_fails(self, tmp_path, monkeypatch):
        _patch_ct(monkeypatch)
        from libreyolo.export import coreml

        model = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(1),
        ).eval()
        original_pool = model[0]
        monkeypatch.setattr(
            coreml.torch.jit,
            "trace",
            MagicMock(side_effect=RuntimeError("trace failed")),
        )

        with pytest.raises(RuntimeError, match="trace failed"):
            coreml.export_coreml(
                model,
                torch.randn(1, 3, 16, 16),
                output_path=str(tmp_path / "m.mlpackage"),
                model_family="resnet",
                model_task="classify",
                model_size="18",
                metadata=_strict_metadata(
                    "resnet",
                    "classify",
                    "18",
                    names={
                        "0": "class_0",
                        "1": "class_1",
                        "2": "class_2",
                    },
                    imgsz=16,
                ),
                compute_units="cpu_only",
            )
        assert model[0] is original_pool


class TestNMSWrap:
    def test_rfdetr_raises(self, tmp_path, monkeypatch):
        fake, mlmodel = _patch_ct(monkeypatch)
        from libreyolo.export.coreml import export_coreml

        with pytest.raises(NotImplementedError, match="RF-DETR"):
            export_coreml(
                _DummyModel().eval(),
                torch.randn(1, 3, 640, 640),
                output_path=str(tmp_path / "m.mlpackage"),
                precision="fp32",
                compute_units="all",
                nms=True,
                metadata={"model_family": "rfdetr"},
                model_family="rfdetr",
            )

    def test_rtdetr_raises(self, tmp_path, monkeypatch):
        _patch_ct(monkeypatch)
        from libreyolo.export.coreml import export_coreml

        with pytest.raises(NotImplementedError, match="RT-DETR"):
            export_coreml(
                _DummyModel().eval(),
                torch.randn(1, 3, 640, 640),
                output_path=str(tmp_path / "m.mlpackage"),
                precision="fp32",
                compute_units="all",
                nms=True,
                metadata={"model_family": "rtdetr"},
                model_family="rtdetr",
            )

    def test_yolox_calls_pipeline(self, tmp_path, monkeypatch):
        fake, mlmodel = _patch_ct(monkeypatch)

        from libreyolo.export import coreml as coreml_mod

        wrap = MagicMock(return_value=mlmodel)
        monkeypatch.setattr(coreml_mod, "_wrap_with_nms", wrap)

        coreml_mod.export_coreml(
            _DummyYoloxExportModel().eval(),
            torch.randn(1, 3, 640, 640),
            output_path=str(tmp_path / "m.mlpackage"),
            precision="fp32",
            compute_units="all",
            nms=True,
            metadata=_strict_metadata(
                "yolox",
                "detect",
                "n",
                names={str(index): f"class_{index}" for index in range(80)},
            ),
            model_family="yolox",
            model_task="detect",
            model_size="n",
        )
        kwargs = fake.convert.call_args.kwargs
        assert kwargs["outputs"] == [
            ("TensorType", {"name": "confidence"}),
            ("TensorType", {"name": "coordinates"}),
        ]
        wrap.assert_called_once_with(
            mlmodel,
            model_family="yolox",
            iou=0.45,
            conf=0.25,
            compute_units="ALL",
        )
        assert mlmodel.user_defined_metadata["nms"] == "True"


class TestCoreMLExporterRegistry:
    def test_format_registered(self):
        from libreyolo.export.exporter import BaseExporter, CoreMLExporter

        assert "coreml" in BaseExporter._registry
        assert BaseExporter._registry["coreml"] is CoreMLExporter

    def test_class_attrs(self):
        from libreyolo.export.exporter import CoreMLExporter

        assert CoreMLExporter.format_name == "coreml"
        assert CoreMLExporter.suffix == ".mlpackage"
        assert CoreMLExporter.requires_onnx is False
        assert CoreMLExporter.supports_int8 is False
        assert CoreMLExporter.apply_model_half is False

    @pytest.mark.parametrize("size", [None, "s", "m", "l", "x", "unknown"])
    def test_public_deimv2_route_rejects_licensed_sizes_before_model_context(
        self,
        size,
    ):
        from libreyolo.export.exporter import CoreMLExporter

        model = SimpleNamespace(
            task="detect",
            size=size,
            _get_model_name=lambda: "deimv2",
        )
        exporter = CoreMLExporter(model)
        model_context = MagicMock(
            side_effect=AssertionError("model context must not be entered")
        )
        exporter._model_context = model_context

        with pytest.raises(NotImplementedError, match="DINOv3 licensing boundary"):
            exporter(dynamic=False)

        model_context.assert_not_called()


class TestCoreMLBackendModule:
    def test_backend_class_importable(self):
        # On non-macOS, importing the class itself must succeed (only
        # instantiation should refuse). Use the lazy import path.
        import libreyolo

        assert hasattr(libreyolo, "CoreMLBackend")
        cls = libreyolo.CoreMLBackend
        assert cls.__name__ == "CoreMLBackend"

    def test_dispatch_mlpackage(self, tmp_path, monkeypatch):
        # Create a fake .mlpackage directory and ensure the model factory
        # routes it to CoreMLBackend (we patch the class to a sentinel).
        pkg = tmp_path / "fake.mlpackage"
        pkg.mkdir()

        sentinel = MagicMock(name="CoreMLBackendSentinel")
        import libreyolo.backends.coreml as coreml_mod
        import libreyolo.backends.coreml_facerec as coreml_facerec_mod

        monkeypatch.setattr(coreml_mod, "CoreMLBackend", sentinel)
        # This test owns only generic package dispatch.  A bare directory is
        # not a valid Core ML package, so keep the separate face-embedding
        # metadata discriminator out of scope.
        monkeypatch.setattr(
            coreml_facerec_mod,
            "coreml_package_family",
            lambda _path: None,
        )

        from libreyolo.models import LibreYOLO

        LibreYOLO(str(pkg), nb_classes=80, device="cpu")
        sentinel.assert_called_once()

    def test_backend_preserves_task_metadata(self, tmp_path, monkeypatch):
        fake, mlmodel = _patch_ct(monkeypatch)
        monkeypatch.setattr(sys, "platform", "darwin")

        pkg = tmp_path / "fake.mlpackage"
        pkg.mkdir()

        mlmodel.user_defined_metadata = {
            "libreyolo_producer": "libreyolo",
            "artifact_format": "coreml",
            "coreml_io_schema_version": "1",
            "schema_version": "1",
            "libreyolo_version": "0.0.1",
            "model_family": "dfine",
            "size": "n",
            "model_size": "n",
            "task": "segment",
            "default_task": "detect",
            "supported_tasks": '["detect", "segment"]',
            "names": '{"0": "person"}',
            "nc": "1",
            "imgsz": "512",
            "dynamic": "False",
            "coreml_io": (
                '{"input":{"name":"image","kind":"image","layout":"NCHW",'
                '"color":"rgb","range":"uint8","geometry":"stretch",'
                '"interpolation":"bilinear","resize_backend":"pillow","pad_value":0},'
                '"outputs":[{"name":"pred_logits","role":"class_logits",'
                '"rank":3,"dtype":"float32"},'
                '{"name":"pred_boxes","role":"boxes",'
                '"encoding":"cxcywh_normalized","rank":3,"dtype":"float32"},'
                '{"name":"pred_masks","role":"mask_probabilities",'
                '"encoding":"sigmoid_probabilities","rank":4,"dtype":"float32"}],'
                '"validation":{"color":"rgb","range":"0_255"}}'
            ),
        }
        package_spec = SimpleNamespace(
            description=SimpleNamespace(
                input=[
                    SimpleNamespace(
                        name="image",
                        type=SimpleNamespace(
                            WhichOneof=lambda _name: "imageType",
                            imageType=SimpleNamespace(width=512, height=512),
                        ),
                    )
                ],
                output=[
                    SimpleNamespace(name="pred_logits"),
                    SimpleNamespace(name="pred_boxes"),
                    SimpleNamespace(name="pred_masks"),
                ],
            )
        )
        package_spec.description.metadata = SimpleNamespace(
            userDefined=mlmodel.user_defined_metadata
        )
        mlmodel.get_spec.return_value = package_spec
        fake.utils.load_spec.side_effect = None
        fake.utils.load_spec.return_value = package_spec
        fake.models.MLModel.return_value = mlmodel

        from libreyolo.backends.coreml import CoreMLBackend

        backend = CoreMLBackend(
            str(pkg),
            nb_classes=80,
            compute_units="cpu_only",
        )

        assert backend.model_family == "dfine"
        assert backend.model_size == "n"
        assert backend.size == "n"
        assert backend.task == "segment"
        assert backend.DEFAULT_TASK == "detect"
        assert backend.SUPPORTED_TASKS == ("detect", "segment")
        assert backend.imgsz == 512
        assert backend.names == {0: "person"}

    def test_backend_parses_legacy_supported_tasks_repr(self):
        from libreyolo.backends.coreml import CoreMLBackend

        parsed = CoreMLBackend._parse_metadata(
            {
                "task": "segment",
                "default_task": "detect",
                "supported_tasks": "['detect', 'segment']",
            },
            default_nb_classes=80,
        )

        assert parsed[2] == "segment"
        assert parsed[3] == ("detect", "segment")
        assert parsed[4] == "detect"

    def test_backend_parses_rectangular_metadata(self):
        from libreyolo.backends.coreml import CoreMLBackend

        parsed = CoreMLBackend._parse_metadata(
            {
                "model_family": "yolo9",
                "imgsz": "640",
                "imgsz_h": "320",
                "imgsz_w": "640",
            },
            default_nb_classes=80,
        )

        assert parsed[6] == (320, 640)

    def test_backend_preprocess_accepts_rectangular_yolo9_imgsz(self):
        from libreyolo.backends.coreml import CoreMLBackend

        backend = CoreMLBackend.__new__(CoreMLBackend)
        backend.model_family = "yolo9"
        backend.input_contract = SimpleNamespace(
            geometry="letterbox_top_left",
            interpolation="bilinear",
            resize_backend="opencv",
            resize_long_side=None,
            resize_rounding="floor",
            pad_value=114,
            crop_pct=0.875,
            shape_mode="fixed",
        )

        tensor, _original_img, original_size, ratio = backend._preprocess(
            np.zeros((8, 16, 3), dtype=np.uint8),
            (32, 64),
            "rgb",
        )

        assert tuple(tensor.shape) == (1, 3, 32, 64)
        assert original_size == (16, 8)
        assert ratio == 4.0
