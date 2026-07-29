"""Hermetic coverage for Depth Anything 3's raw Core ML contract."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

pytestmark = pytest.mark.unit


class _TinyPositionEncoder(nn.Module):
    patch_size = 14

    def __init__(self):
        super().__init__()
        self.pos_embed = nn.Parameter(
            torch.linspace(0.0, 1.0, (1 + 37 * 37) * 2).reshape(
                1,
                1 + 37 * 37,
                2,
            )
        )

    def interpolate_pos_encoding(self, tokens, width, height):
        del tokens
        class_position = self.pos_embed[:, :1]
        patch_position = self.pos_embed[:, 1:].reshape(1, 37, 37, 2)
        patch_position = F.interpolate(
            patch_position.permute(0, 3, 1, 2),
            size=(width // self.patch_size, height // self.patch_size),
            mode="bicubic",
        )
        patch_position = patch_position.permute(0, 2, 3, 1).reshape(
            1,
            -1,
            2,
        )
        return torch.cat((class_position, patch_position), dim=1)


class _TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.pretrained = _TinyPositionEncoder()

    def forward(self, image, *, export_feat_layers):
        assert export_feat_layers == []
        return [image], []


class _TinyHead(nn.Module):
    use_sky_head = True
    out_dim = 1
    activation = "exp"
    sky_activation = "relu"
    down_ratio = 1

    def forward(self, features, height, width, patch_start_idx):
        assert patch_start_idx == 0
        image = features[0]
        depth = image[:, 0, 0].abs().add(0.25)
        sky = image[:, 0, 1].relu()
        return {
            "depth": depth.reshape(image.shape[0], 1, height, width),
            "sky": sky.reshape(image.shape[0], 1, height, width),
        }


class _TinyDA3(nn.Module):
    PATCH_SIZE = 14

    def __init__(self):
        super().__init__()
        from libreyolo.models.depth_anything3.nn import (
            IMAGENET_MEAN,
            IMAGENET_STD,
        )

        self.backbone = _TinyBackbone()
        self.head = _TinyHead()
        self.register_buffer(
            "pixel_mean",
            torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "pixel_std",
            torch.tensor(IMAGENET_STD).view(1, 3, 1, 1),
            persistent=False,
        )

    def forward(self, image):
        raise AssertionError("The raw Core ML adapter must bypass native forward.")


class _Shell(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model


def test_profile_is_fail_closed_and_unwraps_export_shell():
    from libreyolo.export.coreml_depth_anything3 import (
        validate_depth_anything3_coreml_profile,
    )

    model = _TinyDA3().eval()
    wrapped = _Shell(_Shell(model))
    assert (
        validate_depth_anything3_coreml_profile(
            wrapped,
            size="l",
            canvas_hw=(504, 504),
        )
        is model
    )
    with pytest.raises(NotImplementedError, match="size='l'"):
        validate_depth_anything3_coreml_profile(
            model,
            size="b",
            canvas_hw=(504, 504),
        )
    with pytest.raises(NotImplementedError, match="504x504"):
        validate_depth_anything3_coreml_profile(
            model,
            size="l",
            canvas_hw=(490, 504),
        )

    model.head.sky_activation = "sigmoid"
    with pytest.raises(RuntimeError, match="mismatches"):
        validate_depth_anything3_coreml_profile(
            model,
            size="l",
            canvas_hw=504,
        )


def test_adapter_exposes_deterministic_pre_sky_outputs():
    from libreyolo.export.coreml_depth_anything3 import (
        wrap_depth_anything3_coreml_contract,
    )

    model = _TinyDA3().eval()
    adapter = wrap_depth_anything3_coreml_contract(model)
    probe = torch.linspace(0.0, 1.0, 3 * 28 * 42).reshape(1, 3, 28, 42)
    relative_depth, sky_score = adapter(probe)

    normalized = (probe - model.pixel_mean) / model.pixel_std
    assert torch.equal(
        relative_depth,
        normalized[:, 0:1].abs().add(0.25),
    )
    assert torch.equal(sky_score, normalized[:, 1:2].relu())
    assert tuple(relative_depth.shape) == (1, 1, 28, 42)
    assert tuple(sky_score.shape) == (1, 1, 28, 42)


def test_position_embedding_is_eagerly_baked_and_idempotent():
    from libreyolo.export.coreml_depth_anything3 import (
        freeze_depth_anything3_coreml_position_embedding,
    )

    model = _TinyDA3().eval()
    encoder = model.backbone.pretrained
    probe = torch.empty(1, 1 + 36 * 36, 2)
    expected = encoder.interpolate_pos_encoding(probe, 504, 504)

    assert freeze_depth_anything3_coreml_position_embedding(model) is model
    assert tuple(encoder.pos_embed.shape) == (1, 1 + 36 * 36, 2)
    assert torch.equal(encoder.pos_embed, expected)
    parameter = encoder.pos_embed
    assert freeze_depth_anything3_coreml_position_embedding(model) is model
    assert encoder.pos_embed is parameter


def test_adapter_is_strict_torchscript_and_torch_export_capturable():
    from libreyolo.export.coreml_depth_anything3 import (
        wrap_depth_anything3_coreml_contract,
    )

    adapter = wrap_depth_anything3_coreml_contract(_TinyDA3().eval())
    probe = torch.rand(1, 3, 28, 42)
    check = 1.0 - probe
    traced = torch.jit.trace(
        adapter,
        probe,
        check_trace=True,
        check_inputs=[(check,)],
    )
    expected = adapter(check)
    actual = traced(check)
    assert all(torch.equal(left, right) for left, right in zip(actual, expected))

    exported = torch.export.export(adapter, (probe,), strict=True)
    exported_actual = exported.module()(check)
    assert all(
        torch.equal(left, right)
        for left, right in zip(exported_actual, expected)
    )


def test_host_postprocess_matches_native_rule_without_sampling():
    from libreyolo.export.coreml_depth_anything3 import (
        postprocess_depth_anything3_coreml,
    )
    from libreyolo.models.depth_anything3.nn import LibreDepthAnything3Net

    depth = torch.linspace(0.2, 3.0, 2 * 1 * 20 * 20).reshape(2, 1, 20, 20)
    sky = torch.zeros_like(depth)
    sky[:, :, :5] = 1.0

    native = LibreDepthAnything3Net._apply_mono_sky(depth, sky)
    expected = torch.reciprocal(native.clamp_min(1e-6))
    actual = postprocess_depth_anything3_coreml(depth, sky)
    assert torch.equal(actual, expected)


def test_host_postprocess_matches_native_random_sampling_path():
    from libreyolo.export.coreml_depth_anything3 import (
        postprocess_depth_anything3_coreml,
    )
    from libreyolo.models.depth_anything3.nn import LibreDepthAnything3Net

    depth = torch.linspace(0.1, 9.0, 1 * 1 * 400 * 400).reshape(1, 1, 400, 400)
    sky = torch.zeros_like(depth)
    sky[:, :, :100] = 1.0

    torch.manual_seed(1741)
    native = LibreDepthAnything3Net._apply_mono_sky(depth, sky)
    expected = torch.reciprocal(native.clamp_min(1e-6))
    torch.manual_seed(1741)
    actual = postprocess_depth_anything3_coreml(depth, sky)
    assert torch.equal(actual, expected)


def test_host_postprocess_validates_before_data_dependent_operations():
    from libreyolo.export.coreml_depth_anything3 import (
        postprocess_depth_anything3_coreml,
    )

    valid = torch.ones(1, 1, 14, 14)
    with pytest.raises(ValueError, match=r"\[B, 1, H, W\]"):
        postprocess_depth_anything3_coreml(valid[:, 0], valid[:, 0])
    with pytest.raises(ValueError, match="same"):
        postprocess_depth_anything3_coreml(valid, valid[:, :, :-1])
    invalid = valid.clone()
    invalid[..., 0, 0] = float("nan")
    with pytest.raises(ValueError, match="NaN or infinity"):
        postprocess_depth_anything3_coreml(invalid, valid)
    negative = valid.clone()
    negative[..., 0, 0] = -0.01
    with pytest.raises(ValueError, match="non-negative"):
        postprocess_depth_anything3_coreml(negative, valid)
    with pytest.raises(ValueError, match="non-negative"):
        postprocess_depth_anything3_coreml(valid, negative)


def test_contract_declares_raw_abi_host_algorithm_and_geometry():
    from libreyolo.export import coreml
    from libreyolo.export.coreml_depth_anything3 import (
        DEPTH_ANYTHING3_COREML_CONTRACT,
        DEPTH_ANYTHING3_COREML_HOST_POSTPROCESS,
        depth_anything3_coreml_input_contract,
        depth_anything3_coreml_metadata,
        depth_anything3_coreml_output_contract,
        depth_anything3_coreml_validation_contract,
        expected_depth_anything3_coreml_shapes,
        validate_depth_anything3_coreml_metadata,
    )

    assert depth_anything3_coreml_input_contract() == {
        "name": "image",
        "kind": "image",
        "layout": "NCHW",
        "color": "rgb",
        "range": "uint8",
        "geometry": "stretch",
        "interpolation": "bilinear",
        "resize_backend": "opencv",
        "pad_value": 0,
    }
    assert depth_anything3_coreml_validation_contract() == {
        "color": "rgb",
        "range": "0_1",
    }
    assert ("depth_anything3", "depth") in coreml.supported_coreml_exports()
    assert coreml._input_contract("depth_anything3", "depth", "l") == (
        depth_anything3_coreml_input_contract()
    )
    assert coreml._output_contract(
        "depth_anything3",
        "depth",
        nms=False,
    ) == depth_anything3_coreml_output_contract()
    assert coreml._validation_contract("depth_anything3", "depth") == (
        depth_anything3_coreml_validation_contract()
    )
    outputs = depth_anything3_coreml_output_contract()
    assert [item["name"] for item in outputs] == [
        "relative_depth",
        "sky_score",
    ]
    assert [item["role"] for item in outputs] == [
        "relative_depth",
        "sky_score",
    ]
    assert expected_depth_anything3_coreml_shapes(
        batch=1,
        canvas_hw=504,
    ) == {
        "relative_depth": (1, 1, 504, 504),
        "sky_score": (1, 1, 504, 504),
    }

    metadata = depth_anything3_coreml_metadata()
    assert metadata["depth_anything3_contract"] == (
        DEPTH_ANYTHING3_COREML_CONTRACT
    )
    assert metadata["depth_anything3_host_postprocess"] == (
        DEPTH_ANYTHING3_COREML_HOST_POSTPROCESS
    )
    assert metadata["depth_anything3_sky_sampling"] == (
        "random_with_replacement"
    )
    assert metadata["depth_anything3_non_square_geometry"] == (
        "fixed_stretch_approximation"
    )
    validate_depth_anything3_coreml_metadata(metadata)
    validate_depth_anything3_coreml_metadata(
        {key: str(value) for key, value in metadata.items()}
    )
    tampered = dict(metadata)
    tampered["depth_anything3_far_quantile"] = 0.95
    with pytest.raises(ValueError, match="far_quantile"):
        validate_depth_anything3_coreml_metadata(tampered)


def test_coreml_model_context_keeps_position_bake_off_live_model():
    from libreyolo.export.coreml import _wrap_coreml_contract
    from libreyolo.export.exporter import CoreMLExporter

    class _ModelShell:
        task = "depth"
        size = "l"
        device = torch.device("cpu")

        def __init__(self):
            self.model = _TinyDA3().eval()

        @staticmethod
        def _get_model_name():
            return "depth_anything3"

    shell = _ModelShell()
    live_position = shell.model.backbone.pretrained.pos_embed
    assert tuple(live_position.shape) == (1, 1 + 37 * 37, 2)

    exporter = CoreMLExporter(shell)
    with exporter._model_context(
        torch.device("cpu"),
        False,
        False,
        1,
        (504, 504),
    ) as (prepared, _dummy):
        assert prepared is not shell.model
        wrapped = _wrap_coreml_contract(
            prepared,
            "depth_anything3",
            "depth",
        )
        assert tuple(wrapped.model.backbone.pretrained.pos_embed.shape) == (
            1,
            1 + 36 * 36,
            2,
        )

    assert shell.model.backbone.pretrained.pos_embed is live_position
    assert tuple(live_position.shape) == (1, 1 + 37 * 37, 2)


def test_shared_exporter_pins_depth_anything3_raw_output_semantics():
    from libreyolo.export.coreml import (
        _output_contract,
        _validate_output_semantics,
    )

    outputs = _output_contract("depth_anything3", "depth", nms=False)
    depth = torch.ones(1, 1, 504, 504)
    sky = torch.zeros_like(depth)
    _validate_output_semantics(
        outputs,
        [depth, sky],
        family="depth_anything3",
        task="depth",
        nc=1,
        input_hw=(504, 504),
        size="l",
        nms=False,
        metadata={},
    )

    invalid_sky = sky.clone()
    invalid_sky[..., 0, 0] = -1.0
    with pytest.raises(RuntimeError, match="non-negative"):
        _validate_output_semantics(
            outputs,
            [depth, invalid_sky],
            family="depth_anything3",
            task="depth",
            nc=1,
            input_hw=(504, 504),
            size="l",
            nms=False,
            metadata={},
        )


def test_public_model_routes_only_coreml_to_the_shared_exporter(monkeypatch):
    from libreyolo.export.exporter import BaseExporter
    from libreyolo.models.depth_anything3.model import LibreDepthAnything3

    calls = {}

    class _Exporter:
        def __call__(self, **kwargs):
            calls["kwargs"] = kwargs
            return "da3.mlpackage"

    model = object.__new__(LibreDepthAnything3)

    def fake_create(format_name, received):
        calls["format"] = format_name
        calls["model"] = received
        return _Exporter()

    monkeypatch.setattr(BaseExporter, "create", staticmethod(fake_create))
    assert model.export(
        format=" CoreML ",
        imgsz=504,
        half=True,
    ) == "da3.mlpackage"
    assert calls == {
        "format": "coreml",
        "model": model,
        "kwargs": {"imgsz": 504, "half": True},
    }
    with pytest.raises(NotImplementedError, match="only.*Core ML"):
        model.export(format="onnx")
