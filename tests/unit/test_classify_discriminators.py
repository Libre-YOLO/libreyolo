"""Registry-safety: the classify families claim only their exact shipped sizes.

Builds architecture-only (pretrained=False, no network) timm state dicts for
both supported and unsupported variants and checks can_load / detect_size. This
guards autoconvert (which asks every family to claim arbitrary upstream weights)
from mis-claiming or mis-sizing same-architecture checkpoints.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


def _sd(timm, tag):
    return timm.create_model(tag, pretrained=False).state_dict()


def test_supported_variants_detected():
    timm = pytest.importorskip("timm")
    from libreyolo import (
        LibreConvNeXt,
        LibreEfficientNetV2,
        LibreMobileNetV4,
        LibreResNet,
    )

    cases = [
        (LibreResNet, "resnet18", "18"),
        (LibreResNet, "resnet34", "34"),
        (LibreResNet, "resnet50", "50"),
        (LibreResNet, "resnet101", "101"),
        (LibreConvNeXt, "convnext_tiny", "t"),
        (LibreConvNeXt, "convnext_small", "s"),
        (LibreConvNeXt, "convnext_base", "b"),
        (LibreEfficientNetV2, "tf_efficientnetv2_b0", "b0"),
        (LibreEfficientNetV2, "tf_efficientnetv2_b1", "b1"),
        (LibreEfficientNetV2, "tf_efficientnetv2_b2", "b2"),
        (LibreEfficientNetV2, "tf_efficientnetv2_b3", "b3"),
        (LibreMobileNetV4, "mobilenetv4_conv_small", "s"),
        (LibreMobileNetV4, "mobilenetv4_conv_medium", "m"),
        (LibreMobileNetV4, "mobilenetv4_conv_large", "l"),
    ]
    for cls, tag, expected in cases:
        sd = _sd(timm, tag)
        assert cls.detect_size(sd) == expected, f"{cls.__name__}: {tag} -> {cls.detect_size(sd)} != {expected}"
        assert cls.can_load(sd) is True, f"{cls.__name__} should claim {tag}"


def test_unsupported_variants_rejected():
    timm = pytest.importorskip("timm")
    from libreyolo import (
        LibreConvNeXt,
        LibreEfficientNetV2,
        LibreMobileNetV4,
        LibreResNet,
    )

    cases = [
        (LibreResNet, "resnet152"),                  # deeper bottleneck, must not be "101"
        (LibreConvNeXt, "convnext_large"),           # wider dims, must not be "b"
        (LibreEfficientNetV2, "tf_efficientnetv2_s"),  # not a base tier (stem 24)
        (LibreMobileNetV4, "mobilenetv4_conv_small_050"),  # 0.5x width
        (LibreMobileNetV4, "mobilenetv4_hybrid_medium"),   # MQA attention variant
    ]
    for cls, tag in cases:
        sd = _sd(timm, tag)
        assert cls.detect_size(sd) is None, f"{cls.__name__} mis-sized unsupported {tag}"
        assert cls.can_load(sd) is False, f"{cls.__name__} wrongly claimed unsupported {tag}"
