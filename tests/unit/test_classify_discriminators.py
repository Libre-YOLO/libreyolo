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
        LibreAlexNet,
        LibreConvNeXt,
        LibreDeiT,
        LibreEfficientNetV2,
        LibreMobileNetV4,
        LibreResNet,
        LibreVGG,
    )
    from torchvision.models import vgg16, vgg16_bn, vgg19, vgg19_bn

    cases = [
        (LibreResNet, "resnet18", "18"),
        (LibreResNet, "resnet34", "34"),
        (LibreResNet, "resnet50", "50"),
        (LibreResNet, "resnet101", "101"),
        (LibreConvNeXt, "convnext_tiny", "t"),
        (LibreConvNeXt, "convnext_small", "s"),
        (LibreConvNeXt, "convnext_base", "b"),
        (LibreDeiT, "deit_tiny_patch16_224", "t"),
        (LibreDeiT, "deit_small_patch16_224", "s"),
        (LibreDeiT, "deit_base_patch16_224", "b"),
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

    from torchvision.models import alexnet

    alexnet_sd = alexnet(weights=None).state_dict()
    assert LibreAlexNet.detect_size(alexnet_sd) == "b"
    assert LibreAlexNet.can_load(alexnet_sd) is True
    for cls, builder, expected in [
        (LibreVGG, vgg16, "16"),
        (LibreVGG, vgg19, "19"),
        (LibreVGG, vgg16_bn, "16bn"),
        (LibreVGG, vgg19_bn, "19bn"),
    ]:
        sd = builder(weights=None).state_dict()
        assert cls.detect_size(sd) == expected
        assert cls.can_load(sd) is True


def test_unsupported_variants_rejected():
    timm = pytest.importorskip("timm")
    from libreyolo import (
        LibreConvNeXt,
        LibreDeiT,
        LibreEfficientNetV2,
        LibreMobileNetV4,
        LibreResNet,
    )

    cases = [
        (LibreResNet, "resnet152"),                  # deeper bottleneck, must not be "101"
        (LibreConvNeXt, "convnext_large"),           # wider dims, must not be "b"
        (LibreDeiT, "deit_base_distilled_patch16_224"),  # extra token/head
        (LibreDeiT, "deit_base_patch16_384"),        # different pos_embed
        (LibreEfficientNetV2, "tf_efficientnetv2_s"),  # not a base tier (stem 24)
        (LibreMobileNetV4, "mobilenetv4_conv_small_050"),  # 0.5x width
        (LibreMobileNetV4, "mobilenetv4_hybrid_medium"),   # MQA attention variant
    ]
    for cls, tag in cases:
        sd = _sd(timm, tag)
        assert cls.detect_size(sd) is None, f"{cls.__name__} mis-sized unsupported {tag}"
        assert cls.can_load(sd) is False, f"{cls.__name__} wrongly claimed unsupported {tag}"


def test_classify_filenames_require_cls_suffix():
    """Classify families are classify-only, so the ``-cls`` suffix is mandatory
    in weight filenames; a suffixless name (e.g. ``LibreResNet50.pt``) must not
    be accepted as a classify checkpoint (detect families keep it optional)."""
    from libreyolo import (
        LibreAlexNet,
        LibreConvNeXt,
        LibreDeiT,
        LibreEfficientNetV2,
        LibreMobileNetV4,
        LibreResNet,
        LibreViT,
        LibreVGG,
    )

    for cls, size in [
        (LibreAlexNet, "b"),
        (LibreResNet, "50"),
        (LibreConvNeXt, "t"),
        (LibreDeiT, "t"),
        (LibreEfficientNetV2, "b0"),
        (LibreMobileNetV4, "s"),
        (LibreViT, "ti"),
        (LibreVGG, "16bn"),
    ]:
        stem = f"{cls.FILENAME_PREFIX}{size}"
        assert cls.detect_size_from_filename(f"{stem}-cls.pt") == size, (
            f"{cls.__name__} should accept the canonical -cls filename"
        )
        assert cls.detect_size_from_filename(f"{stem}.pt") is None, (
            f"{cls.__name__} must reject a suffixless filename"
        )


def test_vgg_and_existing_classifiers_reject_each_other():
    from libreyolo import (
        LibreConvNeXt,
        LibreEfficientNetV2,
        LibreMobileNetV4,
        LibreResNet,
        LibreVGG,
    )
    from libreyolo.models.convnext.nn import ConvNeXt
    from libreyolo.models.efficientnetv2.nn import EfficientNetV2
    from libreyolo.models.mobilenetv4.nn import MobileNetV4
    from libreyolo.models.resnet.nn import ResNet
    from libreyolo.models.vgg.nn import VGG

    vgg_state = VGG(size="16", num_classes=1000).state_dict()
    sibling_cases = [
        (LibreResNet, ResNet(size="18", num_classes=1000).state_dict()),
        (LibreConvNeXt, ConvNeXt(size="t", num_classes=1000).state_dict()),
        (
            LibreEfficientNetV2,
            EfficientNetV2(size="b0", num_classes=1000).state_dict(),
        ),
        (
            LibreMobileNetV4,
            MobileNetV4(size="s", num_classes=1000).state_dict(),
        ),
    ]
    for sibling, sibling_state in sibling_cases:
        assert LibreVGG.can_load(sibling_state) is False
        assert sibling.can_load(vgg_state) is False
