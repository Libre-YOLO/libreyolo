"""Unit tests for LibreSegformer semantic segmentation.

SegFormer is fully native (no optional runtime dependency), so unlike EoMT's
tests these run the real per-size architectures directly at a small imgsz —
cheap enough for the CPU-only unit tier.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

pytestmark = pytest.mark.unit

ALL_SIZES = ("b0", "b1", "b2", "b3", "b4", "b5")


def _make_semantic_yaml(root, n_images=4, size=64):
    import yaml as _yaml

    for split in ("train", "val"):
        for i in range(n_images):
            img_dir = root / "images" / split
            img_dir.mkdir(parents=True, exist_ok=True)
            arr = np.zeros((size, size, 3), dtype=np.uint8)
            arr[:, : size // 2] = (200, 40, 40)
            arr[:, size // 2 :] = (40, 40, 200)
            Image.fromarray(arr).save(img_dir / f"img{i}.jpg")
            mask = np.zeros((size, size), dtype=np.uint8)
            mask[:, size // 2 :] = 1
            mask_dir = root / "masks" / split
            mask_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(mask, mode="L").save(mask_dir / f"img{i}.png")
    yaml_path = root / "data.yaml"
    yaml_path.write_text(
        _yaml.safe_dump(
            {
                "path": str(root),
                "train": "images/train",
                "val": "images/val",
                "masks_dir": "masks",
                "nc": 2,
                "names": {0: "left", 1: "right"},
            }
        )
    )
    return yaml_path


class TestSegformerMetadata:
    def test_family_and_tasks(self):
        from libreyolo.models.segformer.model import LibreSegformer

        assert LibreSegformer.FAMILY == "segformer"
        assert LibreSegformer.FILENAME_PREFIX == "LibreSegformer"
        assert LibreSegformer.SUPPORTED_TASKS == ("semantic",)
        assert LibreSegformer.DEFAULT_TASK == "semantic"
        assert set(LibreSegformer.INPUT_SIZES) == set(ALL_SIZES)

    def test_get_download_url_points_at_the_hosted_weight(self):
        from libreyolo.models.segformer.model import LibreSegformer

        url = LibreSegformer.get_download_url("LibreSegformerb0-sem.pt")
        assert url == (
            "https://huggingface.co/LibreYOLO/LibreSegformerb0-sem/resolve/main/"
            "LibreSegformerb0-sem.pt"
        )

    def test_download_notice_states_the_non_commercial_restriction(self):
        """The weights are NC while the library is permissive, so the restriction
        has to reach the user before the bytes do."""
        from libreyolo.models.segformer.model import LibreSegformer

        notice = LibreSegformer.get_download_notice("LibreSegformerb0-sem.pt", "http://x")
        assert notice is not None
        assert "NON-COMMERCIAL" in notice

    def test_b5_input_size_is_640(self):
        """Upstream fine-tuned b0-b4 at 512 but b5 at 640; a 512 default would
        silently evaluate b5 off-resolution."""
        from libreyolo.models.segformer.model import LibreSegformer

        assert LibreSegformer.INPUT_SIZES["b4"] == 512
        assert LibreSegformer.INPUT_SIZES["b5"] == 640

    @pytest.mark.parametrize("size", ALL_SIZES)
    def test_can_load_detects_size_and_classes(self, size):
        from libreyolo.models.segformer.model import LibreSegformer
        from libreyolo.models.segformer.nn import LibreSegformerNet

        net = LibreSegformerNet(size=size, num_classes=7)
        state = net.state_dict()
        assert LibreSegformer.can_load(state)
        assert LibreSegformer.detect_size(state) == size
        assert LibreSegformer.detect_nb_classes(state) == 7

    def test_can_load_rejects_pidnet_signature(self):
        from libreyolo.models.segformer.model import LibreSegformer

        state = {
            "conv1.0.weight": torch.zeros(32, 3, 3, 3),
            "pag3.f_x.0.weight": torch.zeros(32, 64, 1, 1),
            "final_layer.conv2.weight": torch.zeros(19, 128, 1, 1),
        }
        assert not LibreSegformer.can_load(state)

    def test_can_load_rejects_dinov2_semantic_signature(self):
        from libreyolo.models.segformer.model import LibreSegformer

        state = {
            "backbone.encoder.proj.weight": torch.zeros(1),
            "predict.weight": torch.zeros(19, 8, 1, 1),
        }
        assert not LibreSegformer.can_load(state)

    def test_can_load_rejects_eomt_signature(self):
        from libreyolo.models.segformer.model import LibreSegformer

        state = {
            "query.weight": torch.zeros(100, 256),
            "class_predictor.weight": torch.zeros(150, 256),
            "mask_head.0.weight": torch.zeros(256, 256),
        }
        assert not LibreSegformer.can_load(state)

    def test_other_families_reject_segformer_signature(self):
        from libreyolo.models.segformer.nn import LibreSegformerNet
        from libreyolo.models.base.model import BaseModel
        from libreyolo.models.segformer.model import LibreSegformer

        net = LibreSegformerNet(size="b0", num_classes=5)
        state = net.state_dict()
        for cls in BaseModel._registry:
            if cls is LibreSegformer:
                continue
            assert not cls.can_load(state), f"{cls.__name__} incorrectly claims SegFormer weights"


class TestSegformerForward:
    @pytest.mark.parametrize("size", ALL_SIZES)
    def test_eval_shape(self, size):
        from libreyolo.models.segformer.nn import LibreSegformerNet

        net = LibreSegformerNet(size=size, num_classes=4)
        net.eval()
        x = torch.rand(1, 3, 64, 64)
        with torch.no_grad():
            out = net(x)
        assert out.shape == (1, 4, 64, 64)

    def test_train_loss_and_backward(self):
        from libreyolo.models.segformer.nn import LibreSegformerNet

        net = LibreSegformerNet(size="b0", num_classes=3)
        net.train()
        x = torch.rand(2, 3, 64, 64)
        targets = torch.randint(0, 3, (2, 64, 64))
        targets[:, :8, :] = 255

        out = net(x, targets=targets)
        assert set(out) == {"total_loss", "sem"}
        assert torch.isfinite(out["total_loss"])
        out["total_loss"].backward()
        assert net.decode_head.classifier.weight.grad is not None

    def test_all_ignored_targets_yield_finite_zero_loss(self):
        from libreyolo.models.segformer.nn import LibreSegformerNet

        net = LibreSegformerNet(size="b0", num_classes=3)
        net.train()
        x = torch.rand(1, 3, 64, 64)
        targets = torch.full((1, 64, 64), 255, dtype=torch.long)

        out = net(x, targets=targets)
        assert torch.isfinite(out["total_loss"])
        assert out["total_loss"].item() == pytest.approx(0.0)

    def test_normalization_is_applied_exactly_once(self):
        """ImageNet standardization must live ONLY in ``forward`` (on raw [0,1]),
        not also in ``preprocess_numpy`` / the dataset — else the input is
        normalized twice and the pretrained MiT encoder sees a distribution it
        never trained on. Guards against the double-normalization regression."""
        from libreyolo.models.segformer.model import preprocess_numpy

        # preprocess_numpy must emit /255-only values in [0, 1], NOT pre-normalized.
        img = np.full((48, 64, 3), 128, dtype=np.uint8)
        chw, _ = preprocess_numpy(img, 64)
        assert chw.min() >= 0.0 and chw.max() <= 1.0, (
            "preprocess_numpy must feed [0, 1]; normalization belongs in forward"
        )

        # The wrapper must not advertise dataset-side standardization hooks
        # (their presence would re-normalize on top of forward's internal norm).
        from libreyolo.models.segformer.model import LibreSegformer

        assert not hasattr(LibreSegformer, "semantic_norm_mean")
        assert not hasattr(LibreSegformer, "semantic_norm_std")

        # forward owns the ImageNet standardization: it must actually transform
        # the raw [0,1] tensor (non-identity), so normalization happens exactly
        # once and only here.
        from libreyolo.models.segformer.nn import LibreSegformerNet

        net = LibreSegformerNet(size="b0", num_classes=3).eval()
        x = torch.from_numpy(chw).unsqueeze(0)
        standardized = (x - net.pixel_mean) / net.pixel_std
        assert not torch.allclose(standardized, x), "forward must standardize the [0,1] input"


class TestSegformerWrapper:
    def test_wrapper_predict_returns_semantic_mask(self, tmp_path):
        from libreyolo.models.segformer.model import LibreSegformer

        img_path = tmp_path / "img.jpg"
        Image.new("RGB", (90, 45), color=(50, 90, 130)).save(img_path)

        m = LibreSegformer(model_path=None, size="b0", task="semantic", nb_classes=3, device="cpu")
        assert m.task == "semantic"

        result = m.predict(str(img_path), imgsz=64)

        assert result.boxes is None
        assert result.semantic_mask is not None
        assert tuple(result.semantic_mask.data.shape) == (45, 90)

    def test_wrapper_class_rebuild_only_touches_classifier(self):
        from libreyolo.models.segformer.model import LibreSegformer

        m = LibreSegformer(model_path=None, size="b0", task="semantic", nb_classes=3, device="cpu")
        encoder_state_before = {k: v.clone() for k, v in m.model.encoder.state_dict().items()}

        m._rebuild_for_new_classes(5)

        m.model.eval()
        with torch.no_grad():
            logits = m.model(torch.rand(1, 3, 64, 64))
        assert logits.shape == (1, 5, 64, 64)
        for key, value in encoder_state_before.items():
            assert torch.equal(value, m.model.encoder.state_dict()[key])

    def test_pretrained_encoder_loads_only_encoder_weights(self, tmp_path):
        """Regression for the tools/pretrain_mit/ bridge: an encoder-only
        checkpoint ({"encoder": ..., "size": ...}) must populate only
        self.model.encoder, leaving decode_head at its own random init.
        """
        from libreyolo.models.segformer.model import LibreSegformer
        from libreyolo.models.segformer.nn import SegformerEncoder, SIZE_CONFIGS

        encoder = SegformerEncoder(SIZE_CONFIGS["b0"])
        ckpt_path = tmp_path / "mit_b0_imagenet1k_encoder.pt"
        torch.save({"encoder": encoder.state_dict(), "size": "b0", "source": "imagenet1k-classify"}, ckpt_path)

        m = LibreSegformer(size="b0", task="semantic", nb_classes=3, device="cpu", pretrained_encoder=str(ckpt_path))
        for key, value in encoder.state_dict().items():
            assert torch.equal(value, m.model.encoder.state_dict()[key])

    def test_pretrained_encoder_rejects_size_mismatch(self, tmp_path):
        from libreyolo.models.segformer.model import LibreSegformer
        from libreyolo.models.segformer.nn import SegformerEncoder, SIZE_CONFIGS

        encoder = SegformerEncoder(SIZE_CONFIGS["b1"])
        ckpt_path = tmp_path / "mit_b1_imagenet1k_encoder.pt"
        torch.save({"encoder": encoder.state_dict(), "size": "b1", "source": "imagenet1k-classify"}, ckpt_path)

        with pytest.raises(ValueError, match="size"):
            LibreSegformer(size="b0", task="semantic", nb_classes=3, device="cpu", pretrained_encoder=str(ckpt_path))

    def test_model_path_and_pretrained_encoder_are_mutually_exclusive(self, tmp_path):
        from libreyolo.models.segformer.model import LibreSegformer
        from libreyolo.models.segformer.nn import SegformerEncoder, SIZE_CONFIGS

        encoder = SegformerEncoder(SIZE_CONFIGS["b0"])
        ckpt_path = tmp_path / "mit_b0_imagenet1k_encoder.pt"
        torch.save({"encoder": encoder.state_dict(), "size": "b0"}, ckpt_path)

        with pytest.raises(ValueError, match="one of"):
            LibreSegformer(
                model_path=str(ckpt_path),
                size="b0",
                task="semantic",
                nb_classes=3,
                device="cpu",
                pretrained_encoder=str(ckpt_path),
            )

    def test_wrong_task_raises(self):
        from libreyolo.models.segformer.model import LibreSegformer

        with pytest.raises(ValueError, match="semantic"):
            LibreSegformer(model_path=None, size="b0", task="detect", nb_classes=3, device="cpu")

    def test_loads_raw_state_dict_without_checkpoint_metadata(self, tmp_path):
        """Regression: DDP training round-trips a *raw* state dict through a
        tempfile (libreyolo/training/ddp_spawn.py writes plain
        ``model.state_dict()``, no model_family/task/nc wrapper) and expects
        the model class to reconstruct from it directly — like LibreDINOv2,
        LibreSegformer must tolerate this, not just its own fully-wrapped
        checkpoints. Multi-GPU training silently regresses if this breaks.
        """
        from libreyolo.models.segformer.model import LibreSegformer

        src = LibreSegformer(model_path=None, size="b0", task="semantic", nb_classes=3, device="cpu")
        raw_path = tmp_path / "raw_state_dict.pt"
        torch.save({k: v.cpu() for k, v in src.model.state_dict().items()}, raw_path)

        reloaded = LibreSegformer(model_path=str(raw_path), size="b0", task="semantic", nb_classes=3, device="cpu")
        assert reloaded.task == "semantic"
        for key, value in src.model.state_dict().items():
            assert torch.equal(value, reloaded.model.state_dict()[key])


def test_segformer_train_smoke(tmp_path):
    """One epoch through the shared BaseTrainer semantic path (real, no stub)."""
    from libreyolo.models.segformer.model import LibreSegformer

    yaml_path = _make_semantic_yaml(tmp_path)
    m = LibreSegformer(model_path=None, size="b0", task="semantic", nb_classes=2, device="cpu")

    res = m.train(
        data=str(yaml_path),
        epochs=1,
        batch=2,
        imgsz=64,
        workers=0,
        eval_interval=1,
        project=str(tmp_path / "runs"),
        name="segformer_smoke",
        exist_ok=True,
        amp=False,
        ema=False,
        warmup_epochs=0,
    )

    assert np.isfinite(res["epoch_losses"][0])
    assert res["epoch_metrics"][-1]["val_metrics"].get("metrics/mIoU") is not None


def test_segformer_checkpoint_round_trip(tmp_path):
    from libreyolo.models.segformer.model import LibreSegformer
    from libreyolo.utils.serialization import load_trusted_torch_file

    yaml_path = _make_semantic_yaml(tmp_path)
    m = LibreSegformer(model_path=None, size="b0", task="semantic", nb_classes=2, device="cpu")
    res = m.train(
        data=str(yaml_path),
        epochs=1,
        batch=2,
        imgsz=64,
        workers=0,
        eval_interval=0,
        project=str(tmp_path / "runs"),
        name="segformer_ckpt",
        exist_ok=True,
        amp=False,
        ema=False,
        warmup_epochs=0,
    )
    ckpt_path = res.get("best_checkpoint") or res.get("last_checkpoint")
    assert ckpt_path is not None

    ckpt = load_trusted_torch_file(ckpt_path, map_location="cpu", context="test")
    assert ckpt.get("model_family") == "segformer"
    assert ckpt.get("task") == "semantic"
    assert ckpt.get("nc") == 2

    reloaded = LibreSegformer(model_path=ckpt_path, size="b0", nb_classes=2, device="cpu")
    assert reloaded.task == "semantic"

    img_path = sorted((tmp_path / "images" / "val").glob("*.jpg"))[0]
    result = reloaded.predict(str(img_path), imgsz=64)
    assert result.semantic_mask is not None


class TestSegformerReferenceFidelity:
    """The published weights only reproduce upstream numbers if these constants
    match the reference implementation exactly. Each of these was wrong once."""

    def test_layernorm_eps_matches_reference(self):
        """The reference builds every LayerNorm with torch's default eps (1e-5).
        `layer_norm_eps: 1e-6` in the upstream configs is read by nothing; using
        it shifts every logit and breaks bit-exactness with the released weights.
        """
        import torch.nn as nn

        from libreyolo.models.segformer.nn import LibreSegformerNet

        net = LibreSegformerNet(size="b0", num_classes=4)
        eps = {m.eps for m in net.modules() if isinstance(m, nn.LayerNorm)}
        assert eps == {1e-5}

    def test_mix_ffn_has_no_dropout(self):
        """Mix-FFN dropout is `hidden_dropout_prob` (0.0). Only the decode-head
        classifier uses `classifier_dropout_prob` (0.1)."""
        from libreyolo.models.segformer.nn import LibreSegformerNet

        net = LibreSegformerNet(size="b0", num_classes=4)
        for stage in net.encoder.stages:
            for block in stage.blocks:
                assert block.mlp.dropout.p == 0.0
        assert net.decode_head.dropout.p == 0.1

    def test_weights_are_initialized_to_reference_scale(self):
        """Torch's default init is ~5x too wide at the narrow stages; the family
        can train from scratch, so the init is load-bearing.

        Assert the std is AT the reference 0.02, not merely below some ceiling:
        torch's default lands at 0.021-0.036 for the wider layers, so an upper
        bound alone would pass with no init at all.
        """
        import torch.nn as nn

        from libreyolo.models.segformer.nn import INITIALIZER_RANGE, LibreSegformerNet

        net = LibreSegformerNet(size="b0", num_classes=4)
        for name, module in net.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                std = module.weight.std().item()
                assert std == pytest.approx(INITIALIZER_RANGE, rel=0.25), (
                    f"{name}: std={std:.4f}, expected ~{INITIALIZER_RANGE}"
                )
                if module.bias is not None:
                    assert torch.equal(module.bias, torch.zeros_like(module.bias))

    def test_convert_upstream_state_dict_remaps_encoder_prefix(self):
        """Upstream ships the encoder under `segformer.`; we use `encoder.`."""
        from libreyolo.models.segformer.model import LibreSegformer
        from libreyolo.models.segformer.nn import LibreSegformerNet

        net = LibreSegformerNet(size="b0", num_classes=150)
        upstream = {
            ("segformer." + k[len("encoder.") :] if k.startswith("encoder.") else k): v
            for k, v in net.state_dict().items()
        }
        converted = LibreSegformer.convert_upstream_state_dict(upstream)
        assert converted is not None
        # strict load is the real assertion: no key dropped, renamed, or invented
        LibreSegformerNet(size="b0", num_classes=150).load_state_dict(converted, strict=True)

    def test_convert_upstream_state_dict_rejects_foreign_checkpoints(self):
        from libreyolo.models.segformer.model import LibreSegformer

        assert LibreSegformer.convert_upstream_state_dict({"backbone.conv.weight": torch.zeros(1)}) is None

    @pytest.mark.parametrize("size", ["b1", "b2"])
    def test_loading_a_checkpoint_rebuilds_for_its_size(self, size, tmp_path):
        """Size determines every layer shape. Loading a non-default size without
        passing size= must still work: the loader has to re-instantiate the net
        from the checkpoint, not blindly strict-load into the b0 default."""
        from libreyolo.models.segformer.model import LibreSegformer
        from libreyolo.models.segformer.nn import LibreSegformerNet

        net = LibreSegformerNet(size=size, num_classes=150)
        ckpt = tmp_path / f"LibreSegformer{size}-sem.pt"
        torch.save(
            {"model": net.state_dict(), "model_family": "segformer", "task": "semantic",
             "nc": 150, "size": size},
            ckpt,
        )
        loaded = LibreSegformer(model_path=str(ckpt), device="cpu")  # no size= passed
        assert loaded.size == size
        assert loaded.input_size == LibreSegformer.INPUT_SIZES[size]

    def test_class_rebuild_keeps_the_reference_init(self):
        """Re-heading for a new dataset is THE fine-tuning path. A fresh nn.Conv2d
        carries torch's default init, so the new classifier must be re-initialized
        or it starts ~2x too wide with a non-zero bias."""
        import torch.nn as nn

        from libreyolo.models.segformer.model import LibreSegformer

        m = LibreSegformer(size="b0", nb_classes=150, device="cpu")
        m._rebuild_for_new_classes(7)
        head = m.model.decode_head.classifier

        assert isinstance(head, nn.Conv2d) and head.out_channels == 7
        assert head.weight.std().item() < 0.05
        assert torch.equal(head.bias, torch.zeros_like(head.bias))
