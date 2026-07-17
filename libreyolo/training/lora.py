"""Parameter-efficient fine-tuning (LoRA) helpers for LibreYOLO.

LibreYOLO's own integration of LoRA on top of the optional ``peft`` dependency.
It targets transformer (DINOv2 ViT) backbones such as RF-DETR, where the
attention projections are ``nn.Linear`` layers that LoRA adapts cheaply. The
backbone base weights are frozen and only the small low-rank adapters remain
trainable in the backbone, while the projector, decoder, and detection head keep
training normally. This lets users with limited GPU memory fine-tune on a custom
dataset.

The adapter recipe is a faithful match of the RF-DETR reference (Apache-2.0):
DoRA (weight-decomposed LoRA) with rank 16 and alpha 16 on the DINOv2 attention
projections. The public surface is a single boolean ``lora=True`` training
argument; the hyperparameters below are fixed, not a user-facing API.
"""

from __future__ import annotations

import logging

import torch.nn as nn

logger = logging.getLogger(__name__)

# Fixed adapter hyperparameters, matching the RF-DETR reference. Not exposed:
# the public API is ``lora=True``.
LORA_RANK = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0.0
USE_DORA = True  # weight-decomposed LoRA (DoRA), as in upstream RF-DETR

# Target-module suffixes for the DINOv2 ViT, matching the RF-DETR reference list
# verbatim. On LibreYOLO's transformers-based DINOv2 only the attention
# ``query``/``key``/``value`` ``nn.Linear`` layers actually match; the remaining
# entries are inert for this backbone but kept for parity:
#   - ``q_proj``/``k_proj``/``v_proj``/``qkv`` name other DINOv2 variants' fused
#     or renamed projections that this implementation does not use.
#   - ``cls_token``/``register_tokens`` are ``nn.Parameter`` attributes, not
#     ``nn.Linear`` modules, so peft's module-name matching does not adapt them
#     here (this matches upstream behavior on the same HF backbone).
DINOV2_TARGET_MODULES = (
    "q_proj",
    "v_proj",
    "k_proj",
    "qkv",
    "query",
    "key",
    "value",
    "cls_token",
    "register_tokens",
)

_PEFT_INSTALL_HINT = (
    'LoRA fine-tuning requires the optional "peft" package. '
    'Install it with: pip install "libreyolo[lora]"'
)


def _require_peft():
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise ImportError(_PEFT_INSTALL_HINT) from exc
    return LoraConfig, get_peft_model


def is_peft_available() -> bool:
    """Return True when the optional ``peft`` dependency is importable."""
    try:
        import peft  # noqa: F401
    except ImportError:
        return False
    return True


def count_trainable_parameters(module: nn.Module) -> tuple[int, int]:
    """Return ``(trainable, total)`` parameter counts for *module*."""
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    total = sum(p.numel() for p in module.parameters())
    return trainable, total


def state_dict_has_lora(state_dict: dict) -> bool:
    """Return True when *state_dict* carries LoRA adapter tensors."""
    return any(is_lora_parameter_name(k) for k in state_dict)


def is_lora_parameter_name(name: str) -> bool:
    """Return True for PEFT LoRA adapter parameter names."""
    return "lora_A" in name or "lora_B" in name or "lora_magnitude" in name


def module_has_lora(module: nn.Module) -> bool:
    """Return True when *module* already carries PEFT/LoRA adapters."""
    if hasattr(module, "peft_config"):
        return True
    return any("lora_" in name for name, _ in module.named_parameters())


def apply_lora_to_rfdetr(core_model: nn.Module) -> nn.Module:
    """Inject LoRA adapters into an RF-DETR DINOv2 encoder, in place.

    Wraps ``core_model.backbone[0].encoder`` (the DINOv2 ViT) with a PEFT model
    so the base weights are frozen and only the low-rank adapters are trainable.
    The wrapped encoder exposes ``merge_and_unload`` which the backbone's
    ``export`` path already uses to bake adapters back into dense weights.

    Args:
        core_model: the LWDETR core module (``LibreRFDETRModel.model``) that
            owns the ``backbone`` Joiner.

    Returns:
        The PEFT-wrapped encoder module that replaced the original encoder.

    Raises:
        ImportError: if ``peft`` is not installed.
        ValueError: if the model does not expose the expected backbone layout.
    """
    backbone = getattr(core_model, "backbone", None)
    if backbone is None:
        raise ValueError("RF-DETR model has no .backbone; cannot apply LoRA.")
    try:
        encoder_owner = backbone[0]
    except (TypeError, IndexError) as exc:
        raise ValueError("RF-DETR model.backbone[0] is not indexable.") from exc
    if not hasattr(encoder_owner, "encoder"):
        raise ValueError("RF-DETR model.backbone[0] has no .encoder to adapt.")

    if module_has_lora(encoder_owner.encoder):
        return encoder_owner.encoder

    LoraConfig, get_peft_model = _require_peft()
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        use_dora=USE_DORA,
        target_modules=list(DINOV2_TARGET_MODULES),
        bias="none",
    )
    wrapped = get_peft_model(encoder_owner.encoder, lora_config)
    encoder_owner.encoder = wrapped

    n_adapted = sum(
        1 for name, _ in wrapped.named_modules() if name.endswith(".lora_A.default")
    )
    if n_adapted == 0:
        raise ValueError(
            "LoRA injection matched zero modules in the RF-DETR backbone. "
            f"Expected target suffixes {DINOV2_TARGET_MODULES} in the DINOv2 encoder; "
            "the backbone module naming may have changed."
        )

    trainable, total = count_trainable_parameters(core_model)
    logger.info(
        "Applied LoRA to RF-DETR backbone: %d adapted modules, "
        "%d/%d trainable params (%.2f%%).",
        n_adapted,
        trainable,
        total,
        100.0 * trainable / max(1, total),
    )
    return wrapped


# ---------------------------------------------------------------------------
# D-FINE / DEIM (DETR decoder) recipe
# ---------------------------------------------------------------------------
# These families pair a CNN (HGNetv2) backbone with a transformer hybrid
# encoder + deformable decoder. The CNN backbone has no nn.Linear layers to
# adapt, so the recipe differs from RF-DETR's ViT-backbone recipe:
#   - the backbone is frozen entirely (no gradients flow through it at all,
#     which also skips its backward pass),
#   - the transformer blocks (AIFI encoder layers + decoder layers) freeze
#     their base weights and train LoRA adapters on their nn.Linear layers,
#   - everything else (encoder conv fusion, input projections, prediction
#     heads, query embeddings) keeps training normally, since custom class
#     counts require freshly trained heads.
#
# Plain LoRA, not DoRA: several decoder Linears (Gate.gate, MSDeformableAttention
# sampling_offsets / attention_weights) are zero-init by design, and DoRA's
# magnitude normalization divides by the weight norm, which is 0 for a
# freshly built model. Rank/alpha match the RF-DETR recipe.
DETR_LORA_RANK = 16
DETR_LORA_ALPHA = 16
DETR_LORA_DROPOUT = 0.0

# Module classes that delimit the frozen+adapted transformer zone. Both
# D-FINE and DEIM define their own copies of these classes with identical
# names (as does RT-DETR, for future adoption).
DETR_BLOCK_CLASSES = ("TransformerEncoderLayer", "TransformerDecoderLayer")

# Leaf names of nn.Linear layers adapted inside those blocks. ``value_proj``
# and ``output_proj`` are inert for D-FINE/DEIM (their deformable attention
# has neither) but kept for parity with other deformable-DETR decoders.
# nn.MultiheadAttention is intentionally NOT adapted: its forward reads
# ``out_proj.weight`` directly, so a swapped-in LoRA layer would be silently
# bypassed. Self-attention stays frozen instead.
DETR_TARGET_LINEAR_NAMES = (
    "linear1",
    "linear2",
    "gate",
    "sampling_offsets",
    "attention_weights",
    "value_proj",
    "output_proj",
)


def apply_lora_to_detr(core_model: nn.Module) -> nn.Module:
    """Inject LoRA adapters into a D-FINE/DEIM style DETR core model, in place.

    Uses ``peft.inject_adapter_in_model`` (no PeftModel wrapper) so the module
    tree, attribute surface, and checkpoint key layout stay put; only the
    targeted ``nn.Linear`` layers become ``lora.Linear`` (their dense weight
    moves under ``.base_layer.``). After injection the trainability policy is
    applied explicitly:

    - ``lora_*`` adapter params: trainable
    - ``backbone.*`` and all params inside transformer blocks: frozen
    - everything else: left exactly as it was (heads/neck stay trainable,
      intentionally fixed params like D-FINE's ``decoder.up`` stay fixed)

    Args:
        core_model: a model exposing ``backbone``/``encoder``/``decoder``
            (``LibreDFINEModel`` / ``LibreDEIMModel``).

    Returns:
        *core_model*, modified in place.

    Raises:
        ImportError: if ``peft`` is not installed.
        ValueError: if the model does not look like a DETR core or no target
            module matched.
    """
    for attr in ("backbone", "encoder", "decoder"):
        if not hasattr(core_model, attr):
            raise ValueError(
                f"Model has no .{attr}; cannot apply the DETR LoRA recipe."
            )

    if module_has_lora(core_model):
        return core_model

    try:
        from peft import LoraConfig, inject_adapter_in_model
    except ImportError as exc:
        raise ImportError(_PEFT_INSTALL_HINT) from exc

    block_roots = [
        name
        for name, module in core_model.named_modules()
        if type(module).__name__ in DETR_BLOCK_CLASSES
    ]
    if not block_roots:
        raise ValueError(
            "No transformer encoder/decoder blocks found; expected "
            f"{DETR_BLOCK_CLASSES} classes in the model."
        )

    target_modules = []
    for root in block_roots:
        block = core_model.get_submodule(root)
        for sub_name, sub in block.named_modules():
            if not isinstance(sub, nn.Linear):
                continue
            if sub_name.rsplit(".", 1)[-1] not in DETR_TARGET_LINEAR_NAMES:
                continue
            params = list(sub.parameters())
            # Respect layers frozen by design (e.g. ``cross_attn_method=
            # "discrete"`` freezes sampling_offsets): no adapter there.
            if params and all(not p.requires_grad for p in params):
                continue
            target_modules.append(f"{root}.{sub_name}")
    if not target_modules:
        raise ValueError(
            "LoRA injection matched zero modules in the DETR transformer "
            f"blocks. Expected Linear leaf names {DETR_TARGET_LINEAR_NAMES}; "
            "the module naming may have changed."
        )

    # Snapshot trainability before injection so the policy below can restore
    # everything outside the frozen zones no matter what peft toggled.
    pre_requires_grad = {
        name: p.requires_grad for name, p in core_model.named_parameters()
    }

    lora_config = LoraConfig(
        r=DETR_LORA_RANK,
        lora_alpha=DETR_LORA_ALPHA,
        lora_dropout=DETR_LORA_DROPOUT,
        use_dora=False,
        target_modules=target_modules,
        bias="none",
    )
    inject_adapter_in_model(lora_config, core_model, adapter_name="default")

    n_adapted = sum(
        1
        for name, _ in core_model.named_modules()
        if name.endswith(".lora_A.default")
    )
    if n_adapted == 0:
        raise ValueError(
            "peft matched the target list but created no adapters; "
            f"targets were {target_modules[:5]}..."
        )

    frozen_prefixes = ("backbone.",) + tuple(f"{root}." for root in block_roots)
    for name, p in core_model.named_parameters():
        if "lora_" in name:
            p.requires_grad = True
        elif name.startswith(frozen_prefixes):
            p.requires_grad = False
        elif name in pre_requires_grad:
            p.requires_grad = pre_requires_grad[name]

    trainable, total = count_trainable_parameters(core_model)
    logger.info(
        "Applied LoRA to DETR transformer blocks: %d adapted modules, "
        "%d/%d trainable params (%.2f%%).",
        n_adapted,
        trainable,
        total,
        100.0 * trainable / max(1, total),
    )
    return core_model


def merge_lora_adapters(module: nn.Module) -> int:
    """Merge injected LoRA layers back into dense weights, in place.

    For models adapted with :func:`apply_lora_to_detr` (peft injection without
    a PeftModel wrapper), folds every adapter into its base ``nn.Linear`` and
    swaps the original layer back in, so the module carries no peft dependency
    afterwards.

    Returns:
        Number of merged adapter layers.
    """
    try:
        from peft.tuners.tuners_utils import BaseTunerLayer
    except ImportError as exc:
        raise ImportError(_PEFT_INSTALL_HINT) from exc

    merged = 0
    for name, sub in list(module.named_modules()):
        if not isinstance(sub, BaseTunerLayer):
            continue
        sub.merge(safe_merge=True)
        parent = (
            module.get_submodule(name.rsplit(".", 1)[0]) if "." in name else module
        )
        setattr(parent, name.rsplit(".", 1)[-1], sub.get_base_layer())
        merged += 1
    return merged


__all__ = [
    "apply_lora_to_rfdetr",
    "apply_lora_to_detr",
    "merge_lora_adapters",
    "is_peft_available",
    "is_lora_parameter_name",
    "state_dict_has_lora",
    "module_has_lora",
    "count_trainable_parameters",
    "DINOV2_TARGET_MODULES",
    "DETR_BLOCK_CLASSES",
    "DETR_TARGET_LINEAR_NAMES",
    "DETR_LORA_RANK",
    "DETR_LORA_ALPHA",
    "LORA_RANK",
    "LORA_ALPHA",
    "USE_DORA",
]
