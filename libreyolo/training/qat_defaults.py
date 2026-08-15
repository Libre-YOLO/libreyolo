"""Training safeguards for quantization-aware fine-tuning."""

from typing import Any, Tuple


def apply_qat_training_guards(config: Any) -> Tuple[str, ...]:
    """Disable training features that can interfere with fake-quant state."""
    changed = []
    for option in ("ema", "sync_bn"):
        if getattr(config, option, False):
            setattr(config, option, False)
            changed.append(option)
    return tuple(changed)
