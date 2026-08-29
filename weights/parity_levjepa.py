"""Black-box numerical parity check for the native LeVJEPA inference graph.

The official remote implementation is executed as an oracle but is never read,
copied, imported into LibreYOLO, or required by users.
"""

from __future__ import annotations

import torch

from libreyolo.models.levjepa.nn import LeVJEPAConfig, LeVJEPAModel


SOURCE_REPO = "galilai-group/LeVJEPA-VideoMix-Large"
SOURCE_REVISION = "e831a0347737fcaa660b39c57d41c109de399845"


def main() -> None:
    from transformers import AutoModel

    if not torch.cuda.is_available():
        raise SystemExit("LeVJEPA parity requires CUDA for the 303M-parameter pair")
    reference = (
        AutoModel.from_pretrained(
            SOURCE_REPO,
            revision=SOURCE_REVISION,
            trust_remote_code=True,
        )
        .eval()
        .cuda()
    )
    native = LeVJEPAModel(LeVJEPAConfig.for_size("l")).eval().cuda()
    native.load_state_dict(reference.state_dict(), strict=True)

    torch.manual_seed(0)
    public_clip = torch.randn(1, 16, 3, 224, 224, device="cuda")
    with torch.no_grad():
        expected = reference(
            pixel_values=public_clip.permute(0, 2, 1, 3, 4)
        ).last_hidden_state
        actual = native(public_clip)
    maximum = (expected - actual).abs().max().item()
    if maximum != 0.0:
        raise AssertionError(f"LeVJEPA max_abs_diff={maximum}")
    print(f"LeVJEPA parity OK: shape={tuple(actual.shape)}, max_abs_diff=0.0")


if __name__ == "__main__":
    main()
