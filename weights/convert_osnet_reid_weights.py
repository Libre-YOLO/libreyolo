"""Convert Torchreid OSNet-AIN checkpoints into LibreYOLO ReID weights.

The Deep OC-SORT tracker's appearance embedder
(``libreyolo/tracking/reid.py``) re-implements OSNet-AIN with the *same*
module structure as Torchreid (MIT), so the upstream multi-source
(MS+D+C) checkpoints load with 0 missing / 0 unexpected keys and forward
parity is bit-exact (``tests/unit/test_reid.py``). This script downloads
the checkpoints from Torchreid's model zoo (Google Drive), strips the
DataParallel prefix and the dataset-specific identity-classifier head,
and saves plain state dicts named ``<variant>.pt``.

Weight provenance: code and released checkpoints are MIT (Kaiyang Zhou,
https://github.com/KaiyangZhou/deep-person-reid). The checkpoints were
trained on person re-identification datasets distributed for research
(MSMT17 / DukeMTMC / CUHK03); see the model card on the Hugging Face
mirror for details.

Usage::

    python weights/convert_osnet_reid_weights.py                     # all 4 -> weights/
    python weights/convert_osnet_reid_weights.py --variants osnet_ain_x0_25
"""

from __future__ import annotations

import argparse
from pathlib import Path

import requests
import torch

from _conversion_utils import add_repo_root_to_path

add_repo_root_to_path()

from libreyolo.tracking.reid import (  # noqa: E402
    _OSNET_CHANNELS,
    OSNet,
    _load_osnet_state_dict,
)

# Torchreid model-zoo multi-source (MS+D+C) checkpoints.
DRIVE_IDS = {
    "osnet_ain_x1_0": "1-CaioD9NaqbHK_kzSMW8VE4_3KcsRjEo",
    "osnet_ain_x0_75": "1apy0hpsMypqstfencdH-jKIUEFOW4xoM",
    "osnet_ain_x0_5": "1KusKvEYyKGDTUBVRxRiz55G31wkihB6l",
    "osnet_ain_x0_25": "1SxQt2AvmEcgWNhaRb2xC4rP6ZwVDP0Wt",
}


def convert(variant: str, out_dir: Path) -> Path:
    url = f"https://drive.google.com/uc?id={DRIVE_IDS[variant]}&export=download"
    raw_path = out_dir / f"{variant}_msdc.pth.tar"
    if not raw_path.exists():
        print(f"downloading {variant} from {url}")
        response = requests.get(url)
        response.raise_for_status()
        raw_path.write_bytes(response.content)

    sd = _load_osnet_state_dict(raw_path)

    # Strict load + forward sanity on the ported architecture.
    model = OSNet(_OSNET_CHANNELS[variant]).eval()
    model.load_state_dict(sd)
    with torch.no_grad():
        feats = model(torch.zeros(1, 3, 256, 128))
    assert feats.shape == (1, 512) and torch.isfinite(feats).all()

    out_path = out_dir / f"{variant}.pt"
    torch.save(sd, out_path)
    print(f"wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variants", nargs="+", default=list(DRIVE_IDS), choices=list(DRIVE_IDS)
    )
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    for variant in args.variants:
        convert(variant, args.out_dir)


if __name__ == "__main__":
    main()
