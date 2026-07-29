"""MHR (Momentum Human Rig) body model wrapper.

MHR is Meta's parametric human body model, released under Apache 2.0 for both
code and assets. It is the body representation the SAM 3D Body regressor
predicts into, and it is the reason a body-mesh task can exist here at all:
the SMPL family, which the rest of the field standardized on, ships under a
non-commercial license whose model files may not be redistributed, so LibreYOLO
could neither host them nor depend on the ``smplx`` package (whose *code* also
carries that non-commercial license).

Only the TorchScript form of MHR is used. It is a single self-contained file
that needs nothing beyond PyTorch, which avoids the ``pymomentum`` native
dependency that the full MHR package requires and that has no reliable Windows
wheel.

Parameterization, verified against the released model rather than taken from
documentation:

* ``model_params`` is 204 wide: 3 translation, 3 global rotation, 130 body
  pose, then 68 per-bone scales. Rotations are Euler angles in radians, not
  axis-angle, and translation enters the rig in decimeters (hence the factor
  of 10 below).
* ``betas`` is 45 identity blendshape coefficients.
* ``expression`` is 72 facial expression coefficients.
* Outputs are 18439 vertices and a 127-joint skeleton state of
  ``(x, y, z, qx, qy, qz, qw, scale)`` per joint, both in centimeters.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional, Tuple

import torch


logger = logging.getLogger(__name__)


# Public, ungated Apache-2.0 release asset. Not mirrored on the LibreYOLO org:
# the upstream release is authoritative and freely reachable, so there is no
# reason to hold a second copy of a 700 MB artifact.
MHR_ASSETS_URL = (
    "https://github.com/facebookresearch/MHR/releases/latest/download/assets.zip"
)
MHR_ARCHIVE_MEMBER = "assets/mhr_model.pt"
MHR_LICENSE_URL = "https://github.com/facebookresearch/MHR/blob/main/LICENSE"


class MHRBodyModel(torch.nn.Module):
    """Decode MHR parameters into a posed mesh and skeleton."""

    NUM_BETAS = 45
    NUM_BODY_POSE = 130
    NUM_SCALES = 68
    NUM_EXPRESSION = 72
    NUM_JOINTS = 127
    NUM_VERTICES = 18439
    MODEL_PARAM_DIM = 204

    # The rig optimizes global translation in decimeters; parameters handed in
    # here are meters, so they are scaled on the way in.
    _TRANSLATION_SCALE = 10.0
    # Vertices and joints come back in centimeters.
    _CM_TO_M = 100.0

    def __init__(self, module: torch.jit.ScriptModule):
        super().__init__()
        self.mhr = module
        for param in self.mhr.parameters():
            param.requires_grad = False

    @classmethod
    def from_file(
        cls, path: str | Path, device: str | torch.device = "cpu"
    ) -> "MHRBodyModel":
        """Load the TorchScript MHR model from a local file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"MHR body model not found at {path}. Fetch it with "
                "libreyolo.models.sam3dbody.mhr_body.ensure_mhr_model()."
            )
        module = torch.jit.load(str(path), map_location=str(device)).eval()
        return cls(module).to(device).eval()

    @property
    def faces(self) -> Optional[torch.Tensor]:
        """Mesh topology, when the loaded module exposes it."""
        return getattr(self.mhr, "faces", None)

    def forward(
        self,
        global_orient: torch.Tensor,
        body_pose: torch.Tensor,
        betas: torch.Tensor,
        scales: torch.Tensor,
        transl: Optional[torch.Tensor] = None,
        expression: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pose the body.

        Args:
            global_orient: ``(B, 3)`` root rotation as Euler angles in radians.
            body_pose: ``(B, 130)`` body joint parameters in radians.
            betas: ``(B, 45)`` identity blendshape coefficients.
            scales: ``(B, 68)`` per-bone skeleton scales.
            transl: ``(B, 3)`` root translation in meters. Defaults to zeros,
                which leaves the body at the rig origin so a camera
                translation can be applied afterwards.
            expression: ``(B, 72)`` facial expression coefficients.

        Returns:
            ``(vertices, joints)`` with shapes ``(B, 18439, 3)`` and
            ``(B, 127, 3)``, both in meters in the model's own frame.
        """
        batch = global_orient.shape[0]
        device, dtype = global_orient.device, global_orient.dtype

        if transl is None:
            transl = torch.zeros(batch, 3, device=device, dtype=dtype)
        if expression is None:
            expression = torch.zeros(
                batch, self.NUM_EXPRESSION, device=device, dtype=dtype
            )

        self._check_width("body_pose", body_pose, self.NUM_BODY_POSE)
        self._check_width("betas", betas, self.NUM_BETAS)
        self._check_width("scales", scales, self.NUM_SCALES)

        model_params = torch.cat(
            [transl * self._TRANSLATION_SCALE, global_orient, body_pose, scales],
            dim=1,
        )
        if model_params.shape[1] != self.MODEL_PARAM_DIM:
            raise ValueError(
                f"assembled MHR model parameters are {model_params.shape[1]} wide, "
                f"expected {self.MODEL_PARAM_DIM}"
            )

        vertices, skeleton_state = self.mhr(betas, model_params, expression)
        # Skeleton state packs position, unit quaternion and scale per joint.
        joints = skeleton_state[..., :3]
        return vertices / self._CM_TO_M, joints / self._CM_TO_M

    @staticmethod
    def _check_width(name: str, tensor: torch.Tensor, expected: int) -> None:
        if tensor.shape[-1] != expected:
            raise ValueError(
                f"{name} must be {expected} wide, got {tensor.shape[-1]}"
            )


def default_mhr_path() -> Path:
    """Location LibreYOLO caches the MHR body model at."""
    root = os.environ.get("LIBREYOLO_MHR_PATH")
    if root:
        return Path(root)
    return Path.home() / ".cache" / "libreyolo" / "mhr" / "mhr_model.pt"


def ensure_mhr_model(path: str | Path | None = None) -> Path:
    """Return a local MHR model path, downloading the release asset if absent.

    The asset is Apache 2.0 and served from a public GitHub release, so no
    token, registration or license acceptance is involved.
    """
    target = Path(path) if path is not None else default_mhr_path()
    if target.exists():
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Downloading the MHR body model (Apache 2.0, ~700 MB) from %s",
        MHR_ASSETS_URL,
    )

    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "assets.zip"
        request = urllib.request.Request(
            MHR_ASSETS_URL, headers={"User-Agent": "libreyolo"}
        )
        with urllib.request.urlopen(request) as response, open(archive, "wb") as fh:
            shutil.copyfileobj(response, fh)

        with zipfile.ZipFile(archive) as zf:
            member = next(
                (n for n in zf.namelist() if n.endswith("mhr_model.pt")), None
            )
            if member is None:
                raise RuntimeError(
                    f"{MHR_ARCHIVE_MEMBER} is missing from the MHR release archive. "
                    "The upstream release layout may have changed."
                )
            staged = Path(tmp) / "mhr_model.pt"
            with zf.open(member) as src, open(staged, "wb") as dst:
                shutil.copyfileobj(src, dst)
            # Move into place only once complete, so an interrupted download
            # never leaves a truncated file that later loads as corrupt.
            shutil.move(str(staged), str(target))

    logger.info("MHR body model ready at %s (license: %s)", target, MHR_LICENSE_URL)
    return target


def load_mhr_body_model(
    path: str | Path | None = None,
    device: str | torch.device = "cpu",
    download: bool = True,
) -> MHRBodyModel:
    """Load the MHR body model, fetching it on first use when allowed."""
    target = Path(path) if path is not None else default_mhr_path()
    if not target.exists():
        if not download:
            raise FileNotFoundError(
                f"MHR body model not found at {target} and download=False."
            )
        target = ensure_mhr_model(target)
    return MHRBodyModel.from_file(target, device=device)
