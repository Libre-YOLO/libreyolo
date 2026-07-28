"""SAM 3D Body: human body mesh recovery on the MHR body model."""

from .camera import crop_cam_to_full_image, default_focal_length, perspective_project
from .mhr_body import (
    MHRBodyModel,
    default_mhr_path,
    ensure_mhr_model,
    load_mhr_body_model,
)
from .person import (
    CallablePersonDetector,
    LibreYOLOPersonDetector,
    PersonBox,
    normalize_person_boxes,
    resolve_person_detector,
)


def __getattr__(name):
    # Imported lazily: building the family pulls in the optional upstream
    # package, so merely importing libreyolo must not require it.
    if name == "LibreSAM3DBody":
        from .model import LibreSAM3DBody

        return LibreSAM3DBody
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "LibreSAM3DBody",
    "MHRBodyModel",
    "PersonBox",
    "CallablePersonDetector",
    "LibreYOLOPersonDetector",
    "crop_cam_to_full_image",
    "default_focal_length",
    "default_mhr_path",
    "ensure_mhr_model",
    "load_mhr_body_model",
    "normalize_person_boxes",
    "perspective_project",
    "resolve_person_detector",
]
