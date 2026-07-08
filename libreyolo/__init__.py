"""Libre YOLO — open source YOLO library with MIT license."""

from importlib.metadata import version, PackageNotFoundError
from pathlib import Path as _Path

# Core API — always available
from .models import (
    LibreYOLO,
    LibreYOLOX,
    LibreYOLO9,
    LibreYOLO9E2E,
    LibreYOLO9P2,
    LibreYOLONAS,
    LibreDFINE,
    LibreDEIM,
    LibreDEIMv2,
    LibreEC,
    LibrePICODET,
    LibreRTDETR,
    LibreRTDETRv2,
    LibreRTDETRv4,
    LibreRTMDet,
    LibreYOLO3,
    LibreYOLO4,
    LibreYOLO2,
    LibreYOLO7,
    LibreL2CS,
    LibreFOMO,
    LibreDepthAnythingV2,
    LibreNAFNet,
    LibreRealESRGAN,
    LibreEoMT,
    LibrePIDNet,
    LibreMobileNetV4,
    LibreConvNeXt,
    LibreEfficientNetV2,
    LibreResNet,
    LibreCLIP,
    LibreSigLIP2,
)
from .utils.results import (
    Results,
    Boxes,
    Masks,
    Keypoints,
    Points,
    Probs,
    OBB,
    Gaze,
    SemanticMask,
    DepthMap,
    RestoredImage,
)

SAMPLE_IMAGE = str(_Path(__file__).parent / "assets" / "parkour.jpg")

try:
    __version__ = version("libreyolo")
except PackageNotFoundError:
    __version__ = "0.0.0.dev0"


# Old class names that were renamed for nomenclature consistency. Resolved
# via __getattr__ with a DeprecationWarning so existing imports keep working.
_DEPRECATED_ALIASES = {
    "LibreYOLORTDETR": "LibreRTDETR",
    "LibreYOLORFDETR": "LibreRFDETR",
}


# Lazy imports for optional/heavy modules
def __getattr__(name):
    if name in _DEPRECATED_ALIASES:
        new_name = _DEPRECATED_ALIASES[name]
        import sys
        import warnings

        warnings.warn(
            f"{name} has been renamed to {new_name}. Update your imports — "
            "the old name will be removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )
        # ``getattr`` on the module object resolves both eager imports
        # (``LibreRTDETR`` in globals) and the lazy ``__getattr__`` path
        # (``LibreRFDETR``); recursing into ``__getattr__`` directly would
        # skip the eager case.
        return getattr(sys.modules[__name__], new_name)

    _lazy = {
        "LibreRFDETR": (".models.rfdetr.model", "LibreRFDETR"),
        "LibreDINOv2": (".models.dinov2.model", "LibreDINOv2"),
        "LibreEnsemble": (".ensemble", "LibreEnsemble"),
        "ExternalDetector": (".ensemble", "ExternalDetector"),
        "OnnxBackend": (".backends.onnx", "OnnxBackend"),
        "OpenVINOBackend": (".backends.openvino", "OpenVINOBackend"),
        "TensorRTBackend": (".backends.tensorrt", "TensorRTBackend"),
        "NcnnBackend": (".backends.ncnn", "NcnnBackend"),
        "CoreMLBackend": (".backends.coreml", "CoreMLBackend"),
        "BaseExporter": (".export", "BaseExporter"),
        "DetectionValidator": (".validation", "DetectionValidator"),
        "SegmentationValidator": (".validation", "SegmentationValidator"),
        "PoseValidator": (".validation", "PoseValidator"),
        "SemanticValidator": (".validation", "SemanticValidator"),
        "DepthValidator": (".validation", "DepthValidator"),
        "ValidationConfig": (".validation", "ValidationConfig"),
        "ByteTracker": (".tracking", "ByteTracker"),
        "TrackConfig": (".tracking", "TrackConfig"),
        "OCSortTracker": (".tracking", "OCSortTracker"),
        "OCSortConfig": (".tracking", "OCSortConfig"),
        "LibreVLM": (".models.vlm", "LibreVLM"),
        "LibreLFM2VL": (".models.vlm", "LibreLFM2VL"),
        "LibreQwen3VL": (".models.vlm", "LibreQwen3VL"),
        "LibreSmolVLM2": (".models.vlm", "LibreSmolVLM2"),
        "LibreInternVL3": (".models.vlm", "LibreInternVL3"),
        "LibreFlorence2": (".models.vlm", "LibreFlorence2"),
        "LibreKosmos2": (".models.vlm", "LibreKosmos2"),
        "LibreLocateAnything": (".models.vlm", "LibreLocateAnything"),
        "LibreSAM": (".models.sam", "LibreSAM"),
        "LibreSAM1": (".models.sam", "LibreSAM1"),
        "LibreSAM2": (".models.sam", "LibreSAM2"),
        "LibreMobileSAM": (".models.mobilesam", "LibreMobileSAM"),
        "LibreOpenVocab": (".models.openvocab", "LibreOpenVocab"),
        "LibreGroundingDINO": (".models.openvocab", "LibreGroundingDINO"),
        "LibreOWLv2": (".models.openvocab", "LibreOWLv2"),
        "DATASETS_DIR": (".data", "DATASETS_DIR"),
        "load_data_config": (".data", "load_data_config"),
        "check_dataset": (".data", "check_dataset"),
        "Distiller": (".distillation", "Distiller"),
        "get_distill_config": (".distillation", "get_distill_config"),
    }
    if name in ("LibreRFDETR", "LibreDINOv2"):
        # RF-DETR and DINOv2 share the same transformers dependency check.
        from .models import _ensure_rfdetr

        _ensure_rfdetr()
    if name in _lazy:
        import importlib

        module_path, attr = _lazy[name]
        mod = importlib.import_module(module_path, package=__name__)
        return getattr(mod, attr)
    raise AttributeError(f"module 'libreyolo' has no attribute '{name}'")


__all__ = [
    # Main API
    "LibreYOLO",
    "LibreYOLO9",
    "LibreYOLO9E2E",
    "LibreYOLO9P2",
    "LibreYOLONAS",
    "LibreYOLOX",
    "LibreRTDETR",
    "LibreRTDETRv2",
    "LibreRTDETRv4",
    "LibreRFDETR",
    "LibreDFINE",
    "LibreDEIM",
    "LibreDEIMv2",
    "LibreEC",
    "LibrePICODET",
    "LibreRTMDet",
    "LibreYOLO3",
    "LibreYOLO4",
    "LibreYOLO2",
    "LibreYOLO7",
    "LibreL2CS",
    "LibreFOMO",
    "LibreDepthAnythingV2",
    "LibreNAFNet",
    "LibreRealESRGAN",
    "LibreEoMT",
    "LibrePIDNet",
    "LibreMobileNetV4",
    "LibreConvNeXt",
    "LibreEfficientNetV2",
    "LibreResNet",
    "LibreCLIP",
    "LibreSigLIP2",
    "LibreDINOv2",
    # VLM-as-detector tier (optional, requires libreyolo[vlm])
    "LibreVLM",
    "LibreLFM2VL",
    "LibreQwen3VL",
    "LibreSmolVLM2",
    "LibreInternVL3",
    "LibreFlorence2",
    "LibreKosmos2",
    "LibreLocateAnything",
    # Promptable-segmentation tier (optional, requires libreyolo[sam])
    "LibreSAM",
    "LibreSAM1",
    "LibreSAM2",
    "LibreMobileSAM",
    # Open-vocabulary detector tier (optional, requires libreyolo[openvocab])
    "LibreOpenVocab",
    "LibreGroundingDINO",
    "LibreOWLv2",
    # Results
    "Results",
    "Boxes",
    "Masks",
    "Keypoints",
    "Points",
    "Probs",
    "OBB",
    "Gaze",
    "SemanticMask",
    "DepthMap",
    "RestoredImage",
    # Assets
    "SAMPLE_IMAGE",
    # Tracking
    "ByteTracker",
    "TrackConfig",
    "OCSortTracker",
    "OCSortConfig",
    # Ensembling
    "LibreEnsemble",
    "ExternalDetector",
    # Lazy-loaded
    "OnnxBackend",
    "OpenVINOBackend",
    "TensorRTBackend",
    "NcnnBackend",
    "CoreMLBackend",
    "BaseExporter",
    "DetectionValidator",
    "SegmentationValidator",
    "PoseValidator",
    "SemanticValidator",
    "DepthValidator",
    "ValidationConfig",
    "DATASETS_DIR",
    "load_data_config",
    "check_dataset",
    # Distillation
    "Distiller",
    "get_distill_config",
]
