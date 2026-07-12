"""Base class for the ``LibreSAM`` tier: promptable segmentation models.

A promptable segmenter is fundamentally different from the detectors in the
``LibreYOLO`` factory: instead of one promptless forward returning every
object, it runs a heavy image encoder once and then answers cheap *spatial
prompts* (points, boxes) by returning a mask. So this tier exposes an
interactive surface — ``set_image()`` to encode once, then ``predict(points=,
bboxes=)`` to prompt many times — plus a "segment everything" mode when no
prompt is given.

The model is loaded through the permissive ``transformers`` model API (so
LibreYOLO ships no model source and stays MIT, exactly as the LibreVLM tier
does). Like LibreVLM, this base does NOT define ``can_load``, which keeps the
family out of the state-dict ``_registry`` and the ``LibreYOLO`` factory.

See ``docs/adr/0007-libresam-contract.md`` for the contract and the
clean-room API notes.
"""

from __future__ import annotations

import json
import logging
import threading
from functools import wraps
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional

import torch
import torch.nn as nn

from ...utils.image_loader import ImageInput, ImageLoader
from ...utils.results import Boxes, Masks, Results
from ..base.model import BaseModel
from .prompts import (
    build_point_grid,
    normalize_boxes,
    normalize_labels,
    normalize_points,
    validate_max_det,
)

logger = logging.getLogger(__name__)

_INSTALL_HINT = (
    "LibreSAM models require the 'sam' extra. Install with:\n"
    "    pip install 'libreyolo[sam]'"
)
_SNAPSHOT_COMPLETE_MARKER = ".libreyolo_snapshot_complete"


def _is_empty_prompt(prompt) -> bool:
    """True for ``None`` and empty containers (so they route to "everything").

    Accepts lists/tuples/numpy arrays; a non-empty array of coordinates is not
    empty. Anything without a length (e.g. a stray scalar) is treated as a real
    prompt and left for normalization to validate.
    """
    if prompt is None:
        return True
    try:
        return len(prompt) == 0
    except TypeError:
        return False


def _synchronized_image_state(method):
    """Serialize operations that read or mutate an interactive image session."""

    @wraps(method)
    def synchronized(self, *args, **kwargs):
        with self._get_image_state_lock():
            return method(self, *args, **kwargs)

    return synchronized


class LibreSAMModel(BaseModel):
    """Promptable segmentation model with an encode-once / prompt-many surface."""

    # Subclasses override these.
    FAMILY: ClassVar[str] = ""
    FILENAME_PREFIX: ClassVar[str] = ""
    HF_REPOS: ClassVar[Dict[str, str]] = {}
    INPUT_SIZES: ClassVar[Dict[str, int]] = {}
    SUPPORTED_TASKS: ClassVar[tuple] = ("segment",)
    DEFAULT_TASK: ClassVar[str] = "segment"

    # The interactive lifecycle is bespoke; the detection pipeline's
    # multi-scale TTA is meaningless here.
    TTA_ENABLED: ClassVar[bool] = False

    # "Segment everything" defaults (basic automatic mask generation).
    DEFAULT_POINTS_PER_SIDE: ClassVar[int] = 32
    DEFAULT_PRED_IOU_THRESH: ClassVar[float] = 0.88
    GRID_CHUNK: ClassVar[int] = 64
    AMG_NMS_IOU: ClassVar[float] = 0.7

    # Off by default: SAM families load through native transformers classes.
    # The SAM repos publish both safetensors and a duplicate ``pytorch_model.bin``
    # pickle (1.25-2.56 GB for large/huge); transformers prefers safetensors, so
    # skip the .bin to avoid downloading a checkpoint that is never loaded.
    SNAPSHOT_IGNORE_PATTERNS: ClassVar[tuple[str, ...]] = (
        "*.bin",
        "*.bin.index.json",
        "*.onnx",
        "onnx/*",
        "*.tflite",
        "*.msgpack",
        "*.h5",
    )

    # =========================================================================
    # Construction
    # =========================================================================

    def __init__(
        self,
        size: str,
        *,
        device: str = "auto",
        task: str | None = None,
        multimask: bool = False,
        **kwargs,
    ):
        if size not in self.HF_REPOS:
            raise ValueError(
                f"Invalid size {size!r} for {type(self).__name__}. "
                f"Must be one of: {', '.join(self.HF_REPOS)}"
            )
        self._image_state_lock = threading.RLock()
        self._default_multimask = multimask
        # Single foreground class; promptable masks are class-agnostic "object".
        super().__init__(
            model_path=self.HF_REPOS[size],
            size=size,
            nb_classes=1,
            device=device,
            task=task,
            **kwargs,
        )
        self.names = {0: "object"}
        self.model.eval()
        self._clear_image_state()

    # =========================================================================
    # Weight acquisition (autodownload via Hugging Face)
    # =========================================================================

    @staticmethod
    def _snapshot_complete(local_dir: Path) -> bool:
        """True only if the marker, config, and an actual weight file exist.

        config.json + marker alone is not proof: an interrupted download can
        write them without the safetensors/bin payload, so check for a weight
        file too (mirrors the LibreVLM weight-acquisition guard).
        """
        if not (local_dir / _SNAPSHOT_COMPLETE_MARKER).exists():
            return False
        if not (local_dir / "config.json").exists():
            return False
        return any(local_dir.glob("*.safetensors")) or any(local_dir.glob("*.bin"))

    def _ensure_weights(self) -> str:
        """Return a local weights dir for this size, downloading if needed.

        Downloads into ``weights/<FILENAME_PREFIX><size>/`` (copies, no
        symlinks) to match LibreYOLO's ``weights/`` convention and avoid the
        symlinked HF cache, which needs admin/Developer Mode on Windows.
        """
        repo = self.HF_REPOS[self.size]
        local_dir = Path("weights") / f"{self.FILENAME_PREFIX}{self.size}"
        if self._snapshot_complete(local_dir):
            return str(local_dir)
        try:
            from huggingface_hub import snapshot_download
            import transformers  # noqa: F401  (ships with the extra)
        except ImportError as exc:
            raise ImportError(_INSTALL_HINT) from exc
        logger.info(
            "Downloading %s weights from %s -> %s ...", self.FAMILY, repo, local_dir
        )
        snapshot_download(
            repo,
            local_dir=str(local_dir),
            ignore_patterns=list(self.SNAPSHOT_IGNORE_PATTERNS),
        )
        (local_dir / _SNAPSHOT_COMPLETE_MARKER).write_text(
            json.dumps({"repo": repo}) + "\n", encoding="utf-8"
        )
        if not self._snapshot_complete(local_dir):
            (local_dir / _SNAPSHOT_COMPLETE_MARKER).unlink(missing_ok=True)
            raise FileNotFoundError(
                f"Downloaded snapshot for {repo} is missing config or weight "
                f"files in {local_dir}."
            )
        return str(local_dir)

    def _resolve_dtype(self) -> "torch.dtype":
        """Always fp32 — including on CUDA.

        Unlike the (multi-GB) VLM tier, SAM-1 checkpoints are small enough that
        fp32 inference is cheap, and half precision actively hurts SAM: mask
        logits lose boundary fidelity, and — more importantly — bf16's 8-bit
        mantissa rounds prompt-point coordinates by several pixels once they
        exceed 256 (the model frame goes up to 1024), silently shifting where
        the user clicked. Half precision can return later as an explicit opt-in.
        """
        return torch.float32

    def _load_pretrained(self, snapshot_dir: str) -> tuple:
        try:
            from transformers import SamModel, SamProcessor
        except ImportError as exc:
            raise ImportError(_INSTALL_HINT) from exc
        model = SamModel.from_pretrained(snapshot_dir, dtype=self._resolve_dtype())
        processor = SamProcessor.from_pretrained(snapshot_dir)
        return model, processor

    def _init_model(self) -> nn.Module:
        snapshot_dir = self._ensure_weights()
        model, processor = self._load_pretrained(snapshot_dir)
        self.processor = processor
        self._model_dtype = next(model.parameters()).dtype
        return model

    # =========================================================================
    # Abstract hooks: SAM does not use the promptless detection pipeline.
    # =========================================================================

    def _get_available_layers(self) -> Dict[str, nn.Module]:
        return {name: module for name, module in self.model.named_modules() if name}

    @staticmethod
    def _get_preprocess_numpy():
        raise NotImplementedError(
            "LibreSAM families preprocess through the transformers processor."
        )

    def _preprocess(self, *args, **kwargs):
        raise NotImplementedError(
            "LibreSAM uses set_image()/predict(points=, bboxes=) directly, "
            "not the promptless detection pipeline."
        )

    def _forward(self, *args, **kwargs):
        raise NotImplementedError(
            "LibreSAM uses set_image()/predict(points=, bboxes=) directly."
        )

    def _postprocess(self, *args, **kwargs):
        raise NotImplementedError(
            "LibreSAM uses set_image()/predict(points=, bboxes=) directly."
        )

    # =========================================================================
    # Interactive image session (encode once, prompt many)
    # =========================================================================

    def _get_image_state_lock(self) -> threading.RLock:
        lock = getattr(self, "_image_state_lock", None)
        if lock is None:
            lock = threading.RLock()
            self._image_state_lock = lock
        return lock

    @_synchronized_image_state
    def _clear_image_state(self) -> None:
        self._image = None
        self._image_path = None
        self._image_embeddings = None

    @_synchronized_image_state
    def set_image(
        self, source: ImageInput, color_format: str = "auto"
    ) -> "LibreSAMModel":
        """Encode an image once so later ``predict()`` prompts are cheap.

        Runs the heavy image encoder a single time and caches the embeddings.
        Subsequent ``predict(points=...)`` / ``predict(bboxes=...)`` calls reuse
        them. Call ``reset_image()`` to clear. Returns ``self`` to allow
        chaining.
        """
        img = ImageLoader.load(source, color_format=color_format)
        enc = self._encode(img)
        self._image_embeddings = self._image_embeddings_from_encoding(enc)
        self._image = img
        self._image_path = source if isinstance(source, (str, Path)) else None
        return self

    @_synchronized_image_state
    def reset_image(self) -> "LibreSAMModel":
        """Drop the cached image and embeddings from ``set_image()``."""
        self._clear_image_state()
        return self

    # =========================================================================
    # Prediction
    # =========================================================================

    @_synchronized_image_state
    def predict(
        self,
        source: Optional[ImageInput] = None,
        *,
        points=None,
        bboxes=None,
        labels=None,
        masks=None,
        text: str | None = None,
        conf: Optional[float] = None,
        multimask: Optional[bool] = None,
        max_det: int = 300,
        device: Optional[str] = None,
        color_format: str = "auto",
        points_per_side: Optional[int] = None,
    ) -> Results:
        """Promptable segmentation.

        Pass ``points``/``bboxes`` to segment specific objects, or omit all
        prompts to segment everything. With ``source=None`` the image cached by
        ``set_image()`` is reused. Returns a standard ``Results`` carrying
        ``masks`` (and tight ``boxes`` derived from them).

        Args:
            source: Image to segment. ``None`` reuses ``set_image()``.
            points: ``[x, y]`` / ``[[x, y], ...]`` / ``[[[x, y], ...], ...]``.
                Plain ``[x, y]`` pixel coordinates; numpy arrays are accepted.
            bboxes: ``[x1, y1, x2, y2]`` or a list of them (one mask per box).
            labels: Point labels, ``1`` positive / ``0`` negative (default all
                positive), shaped to match ``points``.
            text: Concept prompt. Text prompts require SAM 3; spatial-prompt
                families raise ``NotImplementedError``.
            conf: Drop masks whose *predicted IoU* (mask-quality score, not a
                detection confidence) is below this. ``None`` keeps all in the
                prompted path and applies the family's grid threshold in
                "segment everything"; pass ``0.0`` to disable filtering in
                either mode.
            multimask: Return all of SAM's ambiguity masks per prompt (3 masks,
                whole-vs-part) instead of the single best one. Defaults to the
                model's construction setting.
            max_det: Cap on returned masks.
            device: Move the model to this device for this and later calls
                (invalidates any cached ``set_image`` embeddings).
            points_per_side: Grid density for "segment everything" (default
                ``DEFAULT_POINTS_PER_SIDE``; on CPU the default of 32 runs
                ~1024 decoder passes and is slow — lower it for interactivity).
        """
        max_det = validate_max_det(max_det)
        if text is not None:
            raise NotImplementedError(
                "Text prompts require SAM 3; load LibreSAM('sam3')."
            )
        if masks is not None:
            raise NotImplementedError(
                "Mask prompts are not supported in LibreSAM v1; use points= or bboxes=."
            )
        if device is not None:
            self._set_device(device)
        multimask = self._default_multimask if multimask is None else multimask
        img, image_path, cached_emb = self._resolve_image(source, color_format)
        conf_thres = 0.0 if conf is None else float(conf)

        # An empty container means "this prompt type is absent", so a
        # dynamically-built box-only / point-only call doesn't trip
        # normalization (which would reject [] as nesting depth 0).
        if _is_empty_prompt(points):
            points = None
        if _is_empty_prompt(bboxes):
            bboxes = None

        if points is None and bboxes is None:
            if labels is not None:
                raise ValueError(
                    "labels were given without points. Pass points= for a "
                    "prompted call, or omit labels to segment everything."
                )
            # Only None takes the default; an explicit (invalid) 0 must reach
            # build_point_grid's validation rather than silently expanding.
            grid = (
                self.DEFAULT_POINTS_PER_SIDE
                if points_per_side is None
                else points_per_side
            )
            amg_thresh = self.DEFAULT_PRED_IOU_THRESH if conf is None else conf_thres
            return self._segment_everything(
                img,
                image_path,
                points_per_side=grid,
                pred_iou_thresh=amg_thresh,
                max_det=max_det,
                cached_emb=cached_emb,
            )

        npoints = normalize_points(points, image_size=img.size)
        nlabels = normalize_labels(labels, npoints)
        nboxes = normalize_boxes(bboxes, image_size=img.size)
        if npoints is not None and nboxes is not None and len(npoints) != len(nboxes):
            raise ValueError(
                f"points has {len(npoints)} objects but bboxes has "
                f"{len(nboxes)}; when combining points and boxes they must "
                "pair per object."
            )

        enc = self._encode(img, points=npoints, labels=nlabels, boxes=nboxes)
        model_inputs = self._build_model_inputs(enc, cached_emb)
        with torch.no_grad():
            outputs = self.model(**model_inputs, multimask_output=multimask)

        sel_masks, sel_conf = self._post_process(outputs, enc, multimask)
        return self._assemble_results(
            sel_masks, sel_conf, img, image_path, conf_thres=conf_thres, max_det=max_det
        )

    def __call__(self, source: Optional[ImageInput] = None, **kwargs) -> Results:
        return self.predict(source, **kwargs)

    def _move_embeddings(self, embeddings, device: torch.device):
        """Move cached image embeddings to ``device``."""
        return embeddings.to(device)

    @_synchronized_image_state
    def _set_device(self, device: str) -> "LibreSAMModel":
        """Move the model to ``device``, keeping any ``set_image()`` session.

        Cached embeddings are moved to the new device rather than dropped, so an
        interactive session survives a device switch.
        """
        dev = str(device).strip().lower()
        if dev in ("", "auto"):
            return self  # sentinel: keep the currently selected device
        if dev.isdigit():
            dev = f"cuda:{dev}"
        target = torch.device(dev)
        if target != self.device:
            self.device = target
            self.model.to(target)
            if self._image_embeddings is not None:
                self._image_embeddings = self._move_embeddings(
                    self._image_embeddings, target
                )
        return self

    # ---- prediction internals ------------------------------------------------

    @_synchronized_image_state
    def _resolve_image(self, source, color_format):
        """Return ``(pil_image, path, cached_embeddings_or_None)``."""
        if source is not None:
            img = ImageLoader.load(source, color_format=color_format)
            path = source if isinstance(source, (str, Path)) else None
            return img, path, None
        if self._image is None:
            raise RuntimeError(
                "No image set. Pass source=... or call set_image(...) first."
            )
        return self._image, self._image_path, self._image_embeddings

    def _encode(self, img, *, points=None, labels=None, boxes=None) -> Dict[str, Any]:
        """Preprocess one image plus optional normalized prompts."""
        proc_kwargs: Dict[str, Any] = {}
        if points is not None:
            proc_kwargs["input_points"] = [points]
            proc_kwargs["input_labels"] = [labels]
        if boxes is not None:
            proc_kwargs["input_boxes"] = [boxes]
        return self.processor(images=img, return_tensors="pt", **proc_kwargs)

    def _image_embeddings_from_encoding(self, enc):
        pixel_values = enc["pixel_values"].to(self.device, dtype=self._model_dtype)
        with torch.no_grad():
            return self.model.get_image_embeddings(pixel_values)

    def _build_model_inputs(self, enc, cached_emb) -> Dict[str, Any]:
        """Assemble model kwargs, reusing cached embeddings when available."""
        inputs: Dict[str, Any] = {}
        for key in ("input_points", "input_labels", "input_boxes"):
            if key in enc:
                inputs[key] = enc[key].to(self.device)
        if cached_emb is not None:
            inputs["image_embeddings"] = cached_emb
        else:
            inputs["pixel_values"] = enc["pixel_values"].to(
                self.device, dtype=self._model_dtype
            )
        return inputs

    def _post_process_masks(self, pred_masks, enc):
        """Upscale low-res masks to the original image resolution."""
        fn = getattr(self.processor, "post_process_masks", None)
        if fn is None:
            fn = self.processor.image_processor.post_process_masks
        return fn(
            pred_masks.cpu(),
            enc["original_sizes"],
            enc["reshaped_input_sizes"],
        )

    def _post_process(self, outputs, enc, multimask):
        """Return ``(masks (Q,H,W) bool, conf (Q,))`` selecting the best mask."""
        masks = self._post_process_masks(outputs.pred_masks, enc)[0]  # (Q, M, H, W)
        scores = outputs.iou_scores[0].detach().cpu().float()  # (Q, M)
        return self._select_best(masks, scores, multimask)

    @staticmethod
    def _select_best(masks, scores, multimask):
        """Select masks per prompt from the model's ``M`` candidates.

        ``multimask=True`` keeps every candidate — SAM's whole-vs-part ambiguity
        masks — so ``(Q, M, H, W)`` flattens to ``(Q*M, H, W)`` and the caller
        gets all three. Otherwise the single highest-IoU candidate per prompt is
        returned.
        """
        masks = masks.cpu()
        scores = scores.cpu().float()
        if multimask:
            q, m = masks.shape[0], masks.shape[1]
            sel = masks.reshape(q * m, *masks.shape[2:])
            conf = scores.reshape(q * m)
        else:
            idx = scores.argmax(dim=1)
            q = torch.arange(masks.shape[0])
            sel = masks[q, idx]
            conf = scores[q, idx]
        return sel.bool(), conf.float()

    def _assemble_results(
        self, masks, conf, img, image_path, *, conf_thres, max_det, names=None
    ) -> Results:
        from torchvision.ops import masks_to_boxes

        width, height = img.size
        orig_shape = (height, width)

        if conf_thres > 0:
            keep = conf >= conf_thres
            masks, conf = masks[keep], conf[keep]
        # Drop empty masks (a box cannot be derived from them).
        if masks.shape[0]:
            nonempty = masks.flatten(1).any(dim=1)
            masks, conf = masks[nonempty], conf[nonempty]
        if masks.shape[0] > max_det:
            top = conf.argsort(descending=True)[:max_det]
            masks, conf = masks[top], conf[top]

        if masks.shape[0] == 0:
            return self._empty_results(orig_shape, image_path, names=names)

        boxes = masks_to_boxes(masks.float())
        cls = torch.zeros(masks.shape[0], dtype=torch.float32)
        return Results(
            boxes=Boxes(boxes, conf, cls),
            orig_shape=orig_shape,
            path=str(image_path) if image_path else None,
            names=self.names if names is None else names,
            masks=Masks(masks, orig_shape),
        )

    def _empty_results(self, orig_shape, image_path, *, names=None) -> Results:
        return Results(
            boxes=Boxes(
                torch.zeros((0, 4), dtype=torch.float32),
                torch.zeros(0, dtype=torch.float32),
                torch.zeros(0, dtype=torch.float32),
            ),
            orig_shape=orig_shape,
            path=str(image_path) if image_path else None,
            names=self.names if names is None else names,
        )

    def _segment_everything(
        self,
        img,
        image_path,
        *,
        points_per_side,
        pred_iou_thresh,
        max_det,
        cached_emb,
    ) -> Results:
        """Basic automatic mask generation by prompting a dense point grid.

        A deliberately simplified AMG: prompt the model on a uniform grid, keep
        the best mask per point, drop low-confidence/empty masks, and dedupe
        with box-IoU NMS. It is NOT a drop-in for the reference
        AutomaticMaskGenerator — it omits stability-score filtering, multi-crop,
        and mask-IoU dedup, so it under-segments crowded scenes and can merge
        distinct objects that share a bounding box. The prompted path
        (points/boxes) is the precise, recommended API; reach upstream for
        exhaustive automatic masks.
        """
        from torchvision.ops import batched_nms, masks_to_boxes

        width, height = img.size
        orig_shape = (height, width)

        # Encode the image once even in one-shot mode, so the grid reuses it.
        if cached_emb is None:
            enc0 = self._encode(img)
            cached_emb = self._image_embeddings_from_encoding(enc0)

        grid = build_point_grid(points_per_side)
        pixel_points = [[float(x * width), float(y * height)] for x, y in grid]
        # Preprocess the image and all grid points ONCE. The per-image resize
        # metadata (original_sizes/reshaped_input_sizes) is identical for every
        # chunk, so only the prompt tensor is sliced below — this avoids
        # re-running the full image preprocessing once per chunk.
        enc = self._encode(
            img,
            points=[[p] for p in pixel_points],
            labels=[[1] for _ in pixel_points],
        )
        full_points = enc["input_points"].to(self.device)  # (1, N, 1, 2)
        full_labels = enc["input_labels"].to(self.device)  # (1, N, 1)

        all_masks: List[torch.Tensor] = []
        all_conf: List[torch.Tensor] = []
        for i in range(0, full_points.shape[1], self.GRID_CHUNK):
            chunk = full_points[:, i : i + self.GRID_CHUNK]
            chunk_labels = full_labels[:, i : i + self.GRID_CHUNK]
            with torch.no_grad():
                outputs = self.model(
                    input_points=chunk,
                    input_labels=chunk_labels,
                    image_embeddings=cached_emb,
                    multimask_output=True,
                )
            # Keep the best of the 3 candidates per grid point.
            sel, conf = self._post_process(outputs, enc, multimask=False)
            all_masks.append(sel)
            all_conf.append(conf)

        masks = torch.cat(all_masks, dim=0)
        conf = torch.cat(all_conf, dim=0)

        # Only filter when a positive threshold is set — SAM's IoU head is
        # unbounded, so a 0.0 threshold (keep all) must not drop negative
        # scores, matching the prompted path's conf=0.0 behavior.
        if pred_iou_thresh > 0:
            keep = conf >= pred_iou_thresh
            masks, conf = masks[keep], conf[keep]
        if masks.shape[0]:
            nonempty = masks.flatten(1).any(dim=1)
            masks, conf = masks[nonempty], conf[nonempty]
        if masks.shape[0] == 0:
            return self._empty_results(orig_shape, image_path)

        boxes = masks_to_boxes(masks.float())
        cls = torch.zeros(masks.shape[0], dtype=torch.int64)
        keep = batched_nms(boxes, conf, cls, self.AMG_NMS_IOU)[:max_det]
        masks, conf, boxes = masks[keep], conf[keep], boxes[keep]
        cls_final = torch.zeros(masks.shape[0], dtype=torch.float32)

        return Results(
            boxes=Boxes(boxes, conf, cls_final),
            orig_shape=orig_shape,
            path=str(image_path) if image_path else None,
            names=self.names,
            masks=Masks(masks, orig_shape),
        )

    # =========================================================================
    # Out of scope for the inference-first SAM tier
    # =========================================================================

    def track(self, *args, **kwargs):
        raise NotImplementedError(
            f"{type(self).__name__} is an image segmenter; video tracking is "
            "out of scope for LibreSAM v1. Use predict() per frame."
        )

    def train(self, *args, **kwargs):
        raise NotImplementedError(
            f"Training is out of scope for {type(self).__name__} in LibreYOLO. "
            "Fine-tune the model upstream and load the resulting weights."
        )

    def val(self, *args, **kwargs):
        raise NotImplementedError(
            f"Dataset validation is not supported for {type(self).__name__}: "
            "promptable masks have no fixed class set to score against."
        )

    def export(self, format: str = "onnx", **kwargs) -> str:
        raise NotImplementedError(
            f"{type(self).__name__} export is out of scope for LibreSAM v1. "
            "Run it through predict() instead."
        )
