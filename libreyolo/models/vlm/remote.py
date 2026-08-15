"""Remote LibreVLM: hosted vision chat models as open-vocabulary detectors.

Same contract as a local VLM family (``set_classes`` + ``predict`` returning
pixel-``xyxy`` ``Results``, ``chat()`` as the raw-text escape hatch), with the
forward pass replaced by one OpenAI-compatible HTTP request per image. See
``docs/adr/0020-librevlm-remote-transport.md``.

Ground rules implemented here:

- An empty result is never ambiguous. HTTP failure (after the SDK's own
  retries), parse failure, and refusal all yield empty boxes and attach
  ``result.remote = {"error": kind, ...}``; a clean "found nothing" carries
  no error. Multi-image runs log a failure summary.
- Live sources (webcam, network streams, screen capture) raise: every frame
  is a metered API call. Finite sources (image, list, folder, video file)
  work like local.
- ``batch=`` means request concurrency (a thread pool over per-image HTTP),
  not a stacked tensor. Default 8.
- Auth / bad-model / bad-request errors raise loudly; they would fail on
  every image, so they are never converted into empty results.
"""

from __future__ import annotations

import logging
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Generator, Optional, Tuple

import torch

from ...utils.image_loader import ImageInput, ImageLoader
from ...utils.logging import ensure_default_logging
from ...utils.results import Results
from ...utils.source import SourceKind, classify_source
from ..base.inference import InferenceRunner
from ..llm.openai_transport import (
    SUPPORTED_APIS,
    _encode_pil,
    _load_openai,
    build_user_payload,
    client_kwargs,
    response_text,
)
from .base import LibreVLMModel
from .parsing import (
    _iou_xyxy,
    build_detection_dict,
    extract_detections,
    normalize_bbox,
    to_xyxy,
)

logger = logging.getLogger(__name__)

_INFERENCE_ONLY = (
    "Remote LibreVLM is inference-only: the weights live on the provider's "
    "servers. Use a local family (e.g. LibreVLM('qwen3-vl-4b')) or LibreYOLO "
    "for train/val/export."
)
_LIVE_SOURCE_MESSAGE = (
    "Remote LibreVLM does not support live sources (webcam indices, network "
    "streams, screen capture): every frame is a metered API call with seconds "
    "of latency. Pass an image, a list, a folder, or a video file."
)
_DEFAULT_CONCURRENCY = 8

# prefix -> (default base_url, env key, base_url required)
_PROVIDERS: Dict[str, Tuple[Optional[str], str, bool]] = {
    "openai": (None, "OPENAI_API_KEY", False),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", False),
    "openai-compat": (None, "OPENAI_API_KEY", True),
}

_REFUSAL_MARKERS = (
    "i'm sorry",
    "i am sorry",
    "i cannot",
    "i can't",
    "i can not",
    "unable to",
    "not able to",
    "cannot assist",
    "can't assist",
    "cannot help",
    "can't help",
    "against my",
    "as an ai",
    "content policy",
    "i must decline",
    "refuse",
)


class _RemotePayload:
    """Encoded request for one image. Duck-types ``.to(device)`` as a no-op so
    the shared ``_predict_single`` path needs no remote special-casing."""

    __slots__ = ("image_url",)

    def __init__(self, image_url: str):
        self.image_url = image_url

    def to(self, *args, **kwargs):
        return self


class _RemoteReply:
    """One decoded response: the text plus whether it was cut off.

    A truncated answer is a per-image event (a crowded image overflows the
    token budget where a sparse one does not), so it is reported through the
    side channel rather than raised.
    """

    __slots__ = ("text", "truncated")

    def __init__(self, text: str, truncated: bool = False):
        self.text = text
        self.truncated = truncated


class _RemoteFailure:
    """Sentinel returned by ``_forward`` when a request failed in a way that
    is per-image rather than fatal. Carries the kind and detail for the
    ``result.remote`` side channel."""

    __slots__ = ("kind", "detail")

    def __init__(self, kind: str, detail: str):
        self.kind = kind
        self.detail = detail


class _RemoteInferenceRunner(InferenceRunner):
    """InferenceRunner that attaches the per-image error side channel.

    ``_wrap_results`` is the single funnel every predict path (single image,
    folder, list, video frame, tracked frame) passes through, so attaching
    here covers all of them without duplicating routing logic. The pending
    error is thread-local: pooled folder runs execute the whole
    preprocess -> forward -> postprocess -> wrap chain on one worker thread.
    """

    def _wrap_results(self, detections, original_size, image_path, classes):
        result = super()._wrap_results(detections, original_size, image_path, classes)
        error = self.model._consume_error()
        if error is not None:
            result.remote = error
        return result


class RemoteVLMModel(LibreVLMModel):
    """Hosted vision chat model behind the standard LibreVLM surface."""

    FAMILY = "remote-vlm"

    def __init__(
        self,
        provider: str,
        model_id: str,
        *,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        api: str = "chat.completions",
        names: Optional[list] = None,
        prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        device: Optional[str] = None,
        **kwargs,
    ):
        ensure_default_logging()
        if kwargs:
            raise TypeError(
                "Unexpected keyword arguments for remote LibreVLM: "
                f"{', '.join(sorted(kwargs))}. Remote models accept base_url, "
                "api_key, api, names, prompt, and max_new_tokens."
            )
        if device is not None:
            raise ValueError(
                "device= does not apply to a remote LibreVLM: inference runs "
                "on the provider's servers, not on a local GPU."
            )
        provider = str(provider).strip().lower()
        spec = _PROVIDERS.get(provider)
        if spec is None:
            raise ValueError(
                f"Unknown remote provider {provider!r}. Known providers: "
                f"{', '.join(sorted(_PROVIDERS))}. Self-hosted or gateway "
                "endpoints use LibreVLM('openai-compat/<model-id>', "
                "base_url=...)."
            )
        if not str(model_id).strip():
            raise ValueError("Remote model id must be a non-empty string.")
        if api not in SUPPORTED_APIS:
            raise ValueError(f"api must be one of {SUPPORTED_APIS}, got {api!r}")
        default_base_url, env_key, base_url_required = spec
        if base_url is None:
            base_url = default_base_url
        if base_url_required and base_url is None:
            raise ValueError(
                "LibreVLM('openai-compat/...') requires base_url= pointing at "
                "your OpenAI-compatible host (e.g. 'http://localhost:8000/v1')."
            )
        if api_key is None and env_key != "OPENAI_API_KEY":
            import os

            api_key = os.environ.get(env_key)
            if api_key is None:
                raise ValueError(
                    f"Provider {provider!r} reads its key from {env_key}, "
                    "which is not set. Set it or pass api_key=."
                )

        self.provider = provider
        self.model_id = str(model_id)
        self.base_url = base_url
        self.api_key = api_key
        self.api = api

        # Attributes the shared BaseModel/InferenceRunner machinery reads.
        # BaseModel.__init__ is deliberately not called: it exists to load
        # local weights, which a remote model does not have.
        self.family = self.FAMILY
        self.task = "detect"
        self.size = self.model_id
        self.device = torch.device("cpu")  # never used; payloads no-op .to()
        self.input_size = 1024  # nominal; the provider picks its own scaling
        self.model = None
        self.model_path = None
        self._checkpoint_dir = None
        self._custom_prompt = prompt
        self._graph_runner = None
        self._cuda_graph_mode = None
        self._runner_instance = None
        self._token_cap = int(max_new_tokens) if max_new_tokens else None
        self._pending = threading.local()

        self.nb_classes = 80
        from ...utils.general import COCO_CLASSES

        self.names = {i: n for i, n in enumerate(COCO_CLASSES)}
        self._name_to_id = {v.strip().lower(): k for k, v in self.names.items()}
        if names is not None:
            self.set_classes(names)

        # Built eagerly so a missing key fails loud at construction, not on
        # image 7,001 of a folder. Also surfaces the pip-extra hint early.
        self._client_instance = self._make_client()

    # =========================================================================
    # HTTP client
    # =========================================================================

    def _make_client(self):
        openai = _load_openai()
        return openai.OpenAI(**client_kwargs(self.api_key, self.base_url))

    def _request(
        self,
        image_url: str,
        prompt: str,
        max_new_tokens: Optional[int] = None,
    ) -> _RemoteReply:
        """One image+prompt request; returns the text and a truncation flag."""
        cap = max_new_tokens if max_new_tokens is not None else self._token_cap
        payload = build_user_payload(self.api, prompt, image_url)
        body: Dict[str, Any] = {"model": self.model_id}
        if self.api == "responses":
            body["input"] = payload
            if cap:
                body["max_output_tokens"] = int(cap)
            response = self._client_instance.responses.create(**body)
            details = getattr(response, "incomplete_details", None)
            truncated = getattr(response, "status", None) == "incomplete" and (
                getattr(details, "reason", None) == "max_output_tokens"
            )
        else:
            body["messages"] = payload
            if cap:
                # First-party OpenAI rejects max_tokens on current models;
                # every compat host (vLLM, Ollama, OpenRouter) speaks it.
                key = (
                    "max_completion_tokens"
                    if self.provider == "openai"
                    else "max_tokens"
                )
                body[key] = int(cap)
            response = self._client_instance.chat.completions.create(**body)
            choices = getattr(response, "choices", None) or []
            truncated = bool(choices) and (
                getattr(choices[0], "finish_reason", None) == "length"
            )
        return _RemoteReply(response_text(self.api, response), truncated)

    def _request_text(
        self,
        image_url: str,
        prompt: str,
        max_new_tokens: Optional[int] = None,
    ) -> str:
        """One image+prompt request; returns the model's text verbatim."""
        return self._request(image_url, prompt, max_new_tokens=max_new_tokens).text

    @staticmethod
    def _is_truncation_error(exc: BaseException) -> bool:
        """True for the "output limit reached" 400 that OpenAI returns instead
        of a truncated message.

        Matched on the message because the API gives no distinct error code.
        Both a length signal and a "was reached" signal are required, so a
        genuine bad request such as "max_tokens must be an integer" (fatal,
        identical on every image) is not swallowed as a per-image event.
        """
        message = str(exc).lower()
        length_signal = "max_tokens" in message or "max_output_tokens" in message
        reached_signal = "was reached" in message or "output limit" in message
        return length_signal and reached_signal

    @classmethod
    def _failure_kind(cls, exc: BaseException) -> Optional[str]:
        """Per-image failure kind, or None when the error must raise.

        ``"http"`` covers transient transport failures (already retried by the
        SDK). ``"truncated"`` covers the output-limit 400: it depends on how
        much the model had to say about *this* image, so it must not abort a
        folder. Everything else (auth, bad model, malformed request) fails on
        every image, and converting those into empty results would silently
        zero a whole run.
        """
        if cls._is_truncation_error(exc):
            return "truncated"
        try:
            openai = _load_openai()
        except ImportError:
            return None
        for name in (
            "RateLimitError",
            "APIConnectionError",
            "APITimeoutError",
            "InternalServerError",
        ):
            klass = getattr(openai, name, None)
            if isinstance(klass, type) and isinstance(exc, klass):
                return "http"
        status = getattr(exc, "status_code", None)
        try:
            if status is not None and int(status) >= 500:
                return "http"
        except (TypeError, ValueError):
            pass
        return None

    # =========================================================================
    # Error side channel (thread-local; consumed by the runner's wrap step)
    # =========================================================================

    def _record_error(self, kind: str, detail: str) -> None:
        self._pending.error = {
            "error": kind,
            "detail": detail[:500],
            "model": f"{self.provider}/{self.model_id}",
        }

    def _consume_error(self) -> Optional[dict]:
        error = getattr(self._pending, "error", None)
        self._pending.error = None
        return error

    @staticmethod
    def _classify_empty(text: str) -> Optional[str]:
        """Why did a response yield zero detections?

        None means clean (the model followed the format and found nothing).
        ``"refusal"`` and ``"parse"`` are non-clean and get the side channel.
        A chatty prose answer without boxes counts as a parse failure even
        when it reads like a negative: the model broke the format contract,
        and the caller deserves to know before trusting it as a negative.
        """
        stripped = (text or "").strip()
        if stripped.startswith("```"):
            stripped = stripped.strip("`").removeprefix("json").strip()
        if not stripped:
            return "parse"
        compact = "".join(stripped.split())
        if compact in ("[]", "[{}]"):
            return None
        if "[]" in compact and len(compact) <= 40:
            return None
        lowered = stripped.lower()
        if any(marker in lowered for marker in _REFUSAL_MARKERS):
            return "refusal"
        return "parse"

    @staticmethod
    def _normalize_item(item: dict) -> dict:
        """Map a parsed item onto the normalized ``bbox`` [0, 1] convention.

        The generic prompt asks for ``bbox`` in [0, 1], but Qwen-lineage
        hosted models often answer in their native ``bbox_2d`` 0-1000 grid
        regardless; values above ~1 are rescaled. Anything else out of range
        is clamped/dropped downstream by ``normalize_bbox``.
        """
        raw = item.get("bbox")
        if raw is None:
            raw = item.get("bbox_2d")
        if not isinstance(raw, (list, tuple)) or len(raw) != 4:
            return item
        try:
            vals = [float(v) for v in raw]
        except (TypeError, ValueError):
            return item
        if any(abs(v) > 1.5 for v in vals):
            vals = [v / 1000.0 for v in vals]
        return {**item, "bbox": vals}

    # =========================================================================
    # Runner hooks: preprocess -> forward (HTTP) -> postprocess
    # =========================================================================

    @property
    def _runner(self):
        if self._runner_instance is None:
            self._runner_instance = _RemoteInferenceRunner(self)
        return self._runner_instance

    def _get_input_size(self) -> int:
        return self.input_size

    def _get_available_layers(self) -> Dict[str, Any]:
        return {}

    def _preprocess(
        self,
        image: ImageInput,
        color_format: str = "auto",
        input_size: Optional[int] = None,
    ) -> Tuple[Any, Any, Tuple[int, int], float]:
        img = ImageLoader.load(image, color_format=color_format)
        return _RemotePayload(_encode_pil(img)), img, img.size, 1.0

    def _forward(self, inputs: Any):
        prompt = self._detection_prompt()
        try:
            return self._request(inputs.image_url, prompt)
        except Exception as exc:
            kind = self._failure_kind(exc)
            if kind is None:
                raise
            return _RemoteFailure(kind, f"{type(exc).__name__}: {exc}")

    def _postprocess(
        self,
        output: Any,
        conf_thres: float,
        iou_thres: float,
        original_size: Tuple[int, int],
        max_det: int = 300,
        ratio: float = 1.0,
        **kwargs,
    ) -> Dict:
        if isinstance(output, _RemoteFailure):
            self._record_error(output.kind, output.detail)
            items = []
        else:
            text = output.text if isinstance(output, _RemoteReply) else ""
            items = [self._normalize_item(i) for i in extract_detections(text)]
            if isinstance(output, _RemoteReply) and output.truncated:
                # Whatever parsed is kept (partial boxes beat none), but the
                # list is incomplete by definition, so it is never a clean
                # result.
                self._record_error("truncated", text)
            elif not items:
                kind = self._classify_empty(text)
                if kind is not None:
                    self._record_error(kind, text)
        return build_detection_dict(
            items,
            self._name_to_id,
            original_size,
            conf_thres=conf_thres,
            max_det=max_det,
            classes=kwargs.get("classes"),
            default_score=self.DEFAULT_SCORE,
            bbox_key="bbox",
            coord_divisor=1.0,
            box_format="xyxy",
            iou_thres=iou_thres,
        )

    # =========================================================================
    # Public surface: guards, metered banner, concurrency, summary
    # =========================================================================

    @staticmethod
    def _reject_remote_kwargs(kwargs: dict) -> None:
        if kwargs.get("device") is not None:
            raise ValueError(
                "device= does not apply to a remote LibreVLM: inference runs "
                "on the provider's servers, not on a local GPU."
            )
        if kwargs.get("tiling"):
            raise ValueError(
                "tiling= is meaningless for a remote generator; the provider "
                "controls its own image scaling."
            )
        if kwargs.get("augment"):
            raise ValueError(
                "augment= (test-time augmentation) is meaningless for a "
                "remote generator."
            )
        if kwargs.get("cuda_graph"):
            raise ValueError("cuda_graph= does not apply to a remote LibreVLM.")

    def _announce_metered(self, count: Optional[int], unit: str) -> None:
        amount = f"~{count} {unit}" if count else f"multiple {unit}"
        logger.warning(
            "Remote LibreVLM: %s will be sent to %s (%s), a paid/metered "
            "endpoint.",
            amount,
            self.provider,
            self.model_id,
        )

    def _summarize(self, results) -> None:
        failures = Counter(
            r.remote["error"]
            for r in results
            if isinstance(getattr(r, "remote", None), dict) and r.remote.get("error")
        )
        if not failures:
            return
        failed = sum(failures.values())
        detail = ", ".join(f"{k}: {v}" for k, v in sorted(failures.items()))
        logger.warning(
            "Remote LibreVLM run: %d ok, %d failed (%s); failed results carry "
            "result.remote.",
            len(results) - failed,
            failed,
            detail,
        )

    def _summarizing_stream(self, gen) -> Generator[Results, None, None]:
        seen = []
        try:
            for result in gen:
                seen.append(result)
                yield result
        finally:
            self._summarize(seen)

    @staticmethod
    def _video_frame_estimate(source, vid_stride: int = 1) -> Optional[int]:
        try:
            import cv2

            cap = cv2.VideoCapture(str(source))
            try:
                frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            finally:
                cap.release()
            if frames > 0:
                return max(1, frames // max(1, int(vid_stride)))
        except Exception:
            pass
        return None

    def __call__(self, source: Any = None, **kwargs):
        self._reject_remote_kwargs(kwargs)
        spec = classify_source(source)
        if spec.live or spec.kind == SourceKind.SCREEN:
            raise ValueError(_LIVE_SOURCE_MESSAGE)

        concurrency = kwargs.pop("batch", None) or _DEFAULT_CONCURRENCY
        concurrency = max(1, int(concurrency))
        stream = bool(kwargs.get("stream"))

        if spec.kind == SourceKind.VIDEO:
            estimate = self._video_frame_estimate(
                spec.source, kwargs.get("vid_stride", 1)
            )
            self._announce_metered(estimate, "frames")
            out = self._runner(source, **kwargs)
            if stream:
                return self._summarizing_stream(out)
            self._summarize(out)
            return out

        images = None
        if spec.kind == SourceKind.DIRECTORY:
            images = ImageLoader.collect_images(spec.source)
        elif spec.kind == SourceKind.IMAGE_BATCH:
            images = list(spec.items)

        if images is not None:
            if not images:
                return iter(()) if stream else []
            self._announce_metered(len(images), "images")
            if stream:
                return self._summarizing_stream(
                    self._runner(source, batch=1, **kwargs)
                )
            # ``batch`` is request concurrency here, never a stacked tensor.
            # In-memory items with save=True keep the sequential path so the
            # runner's indexed save stems cannot collide across threads.
            poolable = concurrency > 1 and len(images) > 1
            if kwargs.get("save") and not all(
                isinstance(im, (str, Path)) for im in images
            ):
                poolable = False
            if not poolable:
                results = self._runner(source, batch=1, **kwargs)
            else:
                runner = self._runner  # materialize once before forking
                with ThreadPoolExecutor(
                    max_workers=min(concurrency, len(images))
                ) as pool:
                    results = list(
                        pool.map(lambda im: runner(im, **kwargs), images)
                    )
            self._summarize(results)
            return results

        # Single image (path or in-memory).
        return self._runner(source, **kwargs)

    def predict(self, *args, **kwargs):
        return self(*args, **kwargs)

    def track(self, source, **kwargs):
        spec = classify_source(source)
        if spec.live or spec.kind == SourceKind.SCREEN:
            raise ValueError(_LIVE_SOURCE_MESSAGE)
        estimate = self._video_frame_estimate(source, kwargs.get("vid_stride", 1))
        self._announce_metered(estimate, "frames")
        return super().track(source, **kwargs)

    # =========================================================================
    # chat(): the raw escape hatch (same signature/return as local)
    # =========================================================================

    def chat(
        self,
        image: ImageInput,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        color_format: str = "auto",
    ) -> str:
        """Raw multimodal generation over HTTP: image + prompt in, text out.

        Single-shot and loud: transport errors raise. For streaming, async,
        history, or the native SDK object, use ``LibreLLM``.
        """
        img = ImageLoader.load(image, color_format=color_format)
        return self._request_text(
            _encode_pil(img), str(prompt), max_new_tokens=max_new_tokens
        )

    # =========================================================================
    # selftest(): can this hosted model actually ground?
    # =========================================================================

    def selftest(self, iou_pass: float = 0.5) -> dict:
        """Probe whether this hosted model returns usable boxes.

        Two metered calls: a red rectangle on white at a known position
        (expects one parseable box with IoU >= ``iou_pass``), and a blank
        gray image (expects zero boxes). Makes no mAP claim; it is a cheap
        go/no-go before pointing a folder at an unknown model. Does not
        touch the sticky ``set_classes`` vocabulary.

        Returns a dict: ``passed``, ``iou``, ``false_positives``, ``raw``
        (the two response texts).
        """
        from PIL import Image, ImageDraw

        width, height = 640, 480
        target = (0.2, 0.2, 0.8, 0.8)
        positive = Image.new("RGB", (width, height), (255, 255, 255))
        ImageDraw.Draw(positive).rectangle(
            (
                int(target[0] * width),
                int(target[1] * height),
                int(target[2] * width),
                int(target[3] * height),
            ),
            fill=(220, 30, 30),
        )
        blank = Image.new("RGB", (width, height), (128, 128, 128))

        prompt = self._custom_prompt or self._format_detection_prompt(
            "red rectangle"
        )

        def probe_boxes(img):
            text = self._request_text(_encode_pil(img), prompt)
            boxes = []
            for item in extract_detections(text):
                normalized = self._normalize_item(item)
                box = normalize_bbox(to_xyxy(normalized.get("bbox")))
                if box is not None:
                    boxes.append(box)
            return text, boxes

        raw_positive, boxes_positive = probe_boxes(positive)
        raw_blank, boxes_blank = probe_boxes(blank)

        best_iou = max(
            (_iou_xyxy(box, target) for box in boxes_positive), default=0.0
        )
        false_positives = len(boxes_blank)
        passed = best_iou >= iou_pass and false_positives == 0
        if passed:
            logger.warning(
                "Remote LibreVLM selftest PASSED for %s/%s: probe IoU %.2f, "
                "no false positives on a blank image.",
                self.provider,
                self.model_id,
                best_iou,
            )
        else:
            logger.warning(
                "Remote LibreVLM selftest FAILED for %s/%s: probe IoU %.2f "
                "(need >= %.2f), %d box(es) on a blank image. This model does "
                "not ground reliably with the generic prompt; use chat(), "
                "pass prompt=, or pick a different model.",
                self.provider,
                self.model_id,
                best_iou,
                iou_pass,
                false_positives,
            )
        return {
            "passed": passed,
            "iou": best_iou,
            "false_positives": false_positives,
            "raw": [raw_positive, raw_blank],
        }

    # =========================================================================
    # Out of scope
    # =========================================================================

    def train(self, *args, **kwargs):
        raise NotImplementedError(_INFERENCE_ONLY)

    def val(self, *args, **kwargs):
        raise NotImplementedError(_INFERENCE_ONLY)

    def export(self, *args, **kwargs):
        raise NotImplementedError(_INFERENCE_ONLY)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(provider={self.provider!r}, "
            f"model={self.model_id!r}, api={self.api!r})"
        )
