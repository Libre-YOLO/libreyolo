"""Shared OpenAI-compatible transport helpers.

One internal module, two doors: ``LibreLLM`` (chat client, native SDK
responses out) and the remote ``LibreVLM`` transport (detection over HTTP)
both encode images, build user payloads, and construct SDK clients through
these helpers. Everything here is offline-testable; the ``openai`` package
is imported lazily so importing libreyolo never requires the ``llm`` extra.

The official OpenAI Python SDK is Apache-2.0. This module talks to that
public API. It does not wrap provider-specific response types.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
from PIL import Image, ImageOps

SUPPORTED_APIS = ("responses", "chat.completions")
_INSTALL_HINT = (
    "This feature requires the 'llm' extra (the official OpenAI SDK). "
    "Install with:\n"
    "    pip install 'libreyolo[llm]'"
)
_DATA_URI_PREFIX = "data:"
_HTTP_PREFIXES = ("http://", "https://")


def _load_openai():
    try:
        import openai
    except ImportError as exc:
        raise ImportError(_INSTALL_HINT) from exc
    return openai


def _data_uri(jpeg_bytes: bytes) -> str:
    encoded = base64.standard_b64encode(jpeg_bytes).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _encode_pil(img: Image.Image) -> str:
    transposed = ImageOps.exif_transpose(img)
    if transposed is not None:
        img = transposed
    if img.mode != "RGB":
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return _data_uri(buf.getvalue())


def _encode_path(path: str) -> str:
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Image file not found: {path}")
    with Image.open(file_path) as img:
        return _encode_pil(img)


def _encode_bgr_numpy(arr: np.ndarray) -> str:
    """Encode an OpenCV-style BGR (or gray) array as a JPEG data URI."""
    import cv2

    image = np.ascontiguousarray(arr)
    if image.ndim == 3 and image.shape[0] in (1, 3, 4) and image.shape[0] < image.shape[2]:
        image = np.transpose(image, (1, 2, 0))
    if image.dtype != np.uint8:
        if np.issubdtype(image.dtype, np.floating) and float(np.max(image)) <= 1.0:
            image = (image * 255.0).clip(0, 255).astype(np.uint8)
        else:
            image = image.clip(0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", image)
    if not ok:
        raise ValueError("Failed to encode NumPy image as JPEG")
    return _data_uri(buf.tobytes())


def _is_image_object(value: Any) -> bool:
    return isinstance(value, (Image.Image, np.ndarray, Path))


def _as_image_url(value: Any) -> str:
    if isinstance(value, str):
        if value.startswith(_HTTP_PREFIXES) or value.startswith(_DATA_URI_PREFIX):
            return value
        return _encode_path(value)
    if isinstance(value, Path):
        return _encode_path(str(value))
    if isinstance(value, Image.Image):
        return _encode_pil(value)
    if isinstance(value, np.ndarray):
        return _encode_bgr_numpy(value)
    raise TypeError(
        "image must be a path, HTTP URL, data URI, NumPy array, or PIL image, "
        f"got {type(value).__name__}"
    )


def build_user_payload(
    api: str, text: Optional[str], image_url: Optional[str]
) -> Union[str, list]:
    """Build the single-turn user payload for the given wire API.

    Returns the value for ``input=`` (responses) or ``messages=``
    (chat.completions). Text-only responses input stays a bare string, which
    is the SDK's cheapest accepted shape.
    """
    if api == "responses":
        if image_url is None:
            return text
        content: list[dict[str, Any]] = []
        if text:
            content.append({"type": "input_text", "text": text})
        content.append({"type": "input_image", "image_url": image_url})
        return [{"role": "user", "content": content}]

    if image_url is None:
        return [{"role": "user", "content": text}]
    content = []
    if text:
        content.append({"type": "text", "text": text})
    content.append({"type": "image_url", "image_url": {"url": image_url}})
    return [{"role": "user", "content": content}]


def response_text(api: str, response: Any) -> str:
    """Decode a plain string out of a native SDK response object."""
    if api == "responses":
        return getattr(response, "output_text", None) or ""
    choices = getattr(response, "choices", None) or []
    if not choices:
        return ""
    return getattr(choices[0].message, "content", None) or ""


def client_kwargs(api_key: Optional[str], base_url: Optional[str]) -> dict[str, Any]:
    """Constructor kwargs for ``OpenAI`` / ``AsyncOpenAI``, omitting unset."""
    kwargs: dict[str, Any] = {}
    if api_key is not None:
        kwargs["api_key"] = api_key
    if base_url is not None:
        kwargs["base_url"] = base_url
    return kwargs
