"""OpenAI-compatible LibreLLM client.

Implements the de-facto ecosystem LLM call shape from public documentation:
constructor ``(model, api=, base_url=, api_key=, prompt=, **kwargs)``,
``__call__(source, image=)``, ``async_call``, native SDK response out.

The official OpenAI Python SDK is Apache-2.0. This module talks to that
public API. It does not wrap provider-specific response types.
"""

from __future__ import annotations

import os
from typing import Any, Optional, Union

# Image encoding, payload building, and SDK loading live in the transport
# module shared with the remote LibreVLM path. Re-imported here so existing
# ``client._load_openai`` / ``client._as_image_url`` references keep working.
from .openai_transport import (
    SUPPORTED_APIS as _SUPPORTED_APIS,
    _as_image_url,
    _is_image_object,
    _load_openai,
    build_user_payload,
    client_kwargs,
)

_DEFAULT_MODEL = "gpt-5.6-luna"
_UNSUPPORTED = (
    "LibreLLM is inference-only. Use LibreYOLO for train, val, export, "
    "track, and benchmark."
)

# Optional ``provider/`` prefixes, for symmetry with ``LibreVLM``. The bare
# form (``LibreLLM("gpt-5.6-luna")``) stays primary and is never deprecated.
# A known prefix is stripped and supplies the provider's default host and env
# key; anything else is an opaque model id sent to the configured host.
_KNOWN_PROVIDERS: dict[str, tuple[Optional[str], str]] = {
    # prefix -> (default base_url, env key)
    "openai": (None, "OPENAI_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
}


def _resolve_provider(model: str) -> tuple[str, Optional[str], Optional[str]]:
    """Strip a known ``provider/`` prefix from *model*.

    Returns ``(model_id, default_base_url, env_key)``. Splits on the first
    slash only, before lowercasing, so case-sensitive provider model ids
    survive. An unknown prefix leaves the whole string as the model id
    (current behavior; OpenRouter slugs like ``qwen/qwen3-max`` keep working
    against an explicit ``base_url=``).
    """
    if "/" in model:
        prefix, rest = model.split("/", 1)
        known = _KNOWN_PROVIDERS.get(prefix.lower())
        if known is not None:
            return rest, known[0], known[1]
    return model, None, None


def _is_vlm_alias(model: str) -> bool:
    """True when *model* is a ``LibreVLM`` local alias (detector, not chat).

    The bare-name namespace stays open (a hosted model that ships tomorrow
    must work today), so protection against billing accidents is this small
    denylist against the disjoint VLM alias table, never a closed allowlist.
    """
    try:
        from ..vlm import _ALIASES, _LAZY_ALIASES, _MODUS_ALIASES
    except Exception:  # pragma: no cover - torch-less or partial installs
        return False
    key = model.strip().lower()
    return key in _ALIASES or key in _LAZY_ALIASES or key in _MODUS_ALIASES


def _compose_text(prompt: Optional[str], source_text: Optional[str]) -> Optional[str]:
    if prompt and source_text:
        return f"{prompt}\n{source_text}"
    return source_text or prompt


class LibreLLM:
    """OpenAI-compatible language and vision chat client.

    Args:
        model: Model identifier sent to the selected endpoint.
        api: ``"responses"`` (default) or ``"chat.completions"``.
        base_url: Optional OpenAI-compatible endpoint.
        api_key: API key; otherwise the SDK reads ``OPENAI_API_KEY``.
        prompt: Instruction prepended to plain text and image requests.
        **kwargs: Default SDK request arguments. Per-call arguments override
            matching constructor values.
    """

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        *,
        api: str = "responses",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        if not str(model).strip():
            raise ValueError("model must be a non-empty string")
        if api not in _SUPPORTED_APIS:
            raise ValueError(
                f"api must be one of {_SUPPORTED_APIS}, got {api!r}"
            )
        model_id, provider_base_url, provider_env_key = _resolve_provider(str(model))
        if base_url is None and "/" not in str(model) and _is_vlm_alias(model_id):
            raise ValueError(
                f"{model_id!r} is a LibreVLM local alias (a detector), not a "
                "hosted model id.\n"
                f"  detector: LibreVLM({model_id!r})\n"
                "  really a hosted model with this exact id: pass base_url= "
                "to send it there"
            )
        if base_url is None:
            base_url = provider_base_url
        if api_key is None and provider_env_key is not None:
            # Resolved explicitly so a non-OpenAI prefix never silently bills
            # whatever OPENAI_API_KEY happens to point at.
            api_key = os.environ.get(provider_env_key)
            if api_key is None and provider_env_key != "OPENAI_API_KEY":
                raise ValueError(
                    f"Model {model!r} uses a provider prefix whose key is "
                    f"read from {provider_env_key}, which is not set. Set it "
                    "or pass api_key=."
                )
        self.model = model_id
        self.api = api
        self.base_url = base_url
        self.api_key = api_key
        self.prompt = prompt
        self.defaults = dict(kwargs)
        self._sync = None
        self._async = None

    def _client_kwargs(self) -> dict[str, Any]:
        return client_kwargs(self.api_key, self.base_url)

    def _make_sync_client(self):
        return _load_openai().OpenAI(**self._client_kwargs())

    def _make_async_client(self):
        return _load_openai().AsyncOpenAI(**self._client_kwargs())

    def _sync_client(self):
        if self._sync is None:
            self._sync = self._make_sync_client()
        return self._sync

    def _async_client(self):
        if self._async is None:
            self._async = self._make_async_client()
        return self._async

    def _prepare(
        self,
        source: Any,
        image: Any,
    ) -> Union[str, list]:
        if isinstance(source, list):
            if image is not None:
                raise ValueError(
                    "image= cannot be combined with a native message list; "
                    "put the image in the messages or pass text as source"
                )
            return source

        image_url = None
        source_text: Optional[str] = None

        if _is_image_object(source):
            if image is not None:
                raise ValueError(
                    "pass the image as source or as image=, not both"
                )
            image_url = _as_image_url(source)
        elif source is None:
            pass
        elif isinstance(source, str):
            source_text = source
        else:
            raise TypeError(
                "source must be text, a native message list, or an image "
                f"object, got {type(source).__name__}"
            )

        if image is not None:
            image_url = _as_image_url(image)

        text = _compose_text(self.prompt, source_text)
        if text is None and image_url is None:
            raise ValueError(
                "LibreLLM() needs a source, image, or a constructor prompt"
            )
        return self._build_payload(text, image_url)

    def _build_payload(
        self, text: Optional[str], image_url: Optional[str]
    ) -> Union[str, list]:
        return build_user_payload(self.api, text, image_url)

    def _request_body(self, prepared: Union[str, list], kwargs: dict[str, Any]) -> dict[str, Any]:
        body = {**self.defaults, **kwargs}
        body["model"] = self.model
        if self.api == "responses":
            body["input"] = prepared
        else:
            body["messages"] = prepared
        return body

    def __call__(self, source: Any = None, /, *, image: Any = None, **kwargs: Any):
        """Send one request. ``source`` is text, a message list, or an image.

        A string such as ``"bus.jpg"`` is treated as text. Pass the image
        through ``image=``. Returns the SDK's native response object.
        """
        prepared = self._prepare(source, image)
        body = self._request_body(prepared, kwargs)
        client = self._sync_client()
        if self.api == "responses":
            return client.responses.create(**body)
        return client.chat.completions.create(**body)

    async def async_call(
        self, source: Any = None, /, *, image: Any = None, **kwargs: Any
    ):
        """Async counterpart of ``__call__``."""
        prepared = self._prepare(source, image)
        body = self._request_body(prepared, kwargs)
        client = self._async_client()
        if self.api == "responses":
            return await client.responses.create(**body)
        return await client.chat.completions.create(**body)

    def train(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(_UNSUPPORTED)

    def val(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(_UNSUPPORTED)

    def export(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(_UNSUPPORTED)

    def track(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(_UNSUPPORTED)

    def benchmark(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(_UNSUPPORTED)
