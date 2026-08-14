# ADR 0020: Remote transport for LibreVLM

Status: accepted
Date: 2026-08-15

## Context

`LibreVLM` (ADR 0002) runs local vision-language models as open-vocabulary
detectors. `LibreLLM` (ADR 0019) is an OpenAI-compatible chat client. Hosted
vision chat models (OpenAI, OpenRouter, any vLLM/Ollama-style compat host)
can also emit boxes when prompted, which makes them useful as an explore /
auto-label ramp before training a real detector.

The question was where hosted detection lives: a third factory
(`LibreAPI`), a chat method that sometimes returns boxes, or a new
transport behind the existing detector contract.

## Decision

Factories follow contract, not where the weights live. Detect vs chat is
the product; local vs API is a backend. Two factories, no third:

- `LibreVLM` gains an OpenAI-compatible remote transport. Same contract as
  local: sticky `set_classes`, `predict()` returning pixel-xyxy `Results`
  on images/lists/folders/video files, `chat(image, prompt) -> str` as the
  raw escape hatch, soft confidence 1.0.
- `LibreLLM` stays the chat client (native SDK objects, streaming, async).
  It never grows `predict()`/`set_classes()`; `LibreVLM.chat()` never grows
  streaming or SDK returns.

### Routing grammar

A slash means remote; a bare alias stays local; no local alias will ever
contain a slash. Evaluate in order: existing path (checkpoint), path-shaped
string (never a provider; `FileNotFoundError` with the resolved absolute
path), first-slash provider split (before lowercasing, so case-sensitive
model ids survive), bare alias table. The unknown-provider error names the
known providers, the `openai-compat/` + `base_url=` escape, the local
aliases (HF repo ids are not accepted), and the resolved path that does not
exist, because those are the three things a failing string could have meant.

v1 providers: `openai` (SDK default host, `OPENAI_API_KEY`), `openrouter`
(`https://openrouter.ai/api/v1`, `OPENROUTER_API_KEY`), `openai-compat`
(required `base_url=`). The model id after the first slash passes through
verbatim: a model that ships tomorrow works today; the provider's 404 is
the miss signal, not our alias table.

### Wire API defaults (deliberate asymmetry)

| | default `api` | why |
|---|---|---|
| `LibreVLM` remote | `chat.completions` | the format every compat host speaks |
| `LibreLLM` | `responses` | the ecosystem chat-client shape we copied |

`LibreVLM` also requires the `openai-compat/` prefix for self-hosted
endpoints where `LibreLLM` keys off `base_url=` alone: `LibreVLM` has a
closed local alias table to disambiguate against; `LibreLLM` does not.

### An empty result is never ambiguous

Per-image failure never aborts a folder, and every non-clean empty carries
`result.remote = {"error": "http" | "parse" | "refusal", "detail", "model"}`.
Auth / bad-model / bad-request errors raise loudly (they would fail on every
image; converting them to empties would silently zero a run). Multi-image
runs log a failure summary. A chatty prose answer without boxes counts as a
parse failure: the model broke the format contract, and the caller deserves
to know before trusting it as a negative.

### Cost safety (v1 scope; cost accounting is not)

- Live sources (webcam, network streams, screen capture) raise. Every frame
  is a metered call; there is no opt-in kwarg in v1.
- Multi-image runs log a one-line metered banner (count, provider, model)
  before the first request.
- `batch=` means request concurrency (thread pool over per-image HTTP,
  default 8), never a stacked tensor.
- `estimate()` / `budget=` / price tables stay out of v1.

### Detection prompt and parsing

One generic prompt (the existing base `_format_detection_prompt`: JSON
array of `{"label", "bbox"}` normalized to [0, 1]), no per-vendor dialect
registry. `parsing.py` is reused unchanged; Qwen-lineage `bbox_2d` 0-1000
answers are rescaled by heuristic (values above ~1). `prompt=` overrides,
same as local.

### selftest()

`model.selftest()` sends two metered probes (a red rectangle at a known
position, then a blank image) and reports pass/fail with IoU and
false-positive count. It is the honest answer to "does this hosted model
ground at all" and makes no mAP claim.

### Install

Remote `LibreVLM` requires `libreyolo[llm]` (the OpenAI SDK), not
`libreyolo[vlm]`: a machine that only does hosted detect must not pull
transformers. Constructing without the extra raises the pip hint; importing
libreyolo never imports openai.

## Out of scope (named so they stay decisions)

Native Gemini/Anthropic/Moondream transports; `estimate()`/`budget=`;
`autolabel()` and provider batch JSONL; CLI; realtime APIs; SDK objects
from `chat()`; remote models inside `LibreYOLO(...)`; LiteLLM or vendor CV
SDK dependencies; callable `api_key`.

## Consequences

- The explore-with-hosted-model, train-local workflow is one import.
- Boxes from hosted chat models are uncalibrated (confidence 1.0, provider
  resizing, prompt adherence varies); docs steer calibrated use to
  `LibreYOLO` and gate big runs behind `selftest()`.
- Remote sends images to a third party; documented, no policy registry in
  v1.
