# SPDX-License-Identifier: MIT

"""LibreMODUS: external-weight, inference-only any-to-any analysis model."""

from __future__ import annotations

import logging
import types
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Mapping, Optional

import torch
from PIL import Image

from ...utils.general import COCO_CLASSES
from ...utils.image_loader import ImageInput, ImageLoader
from ..base.inference import InferenceRunner
from ..vlm.base import LibreVLMModel
from .decode import detection_payload, image_to_payload, input_to_image
from .modality import CodeCondition
from .prompts import (
    TASK_TO_TARGET,
    normalize_target,
    validate_any2any_request,
)
from .tokenizer import assert_checkpoint_vocabulary, build_modus_tokenizer
from .weights import HF_REPO, HF_REVISION, resolve_modus_snapshot

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .inference import ModusInferencer

_INSTALL_HINT = (
    "LibreMODUS requires the 'modus' extra. Install with:\n"
    "    pip install 'libreyolo[modus]'"
)


class _ModusInputs:
    def __init__(self, image: Image.Image, target: str, grounding_phrases=()):
        self.image = image
        self.target = target
        self.grounding_phrases = tuple(grounding_phrases)

    def to(self, _device):
        # Accelerate owns placement for this multi-device model.
        return self


def _as_device(value) -> torch.device:
    if isinstance(value, int):
        return torch.device(f"cuda:{value}")
    return torch.device(value)


def _mapped_device(device_map: Mapping[str, Any], module_name: str, fallback):
    """Resolve the most-specific accelerate device-map entry for a module."""
    matches = (
        (name, target)
        for name, target in device_map.items()
        if not name or module_name == name or module_name.startswith(f"{name}.")
    )
    return max(matches, key=lambda item: len(item[0]), default=("", fallback))[1]


class LibreMODUS(LibreVLMModel):
    """MODUS-14B-A7B behind the standard task API and ``any2any()``."""

    FAMILY = "libremodus"
    FILENAME_PREFIX = "LibreMODUS"
    HF_REPOS: ClassVar[dict[str, str]] = {"14b-a7b": HF_REPO}
    HF_REVISIONS: ClassVar[dict[str, str]] = {"14b-a7b": HF_REVISION}
    INPUT_SIZES: ClassVar[dict[str, int]] = {"14b-a7b": 1024}
    SUPPORTED_TASKS = ("detect", "depth", "normal", "edge")
    DEFAULT_TASK = "detect"
    SUPPORTS_BATCHED_PREDICT = False
    TTA_ENABLED = False
    CONFIDENCE_METHOD = "constrained_token_min"

    def __init__(
        self,
        size: str = "14b-a7b",
        *,
        checkpoint_path: Optional[str | Path] = None,
        token: Optional[str] = None,
        dtype: str = "bf16",
        max_memory_per_gpu: Optional[str] = None,
        inference_steps: int = 10,
        inference_cfg: float = 4.0,
        inference_image_cfg: float = 2.0,
        seed: int = 0,
        **kwargs,
    ):
        self._checkpoint_path = checkpoint_path
        self._hf_token = token
        self._requested_dtype = str(dtype).strip().lower()
        self._max_memory_per_gpu = max_memory_per_gpu
        self.inference_steps = int(inference_steps)
        self.inference_cfg = float(inference_cfg)
        self.inference_image_cfg = float(inference_image_cfg)
        self.seed = int(seed)
        self.inferencer: Optional["ModusInferencer"] = None
        self.tokenizer = None
        self._token_artifacts = None
        self._user_vocab = False
        super().__init__(size=size, **kwargs)
        if hasattr(self, "_resolved_snapshot"):
            self.model_path = str(self._resolved_snapshot)
        if self._custom_prompt and not self._user_vocab:
            self.set_classes([str(self._custom_prompt)])

    def set_classes(self, classes: list) -> "LibreMODUS":
        self._user_vocab = True
        super().set_classes(classes)
        return self

    # ------------------------------------------------------------------
    # External weights and big-model loading
    # ------------------------------------------------------------------

    def _ensure_weights(self) -> str:
        self._resolved_snapshot = resolve_modus_snapshot(
            checkpoint_path=self._checkpoint_path,
            token=self._hf_token,
        )
        return str(self._resolved_snapshot)

    def _precision(self) -> str:
        aliases = {
            "bf16": "bf16",
            "bfloat16": "bf16",
            "fp8": "fp8",
            "float8": "fp8",
            "e4m3": "fp8",
        }
        try:
            return aliases[self._requested_dtype]
        except KeyError as exc:
            raise ValueError("LibreMODUS dtype must be 'bf16' or 'fp8'.") from exc

    def _max_memory(self, precision: str) -> dict:
        if self.device.type != "cuda":
            return {"cpu": 128 * 1024**3}
        from accelerate.utils import convert_file_size_to_int

        memory = {}
        for index in range(torch.cuda.device_count()):
            if self._max_memory_per_gpu is not None:
                budget = convert_file_size_to_int(self._max_memory_per_gpu)
            else:
                # FP8 still needs VAE/KV/sampler headroom; BF16 uses the same
                # conservative fraction and spills through accelerate.
                budget = int(
                    torch.cuda.get_device_properties(index).total_memory * 0.82
                )
            memory[index] = budget
        memory["cpu"] = 128 * 1024**3
        return memory

    def _load_pretrained(self, snapshot_dir: str):
        try:
            from accelerate import (
                infer_auto_device_map,
                init_empty_weights,
                load_checkpoint_and_dispatch,
            )
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise ImportError(_INSTALL_HINT) from exc

        from ..sensenova.modeling.autoencoder import load_ae
        from ..sensenova.modeling.bagel import BagelConfig
        from ..sensenova.modeling.qwen2_navit import Qwen2Config, Qwen2ForCausalLM
        from ..sensenova.modeling.siglip_navit import (
            SiglipVisionConfig,
            SiglipVisionModel,
        )
        from ..sensenova.transforms import ImageTransform
        from .nn import ModusBagel
        from .inference import ModusInferencer
        from .quantize import (
            eligible_linear_names,
            prepare_fp8_checkpoint,
            replace_fp8_linears,
        )

        snapshot = Path(snapshot_dir)
        llm_config = Qwen2Config.from_json_file(str(snapshot / "llm_config.json"))
        llm_config.qk_norm = True
        llm_config.tie_word_embeddings = False
        llm_config.layer_module = "Qwen2MoTDecoderLayer"

        vit_config = SiglipVisionConfig.from_json_file(
            str(snapshot / "vit_config.json")
        )
        vit_config.rope = False
        vit_config.num_hidden_layers -= 1

        vae_model, vae_config = load_ae(str(snapshot / "ae.safetensors"))
        tokenizer = AutoTokenizer.from_pretrained(snapshot_dir, trust_remote_code=False)
        token_artifacts = build_modus_tokenizer(tokenizer)
        from safetensors import safe_open

        with safe_open(
            str(snapshot / "model.safetensors"), framework="pt", device="cpu"
        ) as checkpoint:
            checkpoint_vocab_size = int(
                checkpoint.get_slice(
                    "language_model.model.embed_tokens.weight"
                ).get_shape()[0]
            )
        assert_checkpoint_vocabulary(token_artifacts, checkpoint_vocab_size)
        if len(tokenizer) != checkpoint_vocab_size:
            raise RuntimeError(
                "MODUS tokenizer/checkpoint mismatch: "
                f"tokenizer has {len(tokenizer)} rows but the released embedding "
                f"has {checkpoint_vocab_size}."
            )
        # The released llm_config.json retains the pre-modality base-vocabulary
        # value (152064); the safetensor and tokenizer are the authoritative
        # 196840-row interface.
        llm_config.vocab_size = checkpoint_vocab_size

        config = BagelConfig(
            visual_gen=True,
            visual_und=True,
            llm_config=llm_config,
            vit_config=vit_config,
            vae_config=vae_config,
            latent_patch_size=2,
            max_latent_size=64,
            vit_max_num_patch_per_side=70,
            connector_act="gelu_pytorch_tanh",
        )
        # Parameters come from the safetensor, but several deterministic
        # rotary/cache buffers do not. Keep those buffers concrete on CPU so
        # Accelerate never has to copy a checkpoint-absent tensor off meta.
        with init_empty_weights(include_buffers=False):
            language_model = Qwen2ForCausalLM(llm_config)
            vit_model = SiglipVisionModel(vit_config)
            model = ModusBagel(
                language_model,
                vit_model,
                config,
                modality_registry=token_artifacts.modality_registry,
            )
            model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(
                vit_config, meta=True
            )

        precision = self._precision()
        checkpoint: str | Path = snapshot / "model.safetensors"
        load_dtype = torch.bfloat16
        if precision == "fp8":
            quantized_names = eligible_linear_names(model, llm_config.num_hidden_layers)
            replace_fp8_linears(model, quantized_names)
            checkpoint = prepare_fp8_checkpoint(
                checkpoint,
                quantized_names,
                source_revision=(
                    HF_REVISION if self._checkpoint_path is None else None
                ),
            )
            # The cache intentionally mixes FP8 buffers, FP16 scales, and BF16
            # exemptions. A global dtype cast would destroy that layout.
            load_dtype = None

        device_map = infer_auto_device_map(
            model,
            max_memory=self._max_memory(precision),
            no_split_module_classes=["Qwen2MoTDecoderLayer"],
            dtype=load_dtype,
        )
        shared_modules = (
            "language_model.model.embed_tokens",
            "time_embedder",
            "latent_pos_embed",
            "vae2llm",
            "llm2vae",
            "connector",
            "vit_pos_embed",
            "dino_pos_embed",
            "dinolocal_pos_embed",
            "clip_pos_embed",
            "imagebind_pos_embed",
            "imagebindlocal_pos_embed",
        )
        fallback_device = (
            self.device.index or 0 if self.device.type == "cuda" else str(self.device)
        )
        shared_device = _mapped_device(device_map, shared_modules[0], fallback_device)
        if str(shared_device) == "disk":
            # These small modules participate in every context update and
            # cannot be disk-offloaded independently of their inputs.
            shared_device = "cpu"
        for module_name in shared_modules:
            if (
                _mapped_device(device_map, module_name, fallback_device)
                != shared_device
            ):
                device_map[module_name] = shared_device
        model.materialize_static_position_embeddings(device_map)

        offload_dir = (
            Path.home()
            / ".cache"
            / "libreyolo"
            / "modus"
            / f"offload-{precision}-{HF_REVISION[:12]}"
        )
        offload_dir.mkdir(parents=True, exist_ok=True)
        model = load_checkpoint_and_dispatch(
            model,
            checkpoint=str(checkpoint),
            device_map=device_map,
            offload_folder=str(offload_dir),
            offload_buffers=True,
            dtype=load_dtype,
            force_hooks=True,
        ).eval()
        model.hf_device_map = device_map
        # BaseModel applies a whole-model .to() after _init_model. Dispatch
        # hooks already own placement, including all-GPU maps.
        model.to = types.MethodType(lambda module, *args, **kwargs: module, model)

        vae_device = _as_device(shared_device)
        vae_model = vae_model.to(device=vae_device, dtype=torch.bfloat16).eval()
        vae_transform = ImageTransform(1024, 512, 16)
        vit_transform = ImageTransform(980, 224, 14)

        self.tokenizer = tokenizer
        self._token_artifacts = token_artifacts
        self._vae_model = vae_model
        self._vae_transform = vae_transform
        self._vit_transform = vit_transform
        self.inferencer = ModusInferencer(
            model=model,
            vae_model=vae_model,
            tokenizer=tokenizer,
            vae_transform=vae_transform,
            vit_transform=vit_transform,
            token_artifacts=token_artifacts,
        )
        return model, None

    def _init_model(self):
        snapshot = self._ensure_weights()
        model, _ = self._load_pretrained(snapshot)
        self.processor = None
        self._model_dtype = torch.bfloat16
        return model

    # ------------------------------------------------------------------
    # Standard task API
    # ------------------------------------------------------------------

    def _standard_target(self) -> tuple[str, tuple[str, ...]]:
        target = TASK_TO_TARGET[self.task]
        if self.task != "detect":
            return target, ()
        if self._user_vocab:
            return "det", tuple(self.names[index] for index in range(len(self.names)))
        if self._custom_prompt:
            return "det", (str(self._custom_prompt),)
        return "cocodet", ()

    def _preprocess(
        self,
        image: ImageInput,
        color_format: str = "auto",
        input_size: Optional[int] = None,
    ):
        del input_size
        loaded = ImageLoader.load(image, color_format=color_format)
        target, phrases = self._standard_target()
        return _ModusInputs(loaded, target, phrases), loaded, loaded.size, 1.0

    def _forward(self, inputs: _ModusInputs) -> dict:
        conditions = [("rgb", inputs.image)]
        if inputs.target == "det":
            boxes = []
            for offset, phrase in enumerate(inputs.grounding_phrases):
                output = self.inferencer.run(
                    conditions,
                    target="det",
                    steps=self.inference_steps,
                    cfg=self.inference_cfg,
                    cfg_img=self.inference_image_cfg,
                    seed=self.seed + offset,
                    grounding_phrase=phrase,
                )
                boxes.extend(output["boxes"])
            return {"target": "det", "boxes": boxes}
        output = self.inferencer.run(
            conditions,
            target=inputs.target,
            steps=self.inference_steps,
            cfg=self.inference_cfg,
            cfg_img=self.inference_image_cfg,
            seed=self.seed,
        )
        output["target"] = inputs.target
        return output

    def _postprocess(
        self,
        output: dict,
        conf_thres: float,
        iou_thres: float,
        original_size: tuple[int, int],
        max_det: int = 300,
        ratio: float = 1.0,
        **kwargs,
    ) -> dict:
        del ratio
        target = output["target"]
        if target in {"depth", "normal", "canny", "samedge"}:
            return image_to_payload(output["image"], target, original_size)
        boxes = output.get("boxes", [])
        if target == "det":
            name_to_id = {
                name.strip().lower(): index for index, name in self.names.items()
            }
            boxes = [
                {
                    **item,
                    "label": name_to_id.get(str(item["label"]).strip().lower(), -1),
                }
                for item in boxes
            ]
        return detection_payload(
            boxes,
            original_size,
            conf=conf_thres,
            iou=iou_thres,
            max_det=max_det,
            classes=kwargs.get("classes"),
        )

    # ------------------------------------------------------------------
    # Any-to-any analysis surface
    # ------------------------------------------------------------------

    def _prepare_any2any_inputs(self, inputs: Mapping[str, Any]):
        if not isinstance(inputs, Mapping):
            raise TypeError("any2any() inputs must be a modality-to-value mapping.")
        normalized_names, _ = validate_any2any_request(inputs.keys(), "depth")
        conditions = []
        original_size = None
        for (raw_name, value), modality in zip(inputs.items(), normalized_names):
            if modality == "text":
                text = str(value).strip()
                if not text:
                    raise ValueError("The auxiliary text input cannot be empty.")
                conditions.append(("text", text))
                continue
            if isinstance(value, (str, Path)):
                value = ImageLoader.load(value, color_format="auto")
            image = input_to_image(value, modality)
            if original_size is None:
                original_size = image.size
            elif image.size != original_size:
                raise ValueError(
                    "any2any() image-derived inputs must share one aligned canvas; "
                    f"{raw_name!r} is {image.size}, expected {original_size}."
                )
            conditions.append((modality, image))
        return conditions, original_size

    @staticmethod
    def _candidate_condition(target: str, raw: dict, phrase: Optional[str]):
        if target in {"depth", "normal", "canny", "samedge"}:
            return target, raw["image"]
        code = CodeCondition(
            modality=target,
            text=raw["token_text"],
            prefix=phrase if target == "det" else None,
        )
        return target, code

    def _raw_to_payload(
        self,
        raw: dict,
        target: str,
        original_size: tuple[int, int],
    ) -> dict:
        if target in {"depth", "normal", "canny", "samedge"}:
            return image_to_payload(raw["image"], target, original_size)
        boxes = raw["boxes"]
        if target == "det":
            boxes = [{**item, "label": 0} for item in boxes]
        return detection_payload(boxes, original_size, conf=0.0, iou=0.45)

    def any2any(
        self,
        inputs: Mapping[str, Any],
        target: str,
        *,
        steps: int = 10,
        cfg: float = 2.0,
        seed: int = 0,
        chain=(),
        verify: int = 0,
    ):
        """Run the documented image-derived some-to-some analysis subset."""
        normalized_inputs, normalized_target = validate_any2any_request(
            inputs.keys(), target
        )
        del normalized_inputs
        conditions, original_size = self._prepare_any2any_inputs(inputs)
        if isinstance(chain, (str, bytes)):
            raise TypeError("chain must be a sequence of target names, not a string.")
        chain_targets = tuple(normalize_target(item) for item in chain)
        condition_modalities = sum(name != "text" for name, _ in conditions)
        if condition_modalities + len(chain_targets) > 3:
            raise ValueError(
                "Inputs plus chained intermediates exceed MODUS's trained "
                "three-modality condition budget (auxiliary text is separate)."
            )
        if verify not in (0,) and (not isinstance(verify, int) or verify < 2):
            raise ValueError("verify must be 0 (off) or an integer >= 2.")

        text_values = [str(value) for name, value in conditions if name == "text"]
        phrase = text_values[-1] if text_values else None
        if normalized_target == "det":
            if phrase is None:
                if self._user_vocab and len(self.names) == 1:
                    phrase = self.names[0]
                else:
                    raise ValueError(
                        "grounding needs inputs={'text': '<phrase>'} or exactly "
                        "one class configured with set_classes()."
                    )
            # The trained grounding instruction already contains the phrase.
            conditions = [(name, value) for name, value in conditions if name != "text"]

        for index, intermediate in enumerate(chain_targets):
            intermediate_phrase = phrase if intermediate == "det" else None
            intermediate_conditions = (
                [(name, value) for name, value in conditions if name != "text"]
                if intermediate == "det"
                else conditions
            )
            raw = self.inferencer.run(
                intermediate_conditions,
                target=intermediate,
                steps=steps,
                cfg=cfg,
                seed=seed + index,
                grounding_phrase=intermediate_phrase,
            )
            conditions.append(
                self._candidate_condition(intermediate, raw, intermediate_phrase)
            )

        candidate_count = verify if verify else 1
        candidates = []
        for candidate_index in range(candidate_count):
            raw = self.inferencer.run(
                conditions,
                target=normalized_target,
                steps=steps,
                cfg=cfg,
                seed=seed + len(chain_targets) + candidate_index,
                grounding_phrase=phrase if normalized_target == "det" else None,
            )
            if verify:
                score = self.inferencer.verification_score(
                    conditions,
                    candidate=self._candidate_condition(normalized_target, raw, phrase),
                    target=normalized_target,
                )
            else:
                score = 0.0
            candidates.append((score, raw))
        verification_score, raw = max(candidates, key=lambda item: item[0])

        payload = self._raw_to_payload(
            raw,
            normalized_target,
            original_size,
        )
        previous_names = self.names
        previous_nb_classes = self.nb_classes
        if normalized_target == "det":
            self.names = {0: phrase}
            self.nb_classes = 1
        elif normalized_target == "cocodet":
            self.names = {index: name for index, name in enumerate(COCO_CLASSES)}
            self.nb_classes = len(self.names)
        try:
            result = InferenceRunner(self)._wrap_results(
                payload, original_size, image_path=None, classes=None
            )
        finally:
            self.names = previous_names
            self.nb_classes = previous_nb_classes
        if verify:
            result.verification_score = verification_score
            result.verification_candidates = candidate_count
        return result

    # ------------------------------------------------------------------
    # Unsupported mutation/export surfaces
    # ------------------------------------------------------------------

    def chat(self, *args, **kwargs):
        raise NotImplementedError(
            "LibreMODUS exposes analysis tasks and any2any(), not free-form chat."
        )

    def train(self, *args, **kwargs):
        raise NotImplementedError("LibreMODUS is inference-only.")

    def val(self, *args, **kwargs):
        raise NotImplementedError(
            "End-to-end LibreMODUS validation requires the gated/manual benchmark "
            "protocol documented in docs/testing.md."
        )

    def export(self, format: str = "onnx", **kwargs):
        raise NotImplementedError(
            "LibreMODUS's MoT KV-cache and flow sampler do not export. "
            "Use the edge specialists for ONNX deployment."
        )
