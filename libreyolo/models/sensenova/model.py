"""LibreYOLO wrapper for SenseNova-Vision, a unified multimodal vision model.

SenseNova-Vision (SenseTime, Apache-2.0 code) casts vision tasks as prompted
generation on a Bagel-MoT backbone: symbolic outputs (boxes, points,
keypoints, OCR words) are generated as tagged text, and dense outputs (depth
maps, segmentation masks, panoptic color maps) are generated as images that a
frozen VAE decodes. One checkpoint therefore serves many LibreYOLO tasks; this
family is the first LibreVLM member whose ``task=`` spans the dense tasks.

    from libreyolo import LibreVLM
    model = LibreVLM("sensenova-vision", task="depth")
    result = model.predict("image.jpg")          # -> Results.depth_map
    model.set_task("detect").set_classes(["bird", "boat"])
    result = model.predict("image.jpg")          # -> Results.boxes

The model is heavy (14.7B parameters, ~29.6 GB in bf16) and runs one diffusion
decode per dense prediction; it is a capability model, not a real-time one.
``dtype="auto"`` picks bf16 on large GPUs and 4-bit NF4 quantization (via
bitsandbytes) on consumer GPUs.

Weights are CC BY-NC 4.0 (non-commercial) and download at runtime from the
byte-identical LibreYOLO mirror of the upstream release, after a one-time
license notice.
"""

from __future__ import annotations

import logging
import random
from typing import Any, ClassVar, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from ...utils.image_loader import ImageInput, ImageLoader
from ..vlm.base import LibreVLMModel
from ..vlm.parsing import build_detection_dict
from . import parsing
from .prompts import (
    BBOX_DETECTION_PROMPT_TEMPLATE,
    CAPTION_QUESTION_TEMPLATE,
    COCO_PANOPTIC_NAMES,
    DEPTH_PROMPT,
    HUMAN_KEYPOINT_CATEGORIES,
    KEYPOINT_PROMPT_TEMPLATES,
    MASK_QUESTION_TEMPLATE,
    OCR_PROMPT,
    POINT_DETECTION_PROMPT_TEMPLATE,
)

logger = logging.getLogger(__name__)

_INSTALL_HINT = (
    "LibreSenseNovaVision requires the 'sensenova' extra. Install with:\n"
    "    pip install 'libreyolo[sensenova]'"
)

# Per-mode generation parameters, ported from inference/sensenova_vision.py
# BASE_PARAMS (Apache-2.0). The released checkpoint was tuned against these
# exact schedules; recon3d is intentionally not ported in v1.
BASE_PARAMS: Dict[str, Dict[str, Any]] = {
    "generate": dict(
        cfg_text_scale=4.0,
        cfg_img_scale=1.0,
        cfg_interval=[0.4, 1.0],
        timestep_shift=3.0,
        num_timesteps=50,
        cfg_renorm_min=1.0,
        cfg_renorm_type="global",
    ),
    "think_generate": dict(
        max_think_token_n=1000,
        do_sample=False,
        cfg_text_scale=4.0,
        cfg_img_scale=1.0,
        cfg_interval=[0.4, 1.0],
        timestep_shift=3.0,
        num_timesteps=50,
        cfg_renorm_min=1.0,
        cfg_renorm_type="global",
        think=True,
    ),
    "caption_generate": dict(
        max_think_token_n=8192,
        do_sample=False,
        cfg_text_scale=4.0,
        cfg_img_scale=1.0,
        cfg_interval=[0.0, 1.0],
        timestep_shift=4.0,
        num_timesteps=50,
        cfg_renorm_min=1.0,
        cfg_renorm_type="global",
        caption=True,
    ),
    "dense_perception": dict(
        cfg_text_scale=4.0,
        cfg_img_scale=1.0,
        cfg_interval=[0.0, 1.0],
        timestep_shift=4.0,
        num_timesteps=50,
        cfg_renorm_min=1.0,
        cfg_renorm_type="text_channel",
    ),
    "edit": dict(
        cfg_text_scale=4.0,
        cfg_img_scale=2.0,
        cfg_interval=[0.0, 1.0],
        timestep_shift=4.0,
        num_timesteps=50,
        cfg_renorm_min=1.0,
        cfg_renorm_type="text_channel",
    ),
    "think_edit": dict(
        max_think_token_n=1000,
        do_sample=False,
        cfg_text_scale=4.0,
        cfg_img_scale=2.0,
        cfg_interval=[0.4, 1.0],
        timestep_shift=3.0,
        num_timesteps=50,
        cfg_renorm_min=0.0,
        cfg_renorm_type="text_channel",
        think=True,
    ),
    "understanding": dict(
        max_think_token_n=8192,
        do_sample=False,
        understanding_output=True,
    ),
    "think_understanding": dict(
        max_think_token_n=8192,
        do_sample=False,
        understanding_output=True,
        think=True,
    ),
    "dense_detection": dict(
        max_think_token_n=8192,
        do_sample=False,
        understanding_output=True,
    ),
    "dense_OCR": dict(
        max_think_token_n=20000,
        do_sample=False,
        understanding_output=True,
    ),
}

IMAGE_OUTPUT_MODES = {
    "generate",
    "think_generate",
    "caption_generate",
    "dense_perception",
    "edit",
    "think_edit",
}

# LibreYOLO task -> (inference mode, needs a class vocabulary in the prompt)
_TASK_TO_MODE = {
    "detect": "dense_detection",
    "point": "dense_detection",
    "pose": "dense_detection",
    "ocr": "dense_OCR",
    "depth": "dense_perception",
    "segment": "dense_perception",
    "panoptic": "caption_generate",
}


class _SenseNovaInputs:
    """Preprocessed inputs for one predict call.

    The shared InferenceRunner calls ``inputs.to(device)`` before forwarding;
    device placement is owned by the accelerate-dispatched model, so ``to`` is
    a no-op.
    """

    def __init__(self, image: Image.Image, prompt: str, mode: str, params: Dict[str, Any]):
        self.image = image
        self.prompt = prompt
        self.mode = mode
        self.params = params

    def to(self, _device):
        return self


def _set_seed(seed: Optional[int]) -> None:
    """Seed diffusion noise for reproducible dense generations."""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class LibreSenseNovaVision(LibreVLMModel):
    """SenseNova-Vision as a multi-task LibreVLM family."""

    FAMILY = "sensenovavision"
    FILENAME_PREFIX = "SenseNovaVision"

    HF_REPOS: ClassVar[Dict[str, str]] = {
        # Byte-identical mirror of sensenova/SenseNova-Vision-7B-MoT at
        # revision 79548fcc (SHA-256 pins in the repo card and family NOTICE).
        "7b": "LibreYOLO/SenseNovaVision7b",
    }
    # Weight-only repo (no remote code); the pin still keeps downloads
    # reproducible across releases.
    HF_REVISIONS: ClassVar[Dict[str, str]] = {}
    INPUT_SIZES: ClassVar[Dict[str, int]] = {
        "7b": 1024,
    }

    SUPPORTED_TASKS = ("detect", "segment", "panoptic", "pose", "point", "depth", "ocr")
    DEFAULT_TASK = "detect"

    # The upstream repo also ships showcase images; only weights/configs are
    # needed to run.
    SNAPSHOT_IGNORE_PATTERNS: ClassVar[tuple] = LibreVLMModel.SNAPSHOT_IGNORE_PATTERNS + (
        "assets/*",
    )

    _LICENSE_NOTICE = (
        "\n"
        "----------------------------------------------------------------\n"
        "SenseNova-Vision-7B-MoT weights are provided by SenseTime under\n"
        "CC BY-NC 4.0: NON-COMMERCIAL use only. LibreYOLO mirrors the\n"
        "unmodified weights with attribution; the license is unchanged and\n"
        "you are responsible for complying with its terms.\n"
        "  Mirror:   https://huggingface.co/LibreYOLO/SenseNovaVision7b\n"
        "  Upstream: https://huggingface.co/sensenova/SenseNova-Vision-7B-MoT\n"
        "----------------------------------------------------------------\n"
    )

    def __init__(
        self,
        size: str = "7b",
        *,
        dtype: str = "auto",
        max_memory_per_gpu: Optional[str] = None,
        noise_seed: Optional[int] = 42,
        **kwargs,
    ):
        self._requested_dtype = str(dtype).lower()
        self._max_memory_per_gpu = max_memory_per_gpu
        self.noise_seed = noise_seed
        self._user_vocab = False
        # Filled by _init_model.
        self.inferencer = None
        self.tokenizer = None
        super().__init__(size=size, **kwargs)

    # =========================================================================
    # Task and vocabulary surface
    # =========================================================================

    def set_classes(self, classes: list) -> "LibreSenseNovaVision":
        self._user_vocab = True
        return super().set_classes(classes)

    def _task_names(self) -> Dict[int, str]:
        """The effective vocabulary for the active task.

        Unless the user set one, panoptic falls back to the COCO panoptic
        categories the model was tuned on, and pose falls back to ``person``.
        """
        if self._user_vocab:
            return self.names
        if self.task == "panoptic":
            return {i: name for i, name in enumerate(COCO_PANOPTIC_NAMES)}
        if self.task == "pose":
            return {0: "person"}
        return self.names

    def _require_user_vocab(self) -> None:
        if not self._user_vocab:
            raise ValueError(
                "Referring segmentation needs a target phrase. Call "
                'set_classes(["<phrase>"]) first, e.g. '
                'set_classes(["person furthest to the right"]).'
            )

    # =========================================================================
    # Prompts
    # =========================================================================

    @staticmethod
    def _format_categories(names: Dict[int, str]) -> str:
        return ", ".join(f"<p>{names[i]}</p>" for i in range(len(names)))

    def _keypoint_prompt(self, names: Dict[int, str]) -> str:
        categories = self._format_categories(names)
        labels = [str(v).strip().lower() for v in names.values()]
        has_human = any(label in HUMAN_KEYPOINT_CATEGORIES for label in labels)
        has_animal = any(label not in HUMAN_KEYPOINT_CATEGORIES for label in labels)
        if has_human and has_animal:
            kind = "mixed"
        elif has_human:
            kind = "human"
        else:
            kind = "animal"
        return KEYPOINT_PROMPT_TEMPLATES[kind].format(categories=categories)

    def _task_prompt(self) -> str:
        """Build the exact instruction the checkpoint expects for the task."""
        if self._custom_prompt:
            return self._custom_prompt
        names = self._task_names()
        if self.task == "detect":
            return BBOX_DETECTION_PROMPT_TEMPLATE.format(
                categories=self._format_categories(names)
            )
        if self.task == "point":
            return POINT_DETECTION_PROMPT_TEMPLATE.format(
                categories=self._format_categories(names)
            )
        if self.task == "pose":
            return self._keypoint_prompt(names)
        if self.task == "ocr":
            return OCR_PROMPT
        if self.task == "depth":
            return DEPTH_PROMPT
        if self.task == "segment":
            self._require_user_vocab()
            return MASK_QUESTION_TEMPLATE.format(
                categories=self._format_categories(names), task_type="binary"
            )
        if self.task == "panoptic":
            category_prompt = MASK_QUESTION_TEMPLATE.format(
                categories=self._format_categories(names), task_type="panoptic"
            )
            caption_prompt = CAPTION_QUESTION_TEMPLATE.format(task_type="panoptic")
            return f"{category_prompt} {caption_prompt}"
        raise ValueError(f"Unsupported task: {self.task!r}")

    # =========================================================================
    # Loading
    # =========================================================================

    def _select_dtype(self) -> str:
        if self._requested_dtype != "auto":
            return self._requested_dtype
        if self.device.type != "cuda":
            # The checkpoint is stored in bf16; fp32 would double memory.
            return "bf16"
        total = torch.cuda.get_device_properties(self.device.index or 0).total_memory
        if total >= 35 * 1024**3:
            return "bf16"
        try:
            import bitsandbytes  # noqa: F401

            logger.info(
                "LibreSenseNovaVision: GPU has %.0f GiB; loading NF4-quantized "
                "(pass dtype='bf16' to override).",
                total / 1024**3,
            )
            return "nf4"
        except ImportError as exc:
            raise ImportError(
                "SenseNova-Vision needs ~30 GiB of VRAM in bf16, and this GPU "
                f"has {total / 1024**3:.0f} GiB. Install bitsandbytes for 4-bit "
                "loading (pip install bitsandbytes) or pass dtype='bf16' with a "
                "larger GPU."
            ) from exc

    def _max_memory(self, plan_scale: float = 1.0) -> Dict[Any, Any]:
        """Per-device budgets for the dispatch planner, in bytes.

        The planner sizes modules at their checkpoint (bf16) width, but a
        quantized load shrinks them ~3.5x on the GPU, so quantized paths pass
        ``plan_scale`` > 1 to let the planner place modules that will fit
        after quantization.
        """
        if self.device.type != "cuda":
            return {"cpu": 128 * 1024**3}
        max_memory = {}
        for i in range(torch.cuda.device_count()):
            if self._max_memory_per_gpu is not None:
                from accelerate.utils import convert_file_size_to_int

                budget = convert_file_size_to_int(self._max_memory_per_gpu)
            else:
                # Leave headroom for activations and the VAE decode.
                budget = int(torch.cuda.get_device_properties(i).total_memory * 0.9)
            max_memory[i] = int(budget * plan_scale)
        return max_memory

    def _load_pretrained(self, snapshot_dir: str):
        """Build the Bagel skeleton and load the checkpoint through accelerate."""
        try:
            from accelerate import (
                infer_auto_device_map,
                init_empty_weights,
                load_checkpoint_and_dispatch,
            )
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise ImportError(_INSTALL_HINT) from exc

        import os

        import json

        from .modeling.autoencoder import load_ae
        from .modeling.bagel import Bagel, BagelConfig
        from .modeling.qwen2_navit import Qwen2Config, Qwen2ForCausalLM
        from .modeling.siglip_navit import SiglipVisionConfig, SiglipVisionModel
        from .data_utils import CheckpointTokenizer, add_special_tokens
        from .inferencer import InterleaveInferencer
        from .transforms import ImageTransform

        llm_config = Qwen2Config.from_json_file(os.path.join(snapshot_dir, "llm_config.json"))
        llm_config.qk_norm = True
        llm_config.tie_word_embeddings = False
        llm_config.layer_module = "Qwen2MoTDecoderLayer"

        vit_config = SiglipVisionConfig.from_json_file(
            os.path.join(snapshot_dir, "vit_config.json")
        )
        vit_config.rope = False
        vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

        vae_model, vae_config = load_ae(
            local_path=os.path.join(snapshot_dir, "ae.safetensors")
        )

        # The released tokenizer.json disagrees with the trained added-token
        # layout recorded in tokenizer_config.json; CheckpointTokenizer
        # overlays the trained layout (see its docstring).
        base_tokenizer = AutoTokenizer.from_pretrained(snapshot_dir)
        with open(
            os.path.join(snapshot_dir, "tokenizer_config.json"), encoding="utf-8"
        ) as f:
            added_tokens = json.load(f).get("added_tokens_decoder", {})
        tokenizer = CheckpointTokenizer(
            base_tokenizer,
            {int(i): t["content"] for i, t in added_tokens.items()},
        )
        tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)
        if max(new_token_ids.values()) >= llm_config.vocab_size:
            raise RuntimeError(
                "Special token ids exceed the checkpoint vocabulary "
                f"({new_token_ids}); the tokenizer files are inconsistent "
                "with the model weights."
            )

        bagel_config = BagelConfig(
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

        with init_empty_weights():
            language_model = Qwen2ForCausalLM(llm_config)
            vit_model = SiglipVisionModel(vit_config)
            model = Bagel(language_model, vit_model, bagel_config)
            model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(
                vit_config, meta=True
            )

        selected_dtype = self._select_dtype()
        # NF4 stores 4-bit weights plus norms/embeddings in bf16 (~3.5x
        # smaller than the bf16 checkpoint the planner measures); int8 is ~1.8x.
        plan_scale = {"nf4": 3.5, "int8": 1.8}.get(selected_dtype, 1.0)
        device_map = infer_auto_device_map(
            model,
            max_memory=self._max_memory(plan_scale),
            no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
        )
        # Keep the shared embedding/connector modules on one device.
        same_device_modules = [
            "language_model.model.embed_tokens",
            "time_embedder",
            "latent_pos_embed",
            "vae2llm",
            "llm2vae",
            "connector",
            "vit_pos_embed",
        ]
        first_device = device_map.get(
            same_device_modules[0],
            str(self.device) if self.device.type == "cuda" else "cpu",
        )
        for key in same_device_modules:
            device_map[key] = device_map.get(key, first_device)

        checkpoint_path = os.path.join(snapshot_dir, "ema.safetensors")
        # Last-resort spill target when a small machine cannot hold the model
        # across GPU and CPU memory.
        offload_folder = os.path.join(snapshot_dir, "offload")

        if selected_dtype in ("nf4", "int8"):
            from accelerate.utils import BnbQuantizationConfig, load_and_quantize_model

            if selected_dtype == "nf4":
                quant_config = BnbQuantizationConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=False,
                    bnb_4bit_quant_type="nf4",
                )
            else:
                quant_config = BnbQuantizationConfig(
                    load_in_8bit=True, torch_dtype=torch.float32
                )
            model = load_and_quantize_model(
                model,
                weights_location=checkpoint_path,
                bnb_quantization_config=quant_config,
                device_map=device_map,
                offload_folder=offload_folder,
            ).eval()
        else:
            torch_dtype = {
                "bf16": torch.bfloat16,
                "bfloat16": torch.bfloat16,
                "fp16": torch.float16,
                "float16": torch.float16,
                "fp32": torch.float32,
                "float32": torch.float32,
            }.get(selected_dtype)
            if torch_dtype is None:
                raise ValueError(
                    f"Unsupported dtype {selected_dtype!r}. Use auto, bf16, fp16, "
                    "fp32, nf4 or int8."
                )
            model = load_checkpoint_and_dispatch(
                model,
                checkpoint=checkpoint_path,
                device_map=device_map,
                offload_folder=offload_folder,
                offload_buffers=True,
                dtype=torch_dtype,
                force_hooks=True,
            ).eval()

        # Record the dispatch layout; the inferencer (and transformers) use it
        # to skip whole-model device moves. When part of the checkpoint is
        # offloaded to CPU, a later ``.to(device)`` (BaseModel moves the model
        # after init) would break the accelerate hooks, so neutralize it.
        model.hf_device_map = device_map
        if any(str(target) in ("cpu", "disk") for target in device_map.values()):
            import types

            model.to = types.MethodType(lambda module, *args, **kwargs: module, model)

        vae_device = str(self.device)
        vae_model = vae_model.to(device=vae_device).eval()

        self.tokenizer = tokenizer
        self._new_token_ids = new_token_ids
        self._vae_model = vae_model
        self._vae_transform = ImageTransform(1024, 512, 16)
        self._vit_transform = ImageTransform(980, 224, 14)
        self.inferencer = InterleaveInferencer(
            model=model,
            vae_model=vae_model,
            tokenizer=tokenizer,
            vae_transform=self._vae_transform,
            vit_transform=self._vit_transform,
            new_token_ids=new_token_ids,
            device=vae_device,
        )
        # No HF processor for this family; the inferencer owns preprocessing.
        return model, None

    def _init_model(self):
        snapshot_dir = self._ensure_weights()
        model, _ = self._load_pretrained(snapshot_dir)
        self.processor = None
        self._model_dtype = next(
            (p.dtype for p in model.parameters() if p.is_floating_point()),
            torch.bfloat16,
        )
        return model

    # =========================================================================
    # Raw escape hatches
    # =========================================================================

    def generate(
        self,
        prompt: str,
        images: Optional[List[ImageInput]] = None,
        mode: str = "understanding",
        max_new_tokens: Optional[int] = None,
        noise_seed: Optional[int] = None,
        **overrides,
    ):
        """Run one raw SenseNova-Vision request: images plus prompt in, text or
        image out.

        ``mode`` selects the upstream inference schedule (understanding,
        dense_perception, dense_detection, dense_OCR, caption_generate,
        generate, edit, and their think variants). Image-output modes return a
        ``PIL.Image``; text modes return ``str``; modes that produce both
        return ``{"image": ..., "text": ...}``.
        """
        if self.inferencer is None:
            raise RuntimeError("Model is not loaded.")
        if mode not in BASE_PARAMS:
            raise ValueError(
                f"Invalid mode {mode!r}. Supported modes: {sorted(BASE_PARAMS)}"
            )
        params = dict(BASE_PARAMS[mode])
        params.update(overrides)
        if max_new_tokens is not None:
            params["max_think_token_n"] = max_new_tokens
        think = params.pop("think", False)
        understanding_output = params.pop("understanding_output", False)

        input_lists: List[Any] = []
        for image in images or []:
            input_lists.append(ImageLoader.load(image, color_format="auto"))
        text = str(prompt).strip()
        if text:
            input_lists.append(text)
        if not input_lists:
            raise ValueError("generate() needs a prompt and/or at least one image.")

        _set_seed(self.noise_seed if noise_seed is None else noise_seed)
        with torch.no_grad():
            outputs = self.inferencer.interleave_inference(
                input_lists,
                think=think,
                understanding_output=understanding_output,
                **params,
            )

        images_out = [item for item in outputs if isinstance(item, Image.Image)]
        texts_out = [item for item in outputs if isinstance(item, str)]
        if mode in IMAGE_OUTPUT_MODES and images_out and not texts_out:
            return images_out[0]
        if not images_out and texts_out:
            return texts_out[0]
        return {
            "image": images_out[0] if images_out else None,
            "text": texts_out[0] if texts_out else None,
        }

    def chat(
        self,
        image: ImageInput,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        color_format: str = "auto",
    ) -> str:
        """Free-form visual question answering on one image."""
        img = ImageLoader.load(image, color_format=color_format)
        result = self.generate(
            prompt, images=[img], mode="understanding", max_new_tokens=max_new_tokens
        )
        return result if isinstance(result, str) else str(result.get("text") or "")

    # =========================================================================
    # InferenceRunner hooks
    # =========================================================================

    def _preprocess(
        self,
        image: ImageInput,
        color_format: str = "auto",
        input_size: Optional[int] = None,
    ) -> Tuple[Any, Any, Tuple[int, int], float]:
        img = ImageLoader.load(image, color_format=color_format)
        mode = _TASK_TO_MODE[self.task]
        params = dict(BASE_PARAMS[mode])
        inputs = _SenseNovaInputs(img, self._task_prompt(), mode, params)
        return inputs, img, img.size, 1.0

    def _forward(self, inputs: _SenseNovaInputs) -> Dict[str, Any]:
        params = dict(inputs.params)
        think = params.pop("think", False)
        understanding_output = params.pop("understanding_output", False)
        _set_seed(self.noise_seed)
        outputs = self.inferencer.interleave_inference(
            [inputs.image, inputs.prompt],
            think=think,
            understanding_output=understanding_output,
            **params,
        )
        result = {"image": None, "text": None}
        for item in outputs:
            if isinstance(item, Image.Image) and result["image"] is None:
                result["image"] = item
            elif isinstance(item, str) and result["text"] is None:
                result["text"] = item
        return result

    def _postprocess(
        self,
        output: Dict[str, Any],
        conf_thres: float,
        iou_thres: float,
        original_size: Tuple[int, int],
        max_det: int = 300,
        ratio: float = 1.0,
        **kwargs,
    ) -> Dict:
        text = output.get("text") or ""
        image = output.get("image")
        classes = kwargs.get("classes")

        if self.task == "detect":
            return self._postprocess_detect(text, conf_thres, iou_thres, original_size, max_det, classes)
        if self.task == "point":
            return self._postprocess_point(text, conf_thres, original_size, max_det, classes)
        if self.task == "pose":
            return self._postprocess_pose(text, conf_thres, original_size, max_det, classes)
        if self.task == "ocr":
            return self._postprocess_ocr(text, original_size, max_det)
        if self.task == "depth":
            return self._postprocess_depth(image, original_size)
        if self.task == "segment":
            return self._postprocess_segment(image, conf_thres, original_size)
        if self.task == "panoptic":
            return self._postprocess_panoptic(image, text, original_size)
        raise ValueError(f"Unsupported task: {self.task!r}")

    # ---- symbolic tasks ----

    def _postprocess_detect(self, text, conf_thres, iou_thres, original_size, max_det, classes):
        parsed = parsing.parse_tagged_boxes(text)
        items = [
            {"label": label, "bbox": box}
            for label, boxes in parsed.items()
            for box in boxes
        ]
        name_to_id = {
            parsing.normalize_category(v): k for k, v in self._task_names().items()
        }
        return build_detection_dict(
            items,
            name_to_id,
            original_size,
            conf_thres=conf_thres,
            max_det=max_det,
            classes=classes,
            default_score=self._score_detections(items),
            bbox_key="bbox",
            coord_divisor=1.0,
            box_format="xyxy",
            iou_thres=iou_thres,
        )

    def _postprocess_point(self, text, conf_thres, original_size, max_det, classes):
        width, height = original_size
        parsed = parsing.parse_tagged_points(text)
        name_to_id = {
            parsing.normalize_category(v): k for k, v in self._task_names().items()
        }
        allowed = set(classes) if classes is not None else None
        score = self._score_detections(parsed)
        rows: List[List[float]] = []
        seen = set()
        if score >= conf_thres and max_det > 0:
            for label, points in parsed.items():
                class_id = name_to_id.get(label)
                if class_id is None:
                    continue
                if allowed is not None and class_id not in allowed:
                    continue
                for x, y in points:
                    key = (class_id, round(x, 3), round(y, 3))
                    if key in seen:
                        continue
                    seen.add(key)
                    rows.append([x * width, y * height, float(class_id), score])
                    if len(rows) >= max_det:
                        break
                if len(rows) >= max_det:
                    break
        return {"points": rows, "num_detections": len(rows)}

    def _postprocess_pose(self, text, conf_thres, original_size, max_det, classes):
        width, height = original_size
        parsed = parsing.parse_tagged_keypoints(text)
        names = self._task_names()
        name_to_id = {parsing.normalize_category(v): k for k, v in names.items()}
        allowed = set(classes) if classes is not None else None
        score = self._score_detections(parsed)

        boxes, scores, class_ids, keypoints = [], [], [], []
        if score >= conf_thres and max_det > 0:
            for label, instances in parsed.items():
                class_id = name_to_id.get(label)
                if class_id is None:
                    continue
                if allowed is not None and class_id not in allowed:
                    continue
                skeleton = (
                    parsing.HUMAN_KEYPOINT_NAMES
                    if label in HUMAN_KEYPOINT_CATEGORIES
                    else parsing.ANIMAL_KEYPOINT_NAMES
                )
                for instance in instances:
                    kpts = parsing.keypoint_instance_to_array(
                        instance, skeleton, width, height
                    )
                    bbox = instance.get("bbox")
                    if bbox is not None:
                        x0, y0, x1, y1 = bbox
                        box = [x0 * width, y0 * height, x1 * width, y1 * height]
                    else:
                        visible = kpts[kpts[:, 2] > 0][:, :2]
                        if len(visible) == 0:
                            continue
                        box = [
                            float(visible[:, 0].min()),
                            float(visible[:, 1].min()),
                            float(visible[:, 0].max()),
                            float(visible[:, 1].max()),
                        ]
                    boxes.append(box)
                    scores.append(score)
                    class_ids.append(class_id)
                    keypoints.append(kpts)
                    if len(boxes) >= max_det:
                        break
                if len(boxes) >= max_det:
                    break

        detections = {
            "boxes": boxes,
            "scores": scores,
            "classes": class_ids,
            "num_detections": len(boxes),
        }
        if keypoints:
            detections["keypoints"] = np.stack(keypoints, axis=0)
        return detections

    def _postprocess_ocr(self, text, original_size, max_det):
        width, height = original_size
        parsed = parsing.parse_tagged_boxes(text, normalize_labels=False)
        polygons, texts = [], []
        for word, boxes in parsed.items():
            for x0, y0, x1, y1 in boxes:
                if len(polygons) >= max_det:
                    break
                polygons.append(
                    [
                        [x0 * width, y0 * height],
                        [x1 * width, y0 * height],
                        [x1 * width, y1 * height],
                        [x0 * width, y1 * height],
                    ]
                )
                texts.append(word)
        return {
            "ocr": {
                "polygons": polygons,
                "texts": texts,
                "confidences": [self.DEFAULT_SCORE] * len(texts),
                "det_confidences": [self.DEFAULT_SCORE] * len(texts),
            }
        }

    # ---- dense tasks ----

    def _postprocess_depth(self, image, original_size):
        width, height = original_size
        if image is None:
            depth = np.zeros((height, width), dtype=np.float32)
        else:
            depth = parsing.depth_from_image(image, original_size)
        return {"depth": depth}

    def _postprocess_segment(self, image, conf_thres, original_size):
        empty = {"boxes": [], "scores": [], "classes": [], "num_detections": 0}
        if image is None or self.DEFAULT_SCORE < conf_thres:
            return empty
        mask = parsing.binary_mask_from_image(image, original_size)
        box = parsing.mask_to_xyxy(mask)
        if box is None:
            return empty
        return {
            "boxes": [box],
            "scores": [self.DEFAULT_SCORE],
            "classes": [0],
            "num_detections": 1,
            "masks": mask[None, :, :].astype(np.uint8),
        }

    def _postprocess_panoptic(self, image, text, original_size):
        width, height = original_size
        if image is None:
            return {
                "panoptic": np.zeros((height, width), dtype=np.int64),
                "segments_info": [],
            }
        id_map, segments_info, names = parsing.panoptic_from_mask_and_caption(
            image, text, self._task_names(), original_size
        )
        # Registered open-vocabulary phrases extend the label table so
        # Results can name every returned segment.
        self.names = names
        return {"panoptic": id_map, "segments_info": segments_info}
