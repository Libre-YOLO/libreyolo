"""LibreMobileSAM: native MobileSAM on the LibreSAM promptable tier."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar, Dict

import torch
import torch.nn as nn

from ..sam.base import (
    _INSTALL_HINT,
    _SNAPSHOT_COMPLETE_MARKER,
    LibreSAMModel,
)
from ...utils.serialization import (
    load_untrusted_torch_file,
    unwrap_libreyolo_checkpoint,
)
from .mask_decoder import MaskDecoder
from .nn import TinyViT
from .preprocess import encode_image_and_prompts, postprocess_masks, preprocess_tensor
from .prompt_encoder import PromptEncoder
from .transformer import TwoWayTransformer

logger = logging.getLogger(__name__)


class MobileSAMNetwork(nn.Module):
    """TinyViT image encoder plus SAM prompt encoder and mask decoder."""

    mask_threshold: float = 0.0
    image_size: int = 1024

    def __init__(self) -> None:
        super().__init__()
        prompt_embed_dim = 256
        image_embedding_size = self.image_size // 16
        self.image_encoder = TinyViT(
            img_size=self.image_size,
            in_chans=3,
            num_classes=1000,
            embed_dims=[64, 128, 160, 320],
            depths=[2, 2, 6, 2],
            num_heads=[2, 4, 5, 10],
            window_sizes=[7, 7, 14, 7],
            mlp_ratio=4.0,
            drop_rate=0.0,
            drop_path_rate=0.0,
            use_checkpoint=False,
            mbconv_expand_ratio=4.0,
            local_conv_size=3,
            layer_lr_decay=0.8,
        )
        self.prompt_encoder = PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(self.image_size, self.image_size),
            mask_in_chans=16,
        )
        self.mask_decoder = MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        )
        self.register_buffer(
            "pixel_mean",
            torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "pixel_std",
            torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1),
            persistent=False,
        )

    @torch.no_grad()
    def get_image_embeddings(self, pixel_values: torch.Tensor) -> torch.Tensor:
        pixel_values = preprocess_tensor(
            pixel_values,
            image_size=self.image_size,
            pixel_mean=self.pixel_mean.to(pixel_values.device, pixel_values.dtype),
            pixel_std=self.pixel_std.to(pixel_values.device, pixel_values.dtype),
        )
        return self.image_encoder(pixel_values)

    def forward(
        self,
        *,
        pixel_values: torch.Tensor | None = None,
        image_embeddings: torch.Tensor | None = None,
        input_points: torch.Tensor | None = None,
        input_labels: torch.Tensor | None = None,
        input_boxes: torch.Tensor | None = None,
        multimask_output: bool = True,
    ):
        if not ((pixel_values is None) ^ (image_embeddings is None)):
            raise ValueError(
                "Exactly one of pixel_values or image_embeddings is required."
            )
        if image_embeddings is None:
            image_embeddings = self.get_image_embeddings(pixel_values)

        points = None
        if input_points is not None:
            if input_labels is None:
                input_labels = torch.ones_like(input_points[..., 0], dtype=torch.long)
            points = (input_points[0], input_labels[0])

        boxes = input_boxes[0] if input_boxes is not None else None
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=points,
            boxes=boxes,
            masks=None,
        )
        low_res_masks, iou_scores = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
        )
        return SimpleNamespace(
            pred_masks=low_res_masks.unsqueeze(0),
            iou_scores=iou_scores.unsqueeze(0),
            image_embeddings=image_embeddings,
        )


class LibreMobileSAM(LibreSAMModel):
    """MobileSAM: TinyViT image encoder plus native SAM decoder."""

    FAMILY = "mobilesam"
    FILENAME_PREFIX = "LibreMobileSAM"
    HF_REPOS: ClassVar[Dict[str, str]] = {"tiny": "LibreYOLO/LibreMobileSAM"}
    INPUT_SIZES: ClassVar[Dict[str, int]] = {"tiny": 1024}
    WEIGHT_FILE: ClassVar[str] = "LibreMobileSAM.pt"
    SNAPSHOT_IGNORE_PATTERNS: ClassVar[tuple[str, ...]] = (
        "*.bin",
        "*.onnx",
        "onnx/*",
        "*.tflite",
        "*.msgpack",
        "*.h5",
    )

    def __init__(self, size: str = "tiny", **kwargs):
        super().__init__(size=size, **kwargs)

    @staticmethod
    def _snapshot_complete(local_dir: Path) -> bool:
        return (local_dir / _SNAPSHOT_COMPLETE_MARKER).exists() and (
            local_dir / LibreMobileSAM.WEIGHT_FILE
        ).exists()

    def _ensure_weights(self) -> str:
        repo = self.HF_REPOS[self.size]
        local_dir = Path("weights") / f"{self.FILENAME_PREFIX}{self.size}"
        if self._snapshot_complete(local_dir):
            return str(local_dir)
        try:
            from huggingface_hub import snapshot_download
            import transformers  # noqa: F401
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
                f"Downloaded snapshot for {repo} is missing {self.WEIGHT_FILE} "
                f"in {local_dir}."
            )
        return str(local_dir)

    def _init_model(self) -> nn.Module:
        snapshot_dir = Path(self._ensure_weights())
        model = MobileSAMNetwork()
        checkpoint = load_untrusted_torch_file(
            snapshot_dir / self.WEIGHT_FILE,
            map_location="cpu",
            context="LibreMobileSAM hosted checkpoint",
        )
        state_dict, metadata = unwrap_libreyolo_checkpoint(checkpoint, strict=True)
        expected = {
            "model_family": self.FAMILY,
            "size": self.size,
            "task": self.DEFAULT_TASK,
        }
        mismatches = {
            key: (metadata.get(key), value)
            for key, value in expected.items()
            if metadata.get(key) != value
        }
        if mismatches:
            details = ", ".join(
                f"{key}={got!r} (expected {expected!r})"
                for key, (got, expected) in mismatches.items()
            )
            raise ValueError(f"Invalid LibreMobileSAM checkpoint metadata: {details}")
        model.load_state_dict(state_dict, strict=True)
        self.processor = None
        self._model_dtype = next(model.parameters()).dtype
        return model

    def _encode(self, img, *, points=None, labels=None, boxes=None):
        return encode_image_and_prompts(
            img,
            target_length=self.INPUT_SIZES[self.size],
            points=points,
            labels=labels,
            boxes=boxes,
        )

    def _post_process_masks(self, pred_masks, enc):
        input_size = tuple(int(v) for v in enc["reshaped_input_sizes"][0])
        original_size = tuple(int(v) for v in enc["original_sizes"][0])
        masks = postprocess_masks(
            pred_masks[0].cpu(),
            image_size=self.INPUT_SIZES[self.size],
            input_size=input_size,
            original_size=original_size,
            mask_threshold=self.model.mask_threshold,
        )
        return [masks]


__all__ = ["LibreMobileSAM", "MobileSAMNetwork"]
