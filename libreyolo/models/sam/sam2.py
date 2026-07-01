"""SAM-2 image promptable segmenter for the LibreSAM tier."""

from __future__ import annotations

from typing import ClassVar, Dict

import torch

from .base import _INSTALL_HINT, LibreSAMModel


class LibreSAM2(LibreSAMModel):
    """SAM-2.1 promptable segmenter (Hiera backbone), image inference only."""

    FAMILY = "sam2"
    FILENAME_PREFIX = "LibreSAM2"
    HF_REPOS: ClassVar[Dict[str, str]] = {
        "tiny": "facebook/sam2.1-hiera-tiny",
        "small": "facebook/sam2.1-hiera-small",
        "base-plus": "facebook/sam2.1-hiera-base-plus",
        "large": "facebook/sam2.1-hiera-large",
    }
    INPUT_SIZES: ClassVar[Dict[str, int]] = {size: 1024 for size in HF_REPOS}

    def _load_pretrained(self, snapshot_dir: str) -> tuple:
        try:
            from transformers import Sam2Model, Sam2Processor
        except ImportError as exc:
            raise ImportError(_INSTALL_HINT) from exc
        model = Sam2Model.from_pretrained(snapshot_dir, dtype=self._resolve_dtype())
        processor = Sam2Processor.from_pretrained(snapshot_dir)
        return model, processor

    def _move_embeddings(self, embeddings, device: torch.device):
        """SAM-2 caches a list of FPN feature tensors."""
        if isinstance(embeddings, list):
            return [emb.to(device) for emb in embeddings]
        return super()._move_embeddings(embeddings, device)

    def _post_process_masks(self, pred_masks, enc):
        """Upscale SAM-2 masks; its processor has no reshaped-size argument."""
        fn = getattr(self.processor, "post_process_masks", None)
        if fn is None:
            fn = self.processor.image_processor.post_process_masks
        return fn(pred_masks.cpu(), enc["original_sizes"])

    def track(self, *args, **kwargs):
        raise NotImplementedError(
            "LibreSAM2 image inference is implemented; SAM-2 video tracking "
            "is a follow-up for the video-memory model. Use predict() per "
            "frame for now."
        )


__all__ = ["LibreSAM2"]
