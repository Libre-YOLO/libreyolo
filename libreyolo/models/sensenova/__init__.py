"""SenseNova-Vision: one generative checkpoint serving many LibreYOLO tasks.

The user-facing entry point is the ``LibreVLM`` factory
(``LibreVLM("sensenova-vision", task=...)``) or the ``LibreSenseNovaVision``
class directly. See ``docs/adr/0012-sensenova-unified-multimodal.md`` for the
contract and ``libreyolo/models/sensenova/modeling`` for the vendored
architecture provenance.
"""

from .model import LibreSenseNovaVision

__all__ = ["LibreSenseNovaVision"]
