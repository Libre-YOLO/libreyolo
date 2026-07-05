"""Parser for Darknet ``.cfg`` model-definition files.

Darknet models are defined by their ``.cfg`` file: an ordered list of ``[type]``
sections with ``key=value`` options. LibreYOLO bundles the public-domain cfgs
(see ``cfgs/``) and builds the runtime graph from them, so the architecture
definition has a single source of truth shared by the model and the weight
converter.

The ``.cfg`` format is public domain (Darknet "YOLO LICENSE"). This parser reads
the format; it copies no Darknet source.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


_CFG_DIR = Path(__file__).resolve().parent / "cfgs"


@dataclass
class CfgLayer:
    """One ``[section]`` from a cfg file: its type plus parsed options."""

    type: str
    options: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.options.get(key, default)

    def get_int(self, key: str, default: int = 0) -> int:
        val = self.options.get(key)
        return int(val) if val is not None else default

    def get_float(self, key: str, default: float = 0.0) -> float:
        val = self.options.get(key)
        return float(val) if val is not None else default

    def get_int_list(self, key: str) -> list[int]:
        val = self.options.get(key)
        if val is None:
            return []
        return [int(x.strip()) for x in str(val).split(",") if x.strip() != ""]

    def get_float_list(self, key: str) -> list[float]:
        val = self.options.get(key)
        if val is None:
            return []
        return [float(x.strip()) for x in str(val).split(",") if x.strip() != ""]


def parse_cfg_text(text: str) -> tuple[dict[str, Any], list[CfgLayer]]:
    """Parse cfg text into ``(net_options, layers)``.

    ``net_options`` is the ``[net]`` block (input width/height/channels, etc.).
    ``layers`` are every subsequent section in file order. Comment lines
    (starting with ``#``) and blank lines are ignored.
    """
    net_options: dict[str, Any] = {}
    layers: list[CfgLayer] = []
    current: CfgLayer | None = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].strip()
            if section == "net":
                current = None  # [net] options collect into net_options
            else:
                current = CfgLayer(type=section)
                layers.append(current)
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if current is None:
            # inside [net] (or before any layer — treat as net options)
            net_options[key] = value
        else:
            current.options[key] = value

    return net_options, layers


def parse_cfg_file(path: str | Path) -> tuple[dict[str, Any], list[CfgLayer]]:
    """Parse a cfg file from disk."""
    text = Path(path).read_text(encoding="utf-8")
    return parse_cfg_text(text)


def bundled_cfg_path(name: str) -> Path:
    """Resolve a bundled cfg by name (with or without the ``.cfg`` suffix)."""
    if not name.endswith(".cfg"):
        name = f"{name}.cfg"
    path = _CFG_DIR / name
    if not path.exists():
        available = ", ".join(sorted(p.name for p in _CFG_DIR.glob("*.cfg")))
        raise FileNotFoundError(
            f"Bundled Darknet cfg not found: {name}. Available: {available}"
        )
    return path


def load_bundled_cfg(name: str) -> tuple[dict[str, Any], list[CfgLayer]]:
    """Parse a bundled cfg by name (e.g. ``"yolov3"`` or ``"yolov3-tiny"``)."""
    return parse_cfg_file(bundled_cfg_path(name))
