"""Generate the repo-safe ocr20 fixture (deterministic, self-owned).

Twenty procedural images of rendered text on varied backgrounds (solid,
gradient, noise, checkerboard), some rotated, plus a couple of deliberately
unreadable don't-care regions labelled ``###``. Text uses Pillow's embedded
default font, so there are no third-party image or font redistribution
rights involved. Run once to (re)generate ``images/val`` and
``labels/val.jsonl`` in the OCR dataset schema (docs/dataset_schema.md).

The strings are ASCII on purpose: the fixture exercises the pipeline,
dataset loader, validator, and goldens. Multi-script behaviour (CJK,
pinyin) is covered by the PaddleOCR parity gate, not by this fixture.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = Path(__file__).parent
IMAGES = HERE / "images" / "val"
LABELS = HERE / "labels"

WORDS = [
    "LibreYOLO",
    "OPEN 24 HOURS",
    "TOTAL 12.50",
    "EXIT",
    "PLATFORM 9",
    "NO PARKING",
    "HELLO WORLD",
    "COFFEE 3.20",
    "GATE B42",
    "SALE 50% OFF",
    "MAIN STREET 7",
    "TICKET 001984",
    "WARNING",
    "FRESH BREAD",
    "TAXI",
    "MUSEUM",
    "RECEIPT 2026-07-11",
    "ROOM 314",
    "CHECK IN",
    "FAST OCR",
]


def _font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.load_default(size=size)


def _backgrounds(rng: np.random.Generator, index: int, w: int, h: int) -> Image.Image:
    kind = index % 4
    if kind == 0:  # solid
        color = tuple(int(v) for v in rng.integers(160, 245, 3))
        return Image.new("RGB", (w, h), color)
    if kind == 1:  # horizontal gradient
        t = np.linspace(0, 1, w)[None, :, None]
        c0 = rng.integers(120, 200, 3)
        c1 = rng.integers(200, 250, 3)
        arr = (c0 * (1 - t) + c1 * t).repeat(h, 0).astype(np.uint8)
        return Image.fromarray(arr)
    if kind == 2:  # mild noise
        base = int(rng.integers(170, 230))
        arr = rng.integers(base - 18, base + 18, (h, w, 3)).astype(np.uint8)
        return Image.fromarray(arr)
    board = (np.indices((h, w)).sum(0) // 20 % 2)[..., None]
    arr = np.where(board == 0, 225, 195).repeat(3, 2).astype(np.uint8)
    return Image.fromarray(arr)


def _render_text_layer(text: str, font_size: int) -> Image.Image:
    font = _font(font_size)
    probe = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    left, top, right, bottom = probe.textbbox((0, 0), text, font=font)
    pad = max(2, font_size // 8)
    layer = Image.new("RGBA", (right - left + 2 * pad, bottom - top + 2 * pad), (0, 0, 0, 0))
    ImageDraw.Draw(layer).text((pad - left, pad - top), text, fill=(15, 15, 25, 255), font=font)
    return layer


def _paste_rotated(canvas: Image.Image, layer: Image.Image, x: int, y: int, angle: float):
    """Paste a rotated text layer; return the rotated quad (tl, tr, br, bl)."""
    rotated = layer.rotate(angle, expand=True, resample=Image.BICUBIC)
    canvas.alpha_composite(rotated, (x, y))
    lw, lh = layer.size
    theta = np.deg2rad(angle)
    cos, sin = np.cos(theta), np.sin(theta)
    cx, cy = lw / 2.0, lh / 2.0
    corners = np.array([[0, 0], [lw, 0], [lw, lh], [0, lh]], dtype=np.float64) - (cx, cy)
    # PIL rotates counter-clockwise; image y grows downward.
    rot = np.stack(
        [
            corners[:, 0] * cos + corners[:, 1] * sin,
            -corners[:, 0] * sin + corners[:, 1] * cos,
        ],
        axis=1,
    )
    rw, rh = rotated.size
    quad = rot + (x + rw / 2.0, y + rh / 2.0)
    return [[round(float(px), 1), round(float(py), 1)] for px, py in quad]


def gen() -> None:
    IMAGES.mkdir(parents=True, exist_ok=True)
    LABELS.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20)

    entries = []
    for index, primary in enumerate(WORDS):
        w = int(rng.integers(360, 560))
        h = int(rng.integers(220, 360))
        canvas = _backgrounds(rng, index, w, h).convert("RGBA")

        regions = []
        n_lines = 1 + index % 3
        y = int(rng.integers(20, 60))
        for line in range(n_lines):
            text = primary if line == 0 else WORDS[(index * 3 + line * 7) % len(WORDS)]
            font_size = int(rng.integers(22, 40))
            layer = _render_text_layer(text, font_size)
            angle = float(rng.uniform(-12, 12)) if (index + line) % 5 == 0 else 0.0
            x = int(rng.integers(12, max(13, w - layer.size[0] - 16)))
            quad = _paste_rotated(canvas, layer, x, y, angle)
            regions.append({"polygon": quad, "text": text})
            y += layer.size[1] + int(rng.integers(18, 44))
            if y > h - 60:
                break

        # Two images get an unreadable, blurred don't-care region.
        if index in (5, 13):
            blob = _render_text_layer("zzzzzzzz", 24).filter(ImageFilter.GaussianBlur(6))
            bx, by = int(w * 0.6), int(h * 0.7)
            canvas.alpha_composite(blob, (bx, by))
            lw, lh = blob.size
            regions.append(
                {
                    "polygon": [[bx, by], [bx + lw, by], [bx + lw, by + lh], [bx, by + lh]],
                    "text": "###",
                }
            )

        name = f"ocr_{index:02d}.png"
        canvas.convert("RGB").save(IMAGES / name)
        entries.append({"image": name, "regions": regions})

    with open(LABELS / "val.jsonl", "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"wrote {len(entries)} images to {IMAGES} and labels to {LABELS/'val.jsonl'}")


if __name__ == "__main__":
    gen()
