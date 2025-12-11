from __future__ import annotations

import io
import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont


@dataclass
class DetectedObject:
    label: str
    confidence: float
    box: Tuple[float, float, float, float]  # xmin, ymin, xmax, ymax in relative coordinates

    def as_absolute_box(self, image: Image.Image) -> Tuple[int, int, int, int]:
        width, height = image.size
        xmin, ymin, xmax, ymax = self.box
        return (
            int(max(0.0, xmin) * width),
            int(max(0.0, ymin) * height),
            int(min(1.0, xmax) * width),
            int(min(1.0, ymax) * height),
        )

    def to_dict(self) -> dict:
        data = asdict(self)
        data["box"] = list(self.box)
        return data


PALETTE = [
    "#EF4444",
    "#F97316",
    "#EAB308",
    "#22C55E",
    "#3B82F6",
    "#A855F7",
    "#EC4899",
]


def draw_boxes(image: Image.Image, objects: Sequence[DetectedObject]) -> Image.Image:
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    try:
        font = ImageFont.load_default()
    except IOError:  # pragma: no cover - PIL always provides a default font
        font = None

    for idx, obj in enumerate(objects):
        color = PALETTE[idx % len(PALETTE)]
        box = obj.as_absolute_box(annotated)
        draw.rectangle(box, outline=color, width=3)
        label = f"{obj.label} ({obj.confidence:.2f})"
        if font:
            text_size = draw.textbbox((0, 0), label, font=font)
            text_bg = (box[0], box[1] - (text_size[3] - text_size[1]) - 4)
            text_bg_rect = (*text_bg, text_bg[0] + (text_size[2] - text_size[0]) + 8, box[1])
            draw.rectangle(text_bg_rect, fill=color)
            draw.text((text_bg_rect[0] + 4, text_bg_rect[1] + 2), label, fill="white", font=font)
    return annotated


def image_to_bytes(image: Image.Image, fmt: str = "PNG") -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return buffer.getvalue()


def write_artifact(data: bytes, suffix: str, artifact_dir: Path) -> Path:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=artifact_dir) as tmp:
        tmp.write(data)
        return Path(tmp.name)


def detections_to_json(objects: Iterable[DetectedObject]) -> str:
    payload: List[dict] = [obj.to_dict() for obj in objects]
    return json.dumps(payload, ensure_ascii=False, indent=2)
