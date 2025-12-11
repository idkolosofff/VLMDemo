from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from app.src.config import Limits


class ValidationError(ValueError):
    """Domain-specific exception used to display friendly error messages."""


def ensure_pil_image(value: Any, limits: Limits) -> Image.Image:
    """Validate and normalize raw inputs to a RGB PIL image."""

    if value is None:
        raise ValidationError("Пожалуйста, загрузите изображение.")

    if isinstance(value, Image.Image):
        image = value
    elif isinstance(value, (bytes, bytearray)):
        image = Image.open(io.BytesIO(value))
    elif isinstance(value, io.IOBase):
        image = Image.open(value)
    elif isinstance(value, np.ndarray):
        image = Image.fromarray(value)
    else:
        # Gradio may pass a dict with {"name": path}
        possible_path = getattr(value, "name", None) or value
        if isinstance(possible_path, (str, Path)) and Path(possible_path).exists():
            image = Image.open(possible_path)
        else:
            raise ValidationError("Поддерживаются только файлы изображений (PNG, JPG, JPEG).")

    image = image.convert("RGB")
    _ensure_image_limits(image, limits)
    return image


def ensure_text(value: Any, field_name: str, min_chars: int = 1) -> str:
    if value is None:
        raise ValidationError(f"Поле '{field_name}' не может быть пустым.")
    text = str(value).strip()
    if len(text) < min_chars:
        raise ValidationError(f"Введите текст (минимум {min_chars} символ).")
    return text


def _ensure_image_limits(image: Image.Image, limits: Limits) -> None:
    width, height = image.size
    if width * height > limits.max_image_pixels:
        raise ValidationError(
            "Изображение слишком большое. Попробуйте уменьшить разрешение и повторить попытку."
        )
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    size_mb = len(buffer.getvalue()) / (1024 * 1024)
    if size_mb > limits.max_image_mb:
        raise ValidationError(
            f"Файл изображения превышает допустимый размер {limits.max_image_mb} МБ."
        )
