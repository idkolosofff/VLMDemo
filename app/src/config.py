from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings


class Limits(BaseModel):
    """Collection of tunable limits for uploads and generation."""

    max_image_mb: int = Field(10, description="Maximum allowed upload size in megabytes")
    max_image_pixels: int = Field(
        4096 * 4096, description="Maximum total number of pixels permitted for an image"
    )


class Settings(BaseSettings):
    model_name: str = Field("OpenGVLab/InternVL3_5-1B", alias="MODEL_NAME")
    device: Literal["cpu", "cuda", "auto"] = Field("auto", alias="DEVICE")
    max_tokens: int = Field(256, alias="MAX_TOKENS")
    port: int = Field(7860, alias="PORT")
    hf_cache_dir: Path = Field(Path("./data/model_cache"), alias="HF_CACHE_DIR")
    artifact_dir: Path = Field(Path("./model_artifacts"), alias="ARTIFACT_DIR")
    object_score_threshold: float = Field(0.25, alias="OBJECT_SCORE_THRESHOLD")
    share_ui: bool = Field(False, alias="SHARE_UI")
    limits: Limits = Field(default_factory=Limits)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    @property
    def resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            if torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        except Exception:  # pragma: no cover - defensive fallback
            return "cpu"

    @validator("hf_cache_dir", "artifact_dir", pre=True)
    def _expand_path(cls, value: Path | str) -> Path:  # noqa: D417 - pydantic signature
        path = Path(value).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
