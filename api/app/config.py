"""Runtime settings, read from environment variables."""

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env_float(name: str, default: float) -> float:
    return float(os.getenv(name, default))


@dataclass(frozen=True)
class Settings:
    config_dir: Path = Path(os.getenv("CONFIG_DIR", Path(__file__).resolve().parents[2] / "config"))
    database_url: str | None = os.getenv("DATABASE_URL")
    llm_model: str = os.getenv("LLM_MODEL", "claude-opus-5-5")
    llm_effort: str = os.getenv("LLM_EFFORT", "low")
    tesseract_lang: str = os.getenv("TESSERACT_LANG", "fra")
    # Below these thresholds a document is sent to human review
    min_llm_confidence: float = _env_float("MIN_LLM_CONFIDENCE", 0.80)
    min_ocr_confidence: float = _env_float("MIN_OCR_CONFIDENCE", 60.0)
    max_upload_mb: float = _env_float("MAX_UPLOAD_MB", 15)
    extra: dict = field(default_factory=dict)


settings = Settings()
