"""OCR with Tesseract (French model)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytesseract

from .config import settings


@dataclass
class OcrResult:
    text: str
    mean_confidence: float  # 0-100, averaged over recognised words
    word_count: int


def run_ocr(image: np.ndarray, lang: str | None = None) -> OcrResult:
    lang = lang or settings.tesseract_lang
    # psm 3: fully automatic page segmentation (documents with blocks and tables)
    config = "--oem 1 --psm 3"
    data = pytesseract.image_to_data(
        image, lang=lang, config=config, output_type=pytesseract.Output.DICT
    )
    confs = [float(c) for c, w in zip(data["conf"], data["text"]) if w.strip() and float(c) >= 0]
    text = pytesseract.image_to_string(image, lang=lang, config=config)
    return OcrResult(
        text=text.strip(),
        mean_confidence=float(np.mean(confs)) if confs else 0.0,
        word_count=len(confs),
    )
