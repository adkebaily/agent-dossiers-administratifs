"""End-to-end processing of one uploaded document: preprocess -> OCR -> LLM -> rules."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from . import llm
from .config import settings
from .ocr import run_ocr
from .preprocessing import PreprocessOptions, load_pages, preprocess
from .validation import check_document


@dataclass
class ProcessedDocument:
    filename: str
    document_type: str
    type_confidence: float
    fields: dict[str, dict]
    ocr_text: str
    ocr_confidence: float
    skew_angle: float
    preprocessing_steps: list[str]
    checks: dict
    needs_review: bool
    review_reasons: list[str] = field(default_factory=list)
    llm_notes: str = ""
    processing_ms: int = 0


def review_reasons(extraction: llm.Extraction, ocr_conf: float, checks) -> list[str]:
    reasons = []
    if extraction.status != "ok":
        reasons.append(f"LLM : {extraction.notes}")
    if extraction.document_type == "autre":
        reasons.append("Type de document non reconnu")
    if extraction.type_confidence < settings.min_llm_confidence:
        reasons.append(f"Classification incertaine ({extraction.type_confidence:.2f})")
    low = [f.name for f in extraction.fields if f.value and f.confidence < settings.min_llm_confidence]
    if low:
        reasons.append(f"Champs incertains : {', '.join(low)}")
    if ocr_conf < settings.min_ocr_confidence:
        reasons.append(f"Qualité OCR faible ({ocr_conf:.0f}/100)")
    if checks.missing_fields:
        reasons.append(f"Champs obligatoires manquants : {', '.join(checks.missing_fields)}")
    reasons.extend(checks.errors)
    return reasons


def process_document(
    data: bytes, filename: str, opts: PreprocessOptions | None = None
) -> ProcessedDocument:
    start = time.perf_counter()
    pages = load_pages(data, filename)

    texts, confs, angles, steps = [], [], [], []
    for page in pages:
        pre = preprocess(page, opts)
        ocr = run_ocr(pre.image)
        texts.append(ocr.text)
        confs.append(ocr.mean_confidence)
        angles.append(pre.skew_angle)
        steps = pre.steps
    ocr_text = "\n\n--- page ---\n\n".join(texts)
    ocr_conf = sum(confs) / len(confs)

    extraction = llm.extract(ocr_text)
    checks = check_document(extraction.document_type, extraction.field_dict())
    reasons = review_reasons(extraction, ocr_conf, checks)

    return ProcessedDocument(
        filename=filename,
        document_type=extraction.document_type,
        type_confidence=extraction.type_confidence,
        fields={f.name: {"value": f.value, "confidence": f.confidence} for f in extraction.fields},
        ocr_text=ocr_text,
        ocr_confidence=ocr_conf,
        skew_angle=angles[0],
        preprocessing_steps=steps,
        checks={"missing_fields": checks.missing_fields, "errors": checks.errors, "warnings": checks.warnings},
        needs_review=bool(reasons),
        review_reasons=reasons,
        llm_notes=extraction.notes,
        processing_ms=int((time.perf_counter() - start) * 1000),
    )
