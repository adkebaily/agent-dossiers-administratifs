"""End-to-end API test with the LLM mocked (no API key needed). Requires Tesseract."""

import shutil

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.app import llm
from api.app.main import app

pytestmark = pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract not installed")


def fake_extract(fields: dict, doc_type: str, conf: float = 0.95):
    def _extract(ocr_text: str) -> llm.Extraction:
        return llm.Extraction(
            document_type=doc_type, type_confidence=conf, language="fr", notes="",
            fields=[llm.ExtractedField(name=k, value=v, confidence=conf) for k, v in fields.items()],
        )
    return _extract


def png_bytes() -> bytes:
    img = np.full((600, 900), 255, np.uint8)
    cv2.putText(img, "RELEVE D'IDENTITE BANCAIRE", (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 0, 2)
    return cv2.imencode(".png", img)[1].tobytes()


def upload(client, dossier_id="D-1"):
    return client.post(
        "/documents",
        files={"file": ("rib.png", png_bytes(), "image/png")},
        data={"dossier_id": dossier_id, "dossier_type": "caf_aide_logement"},
    )


def test_confident_document_is_validated(monkeypatch):
    rib = {"titulaire": "Camille MARTIN", "iban": "FR7630006000011234567890189", "bic": "AGRIFRPP881"}
    monkeypatch.setattr(llm, "extract", fake_extract(rib, "rib"))
    monkeypatch.setattr("api.app.pipeline.settings", type("S", (), {"min_llm_confidence": 0.8, "min_ocr_confidence": 0})())
    r = upload(TestClient(app)).json()
    assert r["document_type"] == "rib"
    assert r["needs_review"] is False
    assert "rib" in r["dossier"]["present"]
    assert "piece_identite" in r["dossier"]["missing"]


def test_low_confidence_goes_to_review_then_approved(monkeypatch):
    rib = {"titulaire": "Camille MARTIN", "iban": "FR7630006000011234567890188", "bic": "AGRIFRPP881"}
    monkeypatch.setattr(llm, "extract", fake_extract(rib, "rib", conf=0.5))
    client = TestClient(app)
    r = upload(client, "D-2").json()
    assert r["needs_review"] is True
    assert any("IBAN invalide" in reason for reason in r["review_reasons"])

    review = client.post(f"/documents/{r['document_id']}/review",
                         json={"decision": "approve", "corrected_fields": {"iban": "FR7630006000011234567890189"}})
    assert review.json()["status"] == "validated"


def test_unknown_dossier_type_rejected():
    r = TestClient(app).post("/documents", files={"file": ("a.png", png_bytes(), "image/png")},
                             data={"dossier_id": "x", "dossier_type": "nope"})
    assert r.status_code == 422
