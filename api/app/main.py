"""FastAPI service called by the n8n workflow."""

from __future__ import annotations

import secrets
from dataclasses import asdict
from typing import Literal

import anthropic
import cv2
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import RedirectResponse, Response
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

from .catalog import dossier_config
from .config import settings
from .db import store
from .pipeline import process_document
from .preprocessing import load_pages, preprocess
from .validation import check_dossier

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(request: Request, key: str | None = Depends(_api_key_header)) -> None:
    if settings.api_key is None or request.url.path in ("/", "/health"):
        return
    if key is None or not secrets.compare_digest(key, settings.api_key):
        raise HTTPException(401, "Clé API manquante ou invalide (en-tête X-API-Key)")


app = FastAPI(
    title="Agent dossiers administratifs",
    version="0.1.0",
    dependencies=[Depends(require_api_key)],
)


async def _read_upload(file: UploadFile) -> bytes:
    data = await file.read()
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, f"Fichier trop volumineux (> {settings.max_upload_mb} Mo)")
    if not data:
        raise HTTPException(400, "Fichier vide")
    return data


def _flat_fields(doc: dict) -> dict:
    return {k: (v or {}).get("value") for k, v in doc["fields"].items()}


def _dossier_status(dossier_id: str) -> dict:
    dossier = store.get_dossier(dossier_id)
    if dossier is None:
        raise HTTPException(404, "Dossier inconnu")
    docs = store.list_documents(dossier_id)
    result = check_dossier(
        dossier["dossier_type"],
        [{"document_type": d["document_type"], "fields": _flat_fields(d), "status": d["status"]} for d in docs],
    )
    return {
        "dossier_id": dossier_id,
        **asdict(result),
        "documents": [
            {"id": d["id"], "filename": d["filename"], "document_type": d["document_type"], "status": d["status"]}
            for d in docs
        ],
    }


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse("/docs")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": settings.llm_model}


@app.get("/dossier-types")
def dossier_types() -> dict:
    return {k: v["label"] for k, v in dossier_config()["dossiers"].items()}


@app.post("/documents")
async def upload_document(
    file: UploadFile = File(...),
    dossier_id: str = Form(...),
    dossier_type: str = Form(...),
) -> dict:
    if dossier_type not in dossier_config()["dossiers"]:
        raise HTTPException(422, f"Type de dossier inconnu : {dossier_type}")
    data = await _read_upload(file)
    try:
        result = process_document(data, file.filename or "upload")
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except anthropic.AuthenticationError as exc:
        raise HTTPException(502, "Clé API Anthropic invalide : vérifiez ANTHROPIC_API_KEY dans .env") from exc
    except anthropic.PermissionDeniedError as exc:
        raise HTTPException(502, f"Accès refusé par l'API Anthropic : {exc.message}") from exc
    except anthropic.RateLimitError as exc:
        raise HTTPException(503, "Limite de requêtes Anthropic atteinte, réessayez dans une minute") from exc
    except anthropic.APIStatusError as exc:
        raise HTTPException(502, f"Erreur de l'API Anthropic ({exc.status_code}) : {exc.message}") from exc
    except anthropic.APIConnectionError as exc:
        cause = exc.__cause__ or exc
        raise HTTPException(
            502, f"Impossible de joindre l'API Anthropic : {type(cause).__name__}: {cause}"
        ) from exc
    except anthropic.AnthropicError as exc:
        # Raised at client creation when no API key is configured
        raise HTTPException(500, f"Configuration Anthropic : {exc}") from exc

    store.upsert_dossier(dossier_id, dossier_type)
    doc_id = store.insert_document(
        {
            "dossier_id": dossier_id,
            "filename": result.filename,
            "document_type": result.document_type,
            "type_confidence": result.type_confidence,
            "fields": result.fields,
            "ocr_confidence": result.ocr_confidence,
            "skew_angle": result.skew_angle,
            "checks": result.checks,
            "status": "pending_review" if result.needs_review else "validated",
            "review_reasons": result.review_reasons,
            "processing_ms": result.processing_ms,
        }
    )
    out = asdict(result)
    out.pop("ocr_text")
    return {"document_id": doc_id, **out, "dossier": _dossier_status(dossier_id)}


class Review(BaseModel):
    decision: Literal["approve", "reject"]
    corrected_fields: dict[str, str | None] | None = None
    comment: str | None = None


@app.post("/documents/{doc_id}/review")
def review_document(doc_id: int, review: Review) -> dict:
    doc = store.get_document(doc_id)
    if doc is None:
        raise HTTPException(404, "Document inconnu")
    fields = None
    if review.corrected_fields:
        fields = dict(doc["fields"])
        for name, value in review.corrected_fields.items():
            fields[name] = {"value": value, "confidence": 1.0, "corrected": True}
    status = "validated" if review.decision == "approve" else "rejected"
    store.review_document(doc_id, status, fields, review.comment)
    return {"document_id": doc_id, "status": status, "dossier": _dossier_status(doc["dossier_id"])}


@app.get("/dossiers/{dossier_id}")
def get_dossier(dossier_id: str) -> dict:
    return _dossier_status(dossier_id)


@app.post("/preprocess", responses={200: {"content": {"image/png": {}}}})
async def preview_preprocessing(file: UploadFile = File(...)) -> Response:
    """Returns the cleaned first page as PNG, useful to debug and for the demo."""
    data = await _read_upload(file)
    result = preprocess(load_pages(data, file.filename or "")[0])
    ok, png = cv2.imencode(".png", result.image)
    return Response(
        png.tobytes(), media_type="image/png",
        headers={"X-Skew-Angle": f"{result.skew_angle:.2f}", "X-Steps": ",".join(result.steps)},
    )
