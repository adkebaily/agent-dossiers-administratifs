"""PostgreSQL persistence (psycopg 3). Falls back to an in-memory store when
DATABASE_URL is not set, so the API and tests run without a database."""

from __future__ import annotations

import itertools
from datetime import datetime, timezone

from .config import settings


class MemoryStore:
    def __init__(self) -> None:
        self.dossiers: dict[str, dict] = {}
        self.documents: dict[int, dict] = {}
        self._ids = itertools.count(1)

    def upsert_dossier(self, dossier_id: str, dossier_type: str) -> None:
        self.dossiers.setdefault(dossier_id, {"id": dossier_id, "dossier_type": dossier_type})

    def get_dossier(self, dossier_id: str) -> dict | None:
        return self.dossiers.get(dossier_id)

    def insert_document(self, doc: dict) -> int:
        doc_id = next(self._ids)
        self.documents[doc_id] = {**doc, "id": doc_id, "created_at": datetime.now(timezone.utc)}
        return doc_id

    def get_document(self, doc_id: int) -> dict | None:
        return self.documents.get(doc_id)

    def list_documents(self, dossier_id: str) -> list[dict]:
        return [d for d in self.documents.values() if d["dossier_id"] == dossier_id]

    def review_document(self, doc_id: int, status: str, fields: dict | None, comment: str | None) -> None:
        doc = self.documents[doc_id]
        doc["status"] = status
        doc["reviewer_comment"] = comment
        if fields is not None:
            doc["fields"] = fields


class PostgresStore:
    def __init__(self, url: str) -> None:
        from psycopg_pool import ConnectionPool
        from psycopg.rows import dict_row

        self.pool = ConnectionPool(url, kwargs={"row_factory": dict_row}, open=True)

    def upsert_dossier(self, dossier_id: str, dossier_type: str) -> None:
        with self.pool.connection() as conn:
            conn.execute(
                """INSERT INTO dossiers (id, dossier_type) VALUES (%s, %s)
                   ON CONFLICT (id) DO UPDATE SET updated_at = now()""",
                (dossier_id, dossier_type),
            )

    def get_dossier(self, dossier_id: str) -> dict | None:
        with self.pool.connection() as conn:
            return conn.execute("SELECT * FROM dossiers WHERE id = %s", (dossier_id,)).fetchone()

    def insert_document(self, doc: dict) -> int:
        from psycopg.types.json import Jsonb

        with self.pool.connection() as conn:
            row = conn.execute(
                """INSERT INTO documents (dossier_id, filename, document_type, type_confidence,
                       fields, ocr_confidence, skew_angle, checks, status, review_reasons, processing_ms)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id""",
                (
                    doc["dossier_id"], doc["filename"], doc["document_type"], doc["type_confidence"],
                    Jsonb(doc["fields"]), doc["ocr_confidence"], doc["skew_angle"],
                    Jsonb(doc["checks"]), doc["status"], Jsonb(doc["review_reasons"]),
                    doc["processing_ms"],
                ),
            ).fetchone()
            return row["id"]

    def get_document(self, doc_id: int) -> dict | None:
        with self.pool.connection() as conn:
            return conn.execute("SELECT * FROM documents WHERE id = %s", (doc_id,)).fetchone()

    def list_documents(self, dossier_id: str) -> list[dict]:
        with self.pool.connection() as conn:
            return conn.execute(
                "SELECT * FROM documents WHERE dossier_id = %s ORDER BY created_at", (dossier_id,)
            ).fetchall()

    def review_document(self, doc_id: int, status: str, fields: dict | None, comment: str | None) -> None:
        from psycopg.types.json import Jsonb

        with self.pool.connection() as conn:
            conn.execute(
                """UPDATE documents SET status = %s, reviewer_comment = %s, reviewed_at = now(),
                       fields = COALESCE(%s, fields) WHERE id = %s""",
                (status, comment, Jsonb(fields) if fields is not None else None, doc_id),
            )


store = PostgresStore(settings.database_url) if settings.database_url else MemoryStore()
