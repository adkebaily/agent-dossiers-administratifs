CREATE TABLE IF NOT EXISTS dossiers (
    id            TEXT PRIMARY KEY,
    dossier_type  TEXT NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS documents (
    id                SERIAL PRIMARY KEY,
    dossier_id        TEXT NOT NULL REFERENCES dossiers(id) ON DELETE CASCADE,
    filename          TEXT NOT NULL,
    document_type     TEXT NOT NULL,
    type_confidence   REAL NOT NULL,
    fields            JSONB NOT NULL,          -- {name: {value, confidence}}
    ocr_confidence    REAL NOT NULL,
    skew_angle        REAL,
    checks            JSONB NOT NULL,          -- missing_fields, errors, warnings
    status            TEXT NOT NULL,           -- validated | pending_review | rejected
    review_reasons    JSONB NOT NULL DEFAULT '[]',
    reviewer_comment  TEXT,
    processing_ms     INTEGER,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    reviewed_at       TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS documents_dossier_idx ON documents(dossier_id);
CREATE INDEX IF NOT EXISTS documents_status_idx ON documents(status);
