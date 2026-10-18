"""Loads the document types and dossier checklists from the YAML config."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

from .config import settings


@lru_cache
def document_types(config_dir: Path | None = None) -> dict:
    path = (config_dir or settings.config_dir) / "document_types.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))["document_types"]


@lru_cache
def dossier_config(config_dir: Path | None = None) -> dict:
    path = (config_dir or settings.config_dir) / "dossiers.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))
