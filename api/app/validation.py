"""Deterministic checks on extracted fields and dossier completeness.

The LLM extracts; these rules decide. Keeping the business rules in plain Python
makes them testable and auditable, which matters for a public-service use case.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date

from rapidfuzz import fuzz

from .catalog import document_types, dossier_config

IBAN_RE = re.compile(r"^[A-Z]{2}\d{2}[A-Z0-9]{11,30}$")


@dataclass
class DocumentCheck:
    missing_fields: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.missing_fields and not self.errors


def parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def iban_is_valid(iban: str) -> bool:
    """ISO 13616 mod-97 check."""
    iban = iban.replace(" ", "").upper()
    if not IBAN_RE.match(iban):
        return False
    rearranged = iban[4:] + iban[:4]
    digits = "".join(str(int(c, 36)) for c in rearranged)
    return int(digits) % 97 == 1


def check_document(doc_type: str, fields: dict[str, str | None], today: date | None = None) -> DocumentCheck:
    today = today or date.today()
    spec = document_types().get(doc_type, {"required": []})
    check = DocumentCheck()

    check.missing_fields = [f for f in spec["required"] if not fields.get(f)]

    if expiry := spec.get("expiry_field"):
        d = parse_date(fields.get(expiry))
        if fields.get(expiry) and d is None:
            check.errors.append(f"{expiry} : date illisible")
        elif d and d < today:
            check.errors.append(f"Document expiré le {d.isoformat()}")

    if (date_field := spec.get("date_field")) and (max_age := spec.get("max_age_days")):
        d = parse_date(fields.get(date_field))
        if d and (today - d).days > max_age:
            check.errors.append(f"Document de plus de {max_age} jours ({d.isoformat()})")
        if d and d > today:
            check.errors.append(f"Date dans le futur ({d.isoformat()})")

    if doc_type == "rib" and fields.get("iban") and not iban_is_valid(fields["iban"]):
        check.errors.append("IBAN invalide (clé de contrôle)")

    if doc_type == "avis_imposition" and (year := fields.get("annee_revenus")):
        if year.isdigit() and int(year) < today.year - 2:
            check.warnings.append(f"Avis portant sur les revenus {year} : vérifier l'année exigée")

    return check


def normalize_name(name: str) -> str:
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z ]", " ", name.lower()).strip()


@dataclass
class DossierCheck:
    dossier_type: str
    present: list[str]
    missing: list[str]
    complete: bool
    issues: list[str]


def check_dossier(dossier_type: str, documents: list[dict]) -> DossierCheck:
    """`documents`: list of {"document_type", "fields", "status"} for the dossier."""
    cfg = dossier_config()
    spec = cfg["dossiers"][dossier_type]
    valid = [d for d in documents if d.get("status") != "rejected"]
    present = sorted({d["document_type"] for d in valid})

    missing = [t for t in spec.get("required", []) if t not in present]
    for group in spec.get("any_of", []):
        if not any(t in present for t in group):
            missing.append(" ou ".join(group))

    issues: list[str] = []
    names = [
        normalize_name(d["fields"].get("nom") or d["fields"].get("titulaire") or "")
        for d in valid
    ]
    names = [n for n in names if n]
    threshold = cfg["cross_checks"]["name_consistency_threshold"]
    if names:
        reference = names[0]
        for n in names[1:]:
            if fuzz.token_set_ratio(reference, n) < threshold:
                issues.append(f"Nom incohérent entre les pièces : « {reference} » / « {n} »")

    for d in valid:
        if d.get("status") == "pending_review":
            issues.append(f"{d['document_type']} en attente de validation humaine")

    return DossierCheck(
        dossier_type=dossier_type,
        present=present,
        missing=missing,
        complete=not missing and not issues,
        issues=issues,
    )
