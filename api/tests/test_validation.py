from datetime import date

from api.app.validation import check_document, check_dossier, iban_is_valid

TODAY = date(2026, 10, 18)


def test_iban_checksum():
    assert iban_is_valid("FR76 3000 6000 0112 3456 7890 189")
    assert not iban_is_valid("FR76 3000 6000 0112 3456 7890 188")


def test_old_proof_of_address_is_rejected():
    fields = {"nom": "MARTIN", "adresse": "1 rue X", "code_postal": "69007", "ville": "Lyon",
              "date_document": "2026-03-01", "emetteur": "EDF"}
    check = check_document("justificatif_domicile", fields, today=TODAY)
    assert not check.ok
    assert any("92 jours" in e for e in check.errors)


def test_expired_id_and_missing_fields():
    check = check_document("piece_identite", {"nom": "MARTIN", "date_expiration": "2025-01-01"}, today=TODAY)
    assert "prenom" in check.missing_fields
    assert any("expiré" in e for e in check.errors)


def doc(t, nom="Camille MARTIN", status="validated"):
    return {"document_type": t, "fields": {"nom": nom}, "status": status}


def test_dossier_missing_pieces():
    result = check_dossier("caf_aide_logement", [doc("piece_identite"), doc("rib")])
    assert not result.complete
    assert set(result.missing) == {"justificatif_domicile", "avis_imposition"}


def test_dossier_any_of_and_name_mismatch():
    docs = [doc("piece_identite"), doc("certificat_scolarite"), doc("avis_imposition"),
            doc("rib"), doc("attestation_caf", nom="Yanis BENALI")]
    result = check_dossier("crous_dse", docs)
    assert result.missing == []
    assert any("incohérent" in i for i in result.issues)
    assert not result.complete


def test_dossier_complete():
    docs = [doc(t) for t in ("piece_identite", "justificatif_domicile", "avis_imposition", "rib")]
    assert check_dossier("caf_aide_logement", docs).complete


def test_rejected_documents_do_not_count():
    docs = [doc(t) for t in ("piece_identite", "justificatif_domicile", "avis_imposition")]
    docs.append(doc("rib", status="rejected"))
    assert check_dossier("caf_aide_logement", docs).missing == ["rib"]
