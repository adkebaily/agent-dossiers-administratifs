"""Generate a labelled test set of synthetic, degraded administrative documents.

No real personal data: names, addresses and numbers are random. Each document is
rendered, then degraded like a phone photo / bad scan (rotation, blur, noise, uneven
lighting, JPEG compression). Ground truth is written to labels.jsonl.

    python eval/generate_synthetic.py --n 60 --out eval/dataset
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import date, timedelta
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PRENOMS = ["Camille", "Yanis", "Inès", "Lucas", "Fatima", "Hugo", "Léa", "Mehdi", "Chloé", "Amine", "Sarah", "Théo"]
NOMS = ["MARTIN", "BENALI", "DUBOIS", "NGUYEN", "LEFÈVRE", "HADDAD", "MOREAU", "TRAORÉ", "GIRARD", "ROUSSEAU"]
RUES = ["rue Victor Hugo", "avenue Jean Jaurès", "boulevard Pasteur", "rue de la République", "place Bellecour"]
VILLES = [("69007", "Lyon"), ("38000", "Grenoble"), ("31000", "Toulouse"), ("13005", "Marseille"), ("75011", "Paris")]
EMETTEURS = ["EDF", "Engie", "TotalEnergies", "Orange", "SFR", "Free"]
ETABLISSEMENTS = ["Université Grenoble Alpes", "Université Claude Bernard Lyon 1", "Université Toulouse III"]

FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    "/Library/Fonts/Arial.ttf",
    "C:/Windows/Fonts/arial.ttf",
]


def font(size: int) -> ImageFont.FreeTypeFont:
    for p in FONT_PATHS:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default(size)


def rand_date(start: date, end: date) -> date:
    return start + timedelta(days=random.randint(0, (end - start).days))


def fr(d: date) -> str:
    return d.strftime("%d/%m/%Y")


def make_iban() -> str:
    bban = "".join(random.choices("0123456789", k=23))
    check = 98 - int("".join(str(int(c, 36)) for c in bban + "FR00")) % 97
    return f"FR{check:02d}{bban}"


def person() -> dict:
    return {"prenom": random.choice(PRENOMS), "nom": random.choice(NOMS)}


def gen_justificatif_domicile(today: date):
    p = person()
    cp, ville = random.choice(VILLES)
    adresse = f"{random.randint(1, 120)} {random.choice(RUES)}"
    d = rand_date(today - timedelta(days=80), today)
    emetteur = random.choice(EMETTEURS)
    fields = {**p, "adresse": adresse, "code_postal": cp, "ville": ville,
              "date_document": d.isoformat(), "emetteur": emetteur,
              "numero_client": str(random.randint(10**9, 10**10 - 1))}
    lines = [
        (emetteur, 46), ("FACTURE", 34), ("",),
        (f"{p['prenom']} {p['nom']}",), (adresse,), (f"{cp} {ville}",), ("",),
        (f"Date de la facture : {fr(d)}",), (f"N° client : {fields['numero_client']}",),
        (f"Montant TTC : {random.randint(30, 180)},{random.randint(10, 99)} €",),
    ]
    return fields, lines


def gen_avis_imposition(today: date):
    p = person()
    annee = today.year - 1
    fields = {**p, "numero_fiscal": "".join(random.choices("0123456789", k=13)),
              "reference_avis": f"{random.randint(10, 99)} {random.randint(10, 99)} {random.randint(1000000, 9999999)} {random.randint(10, 99)}",
              "annee_revenus": str(annee), "revenu_fiscal_reference": str(random.randint(8000, 60000)),
              "nombre_parts": random.choice(["1", "1.5", "2", "2.5", "3"])}
    lines = [
        ("DIRECTION GÉNÉRALE DES FINANCES PUBLIQUES", 26),
        (f"AVIS D'IMPÔT {annee + 1}", 38), (f"SUR LES REVENUS DE L'ANNÉE {annee}", 30), ("",),
        (f"{p['nom']} {p['prenom']}",), ("",),
        (f"Numéro fiscal : {fields['numero_fiscal']}",),
        (f"Référence de l'avis : {fields['reference_avis']}",),
        (f"Nombre de parts : {fields['nombre_parts'].replace('.', ',')}",),
        (f"Revenu fiscal de référence : {int(fields['revenu_fiscal_reference']):,} €".replace(",", " "),),
    ]
    return fields, lines


def gen_rib(today: date):
    p = person()
    iban = make_iban()
    bic = random.choice(["CCBPFRPPLYO", "AGRIFRPP881", "BNPAFRPPXXX", "SOGEFRPP"])
    fields = {"titulaire": f"{p['prenom']} {p['nom']}", "iban": iban, "bic": bic, "banque": "Banque Exemple"}
    lines = [
        ("RELEVÉ D'IDENTITÉ BANCAIRE", 38), ("Banque Exemple", 28), ("",),
        (f"Titulaire : {fields['titulaire']}",),
        (f"IBAN : {' '.join(iban[i:i + 4] for i in range(0, len(iban), 4))}",),
        (f"BIC : {bic}",),
    ]
    return fields, lines


def gen_certificat_scolarite(today: date):
    p = person()
    y = today.year if today.month >= 9 else today.year - 1
    fields = {**p, "etablissement": random.choice(ETABLISSEMENTS), "annee_universitaire": f"{y}-{y + 1}",
              "formation": random.choice(["Master Traitement du Signal et des Images", "Licence Informatique", "Master Data Science"]),
              "numero_etudiant": str(random.randint(10**7, 10**8 - 1))}
    lines = [
        (fields["etablissement"], 34), ("CERTIFICAT DE SCOLARITÉ", 38), ("",),
        (f"Année universitaire {fields['annee_universitaire']}",), ("",),
        (f"Nous certifions que {p['prenom']} {p['nom']}",),
        (f"N° étudiant : {fields['numero_etudiant']}",),
        (f"est régulièrement inscrit(e) en {fields['formation']}.",),
    ]
    return fields, lines


GENERATORS = {
    "justificatif_domicile": gen_justificatif_domicile,
    "avis_imposition": gen_avis_imposition,
    "rib": gen_rib,
    "certificat_scolarite": gen_certificat_scolarite,
}


def render(lines) -> np.ndarray:
    img = Image.new("L", (1240, 1754), 255)  # A4 at 150 dpi
    draw = ImageDraw.Draw(img)
    y = 110
    for line in lines:
        text, size = line[0], (line[1] if len(line) > 1 else 26)
        draw.text((100, y), text, fill=0, font=font(size))
        y += int(size * 1.7) if text else 30
    return np.array(img)


def degrade(img: np.ndarray, level: float, rng: np.random.Generator) -> np.ndarray:
    """level in [0, 1]: 0 = clean scan, 1 = bad phone photo."""
    h, w = img.shape
    angle = rng.uniform(-8, 8) * level
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    img = cv2.warpAffine(img, m, (w, h), borderValue=255)
    # Uneven illumination: multiply by a smooth gradient
    gx = np.linspace(1.0, 1.0 - 0.45 * level, w)[None, :]
    gy = np.linspace(1.0 - 0.2 * level, 1.0, h)[:, None]
    img = img.astype(np.float64) * gx * gy
    k = 1 + 2 * int(rng.integers(0, 2 + int(2 * level)))
    img = cv2.GaussianBlur(img, (k, k), 0)
    img = img + rng.normal(0, 25 * level, img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    quality = int(90 - 50 * level)
    _, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--out", type=Path, default=Path("eval/dataset"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    today = date.today()
    with open(args.out / "labels.jsonl", "w", encoding="utf-8") as f:
        for i in range(args.n):
            doc_type = list(GENERATORS)[i % len(GENERATORS)]
            fields, lines = GENERATORS[doc_type](today)
            level = float(rng.uniform(0, 1))
            img = degrade(render(lines), level, rng)
            name = f"{i:03d}_{doc_type}.jpg"
            cv2.imwrite(str(args.out / name), img)
            text = "\n".join(l[0] for l in lines if l[0])
            f.write(json.dumps({"file": name, "document_type": doc_type, "fields": fields,
                                "text": text, "degradation": round(level, 3)}, ensure_ascii=False) + "\n")
    print(f"{args.n} documents written to {args.out}")


if __name__ == "__main__":
    main()
