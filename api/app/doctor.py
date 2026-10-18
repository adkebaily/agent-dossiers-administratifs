"""Self-check of the local setup. Run: python -m api.app.doctor

Prints one line per check, never prints secret values.
"""

from __future__ import annotations

import os
import shutil
import socket
import sys


def ok(msg: str) -> None:
    print(f"  OK      {msg}")


def bad(msg: str, fix: str) -> None:
    print(f"  ERREUR  {msg}\n          -> {fix}")


def main() -> int:
    errors = 0
    print("Vérification de l'agent dossiers administratifs\n")

    key = os.getenv("ANTHROPIC_API_KEY", "")
    if not key:
        errors += 1
        bad("ANTHROPIC_API_KEY n'est pas chargée",
            "lancez d'abord : set -a; source .env; set +a  (et vérifiez la ligne dans .env)")
    elif key != key.strip() or key.startswith(('"', "'")):
        errors += 1
        bad("ANTHROPIC_API_KEY contient des espaces ou des guillemets", "réécrivez la ligne sans espace ni guillemet")
    elif not key.startswith("sk-ant-api"):
        errors += 1
        bad(f"ANTHROPIC_API_KEY ne commence pas par sk-ant-api (début : {key[:10]}…)",
            "créez une clé sur console.anthropic.com > API Keys")
    else:
        ok(f"ANTHROPIC_API_KEY chargée (début : {key[:13]}…, {len(key)} caractères)")

    api_key = os.getenv("API_KEY")
    if api_key:
        ok(f"API_KEY (protection) active, {len(api_key)} caractères : à coller dans le champ « Clé API »")
    else:
        ok("API_KEY (protection) désactivée : laissez le champ « Clé API » vide")

    if shutil.which("tesseract"):
        import pytesseract
        langs = pytesseract.get_languages(config="")
        if "fra" in langs:
            ok("Tesseract installé avec le français")
        else:
            errors += 1
            bad("Tesseract installé mais sans le français", "sudo apt-get install -y tesseract-ocr-fra")
    else:
        errors += 1
        bad("Tesseract n'est pas installé",
            "sudo apt-get update && sudo apt-get install -y tesseract-ocr tesseract-ocr-fra")

    try:
        socket.getaddrinfo("api.anthropic.com", 443)
        ok("Réseau : api.anthropic.com joignable")
    except OSError as exc:
        errors += 1
        bad(f"Réseau : impossible de résoudre api.anthropic.com ({exc})", "vérifiez la connexion du codespace")

    if key.startswith("sk-ant-api"):
        import anthropic

        from .config import settings
        try:
            anthropic.Anthropic().messages.create(
                model=settings.llm_model, max_tokens=16,
                messages=[{"role": "user", "content": "Réponds juste : ok"}],
            )
            ok(f"Appel au modèle {settings.llm_model} réussi")
        except anthropic.AuthenticationError:
            errors += 1
            bad("Clé Anthropic refusée (invalide ou supprimée)", "créez une nouvelle clé et mettez-la dans .env")
        except anthropic.PermissionDeniedError as exc:
            errors += 1
            bad(f"Accès refusé par Anthropic : {exc.message}", "vérifiez votre compte sur console.anthropic.com")
        except anthropic.BadRequestError as exc:
            errors += 1
            msg = exc.message
            fix = ("ajoutez du crédit : console.anthropic.com > Billing" if "credit" in msg.lower()
                   else "copiez ce message dans la conversation")
            bad(f"Requête refusée : {msg}", fix)
        except anthropic.APIStatusError as exc:
            errors += 1
            bad(f"Erreur Anthropic {exc.status_code} : {exc.message}", "copiez ce message dans la conversation")
        except anthropic.APIConnectionError as exc:
            errors += 1
            bad(f"Connexion à Anthropic impossible : {exc.__cause__ or exc}", "vérifiez le réseau du codespace")

    print("\nTout est prêt." if errors == 0 else f"\n{errors} problème(s) à corriger (voir ci-dessus).")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
