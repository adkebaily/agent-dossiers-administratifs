"""Evaluate the pipeline on a labelled dataset.

Two modes:
  --mode ocr   Preprocessing ablation, no LLM call: character error rate (CER) of
               Tesseract with and without each preprocessing step. Free to run.
  --mode full  End-to-end: classification accuracy and field-level extraction
               precision / recall / F1 (needs ANTHROPIC_API_KEY, costs API credits).

    python eval/evaluate.py --dataset eval/dataset --mode ocr
    python eval/evaluate.py --dataset eval/dataset --mode full --limit 20
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from api.app.ocr import run_ocr  # noqa: E402
from api.app.preprocessing import PreprocessOptions, load_pages, preprocess  # noqa: E402

ABLATIONS = {
    "raw (aucun prétraitement)": PreprocessOptions(denoise=False, clahe=False, deskew=False, binarize=False),
    "deskew seul": PreprocessOptions(denoise=False, clahe=False, deskew=True, binarize=False),
    "deskew + sauvola": PreprocessOptions(denoise=False, clahe=False, deskew=True, binarize=True),
    "pipeline complet": PreprocessOptions(),
}


def load_labels(dataset: Path, limit: int | None) -> list[dict]:
    rows = [json.loads(l) for l in (dataset / "labels.jsonl").read_text(encoding="utf-8").splitlines() if l]
    return rows[:limit] if limit else rows


def normalize(value: str | None) -> str:
    if value is None:
        return ""
    v = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode().lower()
    v = v.replace(",", ".")
    return re.sub(r"[\s€]", "", v)


def cer(reference: str, hypothesis: str) -> float:
    import jiwer

    ref = " ".join(reference.split())
    hyp = " ".join(hypothesis.split())
    return jiwer.cer(ref, hyp)


def eval_ocr(rows: list[dict], dataset: Path) -> dict:
    results = {}
    for name, opts in ABLATIONS.items():
        errs = []
        for r in rows:
            page = load_pages((dataset / r["file"]).read_bytes(), r["file"])[0]
            text = run_ocr(preprocess(page, replace(opts)).image).text
            errs.append(cer(r["text"], text))
        results[name] = sum(errs) / len(errs)
        print(f"  {name:<28} CER = {results[name]:.3f}")
    return {"cer_by_preprocessing": results}


def eval_full(rows: list[dict], dataset: Path) -> dict:
    from api.app.pipeline import process_document

    correct_type = 0
    tp, fp, fn = defaultdict(int), defaultdict(int), defaultdict(int)
    flagged, errors = 0, []
    for r in rows:
        out = process_document((dataset / r["file"]).read_bytes(), r["file"])
        correct_type += out.document_type == r["document_type"]
        flagged += out.needs_review
        pred = {k: v["value"] for k, v in out.fields.items()}
        for name, truth in r["fields"].items():
            key = f"{r['document_type']}.{name}"
            got = pred.get(name)
            if got is None:
                fn[key] += 1
            elif normalize(got) == normalize(truth):
                tp[key] += 1
            else:
                fp[key] += 1
                fn[key] += 1
                errors.append({"file": r["file"], "field": name, "expected": truth, "got": got})
        print(f"  {r['file']:<36} -> {out.document_type:<22} review={out.needs_review}")

    def prf(t, p, n):
        prec = t / (t + p) if t + p else 0.0
        rec = t / (t + n) if t + n else 0.0
        return prec, rec, (2 * prec * rec / (prec + rec) if prec + rec else 0.0)

    per_field = {k: dict(zip(("precision", "recall", "f1"), prf(tp[k], fp[k], fn[k])))
                 for k in sorted(set(tp) | set(fn))}
    micro = dict(zip(("precision", "recall", "f1"),
                     prf(sum(tp.values()), sum(fp.values()), sum(fn.values()))))
    return {
        "classification_accuracy": correct_type / len(rows),
        "field_micro": micro,
        "field_exact_match_accuracy": sum(tp.values()) / sum(len(r["fields"]) for r in rows),
        "human_review_rate": flagged / len(rows),
        "per_field": per_field,
        "errors": errors[:50],
    }


def write_report(metrics: dict, path: Path) -> None:
    lines = ["# Résultats d'évaluation", ""]
    if "cer_by_preprocessing" in metrics:
        lines += ["## OCR — effet du prétraitement (CER, plus bas = mieux)", "",
                  "| Configuration | CER |", "|---|---|"]
        lines += [f"| {k} | {v:.3f} |" for k, v in metrics["cer_by_preprocessing"].items()]
        lines.append("")
    if "classification_accuracy" in metrics:
        m = metrics
        lines += ["## Extraction de bout en bout", "",
                  f"- Précision de classification : **{m['classification_accuracy']:.1%}**",
                  f"- Exactitude des champs (exact match normalisé) : **{m['field_exact_match_accuracy']:.1%}**",
                  f"- F1 micro des champs : **{m['field_micro']['f1']:.3f}**",
                  f"- Taux de renvoi en validation humaine : **{m['human_review_rate']:.1%}**", "",
                  "| Champ | Précision | Rappel | F1 |", "|---|---|---|---|"]
        lines += [f"| {k} | {v['precision']:.2f} | {v['recall']:.2f} | {v['f1']:.2f} |"
                  for k, v in m["per_field"].items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=ROOT / "eval" / "dataset")
    ap.add_argument("--mode", choices=["ocr", "full", "both"], default="ocr")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out", type=Path, default=ROOT / "eval" / "results")
    args = ap.parse_args()

    rows = load_labels(args.dataset, args.limit)
    print(f"{len(rows)} documents")
    metrics: dict = {}
    if args.mode in ("ocr", "both"):
        print("OCR ablation")
        metrics.update(eval_ocr(rows, args.dataset))
    if args.mode in ("full", "both"):
        print("End-to-end")
        metrics.update(eval_full(rows, args.dataset))

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(metrics, args.out / "report.md")
    print(f"Report written to {args.out / 'report.md'}")


if __name__ == "__main__":
    main()
