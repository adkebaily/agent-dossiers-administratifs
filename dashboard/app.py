"""Streamlit dashboard: processed dossiers, review queue and evaluation metrics."""

import json
import os
from pathlib import Path

import pandas as pd
import psycopg
import streamlit as st

st.set_page_config(page_title="Dossiers administratifs", layout="wide")
DATABASE_URL = os.environ["DATABASE_URL"]
RESULTS = Path(os.getenv("EVAL_RESULTS", "/srv/eval/results/metrics.json"))


@st.cache_data(ttl=10)
def load_documents() -> pd.DataFrame:
    with psycopg.connect(DATABASE_URL) as conn:
        return pd.read_sql(
            """SELECT d.id, d.dossier_id, s.dossier_type, d.filename, d.document_type,
                      d.type_confidence, d.ocr_confidence, d.status, d.review_reasons,
                      d.processing_ms, d.created_at
               FROM documents d JOIN dossiers s ON s.id = d.dossier_id
               ORDER BY d.created_at DESC""",
            conn,
        )


st.title("Agent de traitement de dossiers administratifs")
docs = load_documents()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Documents traités", len(docs))
c2.metric("Dossiers", docs["dossier_id"].nunique() if len(docs) else 0)
c3.metric("En attente de validation", int((docs["status"] == "pending_review").sum()) if len(docs) else 0)
c4.metric("Temps moyen (s)", f"{docs['processing_ms'].mean() / 1000:.1f}" if len(docs) else "—")

tab_docs, tab_review, tab_eval = st.tabs(["Documents", "File de validation", "Évaluation"])

with tab_docs:
    if len(docs):
        st.bar_chart(docs.groupby(["document_type", "status"]).size().unstack(fill_value=0))
    st.dataframe(docs.drop(columns=["review_reasons"]), use_container_width=True)

with tab_review:
    pending = docs[docs["status"] == "pending_review"] if len(docs) else docs
    for _, row in pending.iterrows():
        with st.expander(f"#{row['id']} — {row['filename']} ({row['document_type']})"):
            for reason in row["review_reasons"]:
                st.write(f"- {reason}")
    if not len(pending):
        st.info("Aucun document en attente.")

with tab_eval:
    if RESULTS.exists():
        m = json.loads(RESULTS.read_text())
        if "classification_accuracy" in m:
            e1, e2, e3 = st.columns(3)
            e1.metric("Précision de classification", f"{m['classification_accuracy']:.1%}")
            e2.metric("Exactitude des champs", f"{m['field_exact_match_accuracy']:.1%}")
            e3.metric("Taux de revue humaine", f"{m['human_review_rate']:.1%}")
            st.dataframe(pd.DataFrame(m["per_field"]).T, use_container_width=True)
        if "cer_by_preprocessing" in m:
            st.subheader("OCR : taux d'erreur caractère selon le prétraitement")
            st.bar_chart(pd.Series(m["cer_by_preprocessing"], name="CER"))
    else:
        st.info("Lancez `python eval/evaluate.py` pour générer les métriques.")
