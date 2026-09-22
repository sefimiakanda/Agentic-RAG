"""Interface Streamlit de l'assistant RAG local."""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from projetcrag.rag import LocalRetriever, extractive_answer, load_pdf


DEFAULT_PDF = Path("doc/shuri.pdf")


@st.cache_resource(show_spinner="Chargement du modèle et indexation du document...")
def build_retriever(pdf_bytes: bytes, file_name: str) -> LocalRetriever:
    """Construit et met en cache l'index pour le document actuellement ouvert."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temporary_file:
        temporary_file.write(pdf_bytes)
        temporary_path = Path(temporary_file.name)
    try:
        return LocalRetriever(load_pdf(temporary_path))
    finally:
        temporary_path.unlink(missing_ok=True)


def main() -> None:
    st.set_page_config(page_title="ShuriX RAG", page_icon="📄", layout="centered")
    st.title("📄 Assistant documentaire ShuriX")
    st.caption("Recherche locale avec all-MiniLM-L6-v2 — aucune clé API requise.")

    uploaded_file = st.sidebar.file_uploader("Importer un PDF", type="pdf")
    if uploaded_file is None:
        if not DEFAULT_PDF.is_file():
            st.error("Le PDF par défaut est introuvable. Importez un PDF dans la barre latérale.")
            return
        pdf_bytes = DEFAULT_PDF.read_bytes()
        file_name = DEFAULT_PDF.name
    else:
        pdf_bytes = uploaded_file.getvalue()
        file_name = uploaded_file.name

    st.sidebar.info(f"Document actif : {file_name}")
    question = st.text_input(
        "Votre question",
        placeholder="Ex. Quelle est la mission de ShuriX Tech ?",
    )
    if not question:
        return

    try:
        retriever = build_retriever(pdf_bytes, file_name)
        passages = retriever.search(question, k=3)
        st.subheader("Réponse")
        st.write(extractive_answer(question, passages))
        with st.expander("Voir les passages récupérés"):
            for passage in passages:
                st.markdown(f"**Page {passage.page}**")
                st.write(passage.text)
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        st.error(str(error))


if __name__ == "__main__":
    main()
