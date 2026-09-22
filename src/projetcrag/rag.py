"""Moteur RAG local fondé uniquement sur all-MiniLM-L6-v2.

all-MiniLM-L6-v2 est un modèle d'embeddings, non un modèle génératif. Le
projet produit donc une réponse extractive et cite les passages du PDF au lieu
d'inventer une synthèse via une API externe.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim


SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass(frozen=True)
class Chunk:
    """Un extrait de PDF et sa page source."""

    text: str
    page: int


def load_pdf(path: str | Path, *, chunk_size: int = 700, overlap: int = 120) -> list[Chunk]:
    """Extrait un PDF et le découpe sans perdre l'information de page."""
    if chunk_size <= overlap or overlap < 0:
        raise ValueError("chunk_size doit être supérieur à overlap (overlap >= 0).")
    pdf_path = Path(path)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF introuvable : {pdf_path}")

    chunks: list[Chunk] = []
    for page_number, page in enumerate(PdfReader(str(pdf_path)).pages, start=1):
        text = re.sub(r"\s+", " ", page.extract_text() or "").strip()
        if not text:
            continue
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            if end < len(text):
                boundary = text.rfind(" ", start, end)
                if boundary > start + chunk_size // 2:
                    end = boundary
            excerpt = text[start:end].strip()
            if excerpt:
                chunks.append(Chunk(excerpt, page_number))
            if end == len(text):
                break
            start = end - overlap
    if not chunks:
        raise ValueError("Le PDF ne contient aucun texte extractible.")
    return chunks


class LocalRetriever:
    """Recherche sémantique locale avec all-MiniLM-L6-v2."""

    def __init__(
        self,
        chunks: Iterable[Chunk],
        *,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
    ) -> None:
        self.chunks = list(chunks)
        if not self.chunks:
            raise ValueError("Impossible de créer un index vide.")
        self.model = SentenceTransformer(model_name, local_files_only=True)
        self.embeddings = self.model.encode(
            [chunk.text for chunk in self.chunks],
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def search(self, query: str, *, k: int = 3) -> list[Chunk]:
        if not query.strip():
            raise ValueError("La question ne peut pas être vide.")
        if k < 1:
            raise ValueError("k doit être supérieur ou égal à 1.")
        query_embedding = self.model.encode(
            query,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        scores = cos_sim(query_embedding, self.embeddings)[0]
        best_indices = scores.argsort(descending=True)[:k].tolist()
        return [self.chunks[index] for index in best_indices]


def citations(chunks: Iterable[Chunk]) -> str:
    return ", ".join(f"p. {page}" for page in sorted({chunk.page for chunk in chunks}))


def extractive_answer(question: str, context: list[Chunk]) -> str:
    """Sélectionne les phrases les plus proches sémantiquement de la question."""
    if not context:
        return "Aucun passage pertinent n'a été trouvé."
    sentences: list[Chunk] = []
    seen: set[str] = set()
    for chunk in context:
        for sentence in SENTENCE_RE.split(chunk.text):
            sentence = sentence.strip()
            if sentence and sentence not in seen:
                seen.add(sentence)
                sentences.append(Chunk(sentence, chunk.page))
    # Le premier passage est déjà classé par pertinence. Deux phrases donnent
    # une réponse lisible tout en restant strictement ancrée dans le document.
    selected = sentences[:2] or context[:1]
    return f"{' '.join(item.text for item in selected)}\n\nSources : {citations(selected)}"


def ask(pdf_path: str | Path, question: str, *, k: int = 3) -> str:
    """Interroge un PDF local sans clé API ni service distant."""
    context = LocalRetriever(load_pdf(pdf_path)).search(question, k=k)
    return extractive_answer(question, context)
