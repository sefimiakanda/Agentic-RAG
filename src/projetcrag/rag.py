"""Petit assistant RAG, reproductible et utilisable en ligne de commande.

Le moteur de recherche est volontairement local (TF-IDF) : indexer un PDF ne
nécessite donc ni téléchargement d'un modèle, ni clé API.  Une clé OpenAI est
uniquement nécessaire pour la synthèse générative.
"""

from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv
from pypdf import PdfReader


TOKEN_RE = re.compile(r"[\wÀ-ÖØ-öø-ÿ'-]+", re.UNICODE)
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
STOP_WORDS = frozenset(
    "à au aux avec ce ces dans de des du elle en est et il je la le les leur leurs "
    "ma mais me mes moi mon ne nos notre nous on ou par pas pour qu que quel quelle "
    "qui sa se ses son sur ta te tes ton tu un une vos votre vous y quelle quels quelles "
    "d l j c n s t".split()
)


@dataclass(frozen=True)
class Chunk:
    """Un extrait de document, avec son numéro de page pour la citation."""

    text: str
    page: int


def _tokens(text: str) -> list[str]:
    return [
        token.lower()
        for token in TOKEN_RE.findall(text)
        if len(token) > 1 and token.lower() not in STOP_WORDS
    ]


def _extract_page_text(pdf_path: Path, page_number: int, fallback: str) -> str:
    """Corrige les PDF dont la table Unicode est défectueuse quand pdftotext existe."""
    if "�" not in fallback:
        return fallback
    executable = shutil.which("pdftotext")
    if not executable:
        return fallback
    try:
        result = subprocess.run(
            [
                executable,
                "-enc",
                "UTF-8",
                "-f",
                str(page_number),
                "-l",
                str(page_number),
                str(pdf_path),
                "-",
            ],
            capture_output=True,
            check=True,
            encoding="utf-8",
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError, UnicodeError):
        return fallback
    return result.stdout if result.stdout.strip() else fallback


def load_pdf(path: str | Path, *, chunk_size: int = 900, overlap: int = 150) -> list[Chunk]:
    """Extrait et découpe un PDF en préservant la page d'origine."""
    if chunk_size <= overlap or overlap < 0:
        raise ValueError("chunk_size doit être supérieur à overlap (overlap >= 0).")

    pdf_path = Path(path)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF introuvable : {pdf_path}")

    chunks: list[Chunk] = []
    for page_number, page in enumerate(PdfReader(str(pdf_path)).pages, start=1):
        extracted = page.extract_text() or ""
        text = re.sub(
            r"\s+", " ", _extract_page_text(pdf_path, page_number, extracted)
        ).strip()
        if not text:
            continue
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            # Préférer une limite de mot pour ne pas couper un terme en deux.
            if end < len(text):
                boundary = text.rfind(" ", start, end)
                if boundary > start + chunk_size // 2:
                    end = boundary
            excerpt = text[start:end].strip()
            if excerpt:
                chunks.append(Chunk(text=excerpt, page=page_number))
            if end == len(text):
                break
            start = end - overlap
    if not chunks:
        raise ValueError("Le PDF ne contient aucun texte extractible.")
    return chunks


class LocalRetriever:
    """Recherche TF-IDF déterministe, adaptée aux petits corpus documentaires."""

    def __init__(self, chunks: Iterable[Chunk]) -> None:
        self.chunks = list(chunks)
        if not self.chunks:
            raise ValueError("Impossible de créer un index vide.")
        self._term_counts = [Counter(_tokens(chunk.text)) for chunk in self.chunks]
        document_frequency = Counter(
            term for counts in self._term_counts for term in counts
        )
        total = len(self.chunks)
        self._idf = {
            term: math.log((total + 1) / (frequency + 1)) + 1
            for term, frequency in document_frequency.items()
        }

    def search(self, query: str, *, k: int = 3) -> list[Chunk]:
        if not query.strip():
            raise ValueError("La question ne peut pas être vide.")
        if k < 1:
            raise ValueError("k doit être supérieur ou égal à 1.")

        query_counts = Counter(_tokens(query))
        if not query_counts:
            return self.chunks[:k]
        query_weights = {
            term: count * self._idf.get(term, 0.0)
            for term, count in query_counts.items()
        }
        query_norm = math.sqrt(sum(weight**2 for weight in query_weights.values()))

        scored: list[tuple[float, int]] = []
        for index, counts in enumerate(self._term_counts):
            dot = sum(
                query_weights.get(term, 0.0) * count * self._idf.get(term, 0.0)
                for term, count in counts.items()
            )
            chunk_norm = math.sqrt(
                sum((count * self._idf.get(term, 0.0)) ** 2 for term, count in counts.items())
            )
            score = dot / (query_norm * chunk_norm) if query_norm and chunk_norm else 0.0
            scored.append((score, index))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [self.chunks[index] for _, index in scored[:k]]


def citations(chunks: Iterable[Chunk]) -> str:
    pages = sorted({chunk.page for chunk in chunks})
    return ", ".join(f"p. {page}" for page in pages)


def extractive_answer(question: str, context: list[Chunk]) -> str:
    """Réponse de secours sans LLM, fondée exclusivement sur le contexte."""
    terms = set(_tokens(question))
    # Les noms propres dans une question (p. ex. « ShuriX Tech ») décrivent
    # souvent le sujet, pas l'information recherchée. Les écarter privilégie
    # « mission », « drone », « formations », etc. dans la réponse extractive.
    focus_terms = {
        token.lower()
        for token in TOKEN_RE.findall(question)
        if len(token) > 1 and not token[0].isupper() and token.lower() not in STOP_WORDS
    }
    if focus_terms:
        terms = focus_terms
    sentence_pages: dict[str, int] = {}
    for chunk in context:
        for sentence in SENTENCE_RE.split(chunk.text):
            sentence = sentence.strip()
            if sentence:
                sentence_pages.setdefault(sentence, chunk.page)
    sentences = list(sentence_pages)
    sentence_frequency = Counter(
        term for sentence in sentences for term in set(_tokens(sentence))
    )
    total_sentences = len(sentences)
    ranked = sorted(
        enumerate(sentences),
        key=lambda item: (
            -sum(
                math.log((total_sentences + 1) / (sentence_frequency[term] + 1)) + 1
                for term in terms.intersection(_tokens(item[1]))
            ),
            item[0],
        ),
    )
    selected = [sentence for _, sentence in ranked[:2]]
    selected_chunks = [Chunk(sentence, sentence_pages[sentence]) for sentence in selected]
    return " ".join(selected) + f"\n\nSources : {citations(selected_chunks)}"


def generate_answer(question: str, context: list[Chunk], *, model: str) -> str:
    """Synthétise une réponse avec OpenAI, en refusant toute invention."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY est absente. Ajoutez-la au fichier .env ou utilisez --offline."
        )
    from langchain_openai import ChatOpenAI

    source_text = "\n\n".join(f"[Page {chunk.page}]\n{chunk.text}" for chunk in context)
    prompt = (
        "Tu es un assistant documentaire. Réponds en français uniquement à partir "
        "des extraits ci-dessous. Si l'information n'y figure pas, dis-le clairement. "
        "Sois concis et termine par 'Sources : p. X'.\n\n"
        f"EXTRAITS\n{source_text}\n\nQUESTION\n{question}"
    )
    response = ChatOpenAI(model=model, temperature=0).invoke(prompt)
    return str(response.content).strip()


def ask(
    pdf_path: str | Path,
    question: str,
    *,
    k: int = 3,
    offline: bool = False,
    model: str | None = None,
) -> str:
    """Indexe *pdf_path*, récupère les extraits pertinents et répond à *question*."""
    context = LocalRetriever(load_pdf(pdf_path)).search(question, k=k)
    if offline:
        return extractive_answer(question, context)
    return generate_answer(question, context, model=model or os.getenv("OPENAI_MODEL", "gpt-4.1-mini"))
