"""Interface de ligne de commande du projet."""

from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from .rag import ask


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Posez une question à un document PDF.")
    parser.add_argument("question", help="Question à poser au document.")
    parser.add_argument("--pdf", type=Path, default=Path("doc/shuri.pdf"), help="PDF à indexer.")
    parser.add_argument("--k", type=int, default=3, help="Nombre d'extraits récupérés (défaut : 3).")
    parser.add_argument("--offline", action="store_true", help="Réponse extractive, sans appel à un LLM.")
    parser.add_argument("--model", help="Modèle OpenAI (défaut : OPENAI_MODEL ou gpt-4.1-mini).")
    return parser


def main() -> None:
    load_dotenv()
    args = build_parser().parse_args()
    try:
        print(ask(args.pdf, args.question, k=args.k, offline=args.offline, model=args.model))
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        raise SystemExit(f"Erreur : {error}") from error


if __name__ == "__main__":
    main()
