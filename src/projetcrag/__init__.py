"""Assistant RAG pour interroger un PDF."""


def main() -> None:
    """Point d'entrée installé par ``projetcrag``."""
    from .cli import main as cli_main

    cli_main()


__all__ = ["main"]
