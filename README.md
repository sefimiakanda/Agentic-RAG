# Projet RAG ShuriX Tech

Assistant de questions-réponses sur un PDF. Le projet remplace le prototype
notebook par une commande reproductible : extraction du PDF, découpage,
recherche locale TF-IDF et synthèse par OpenAI (ou réponse extractive sans API).

## Installation

```powershell
uv sync
```

Créez un fichier `.env` pour la synthèse générative :

```text
OPENAI_API_KEY=...
# Facultatif : OPENAI_MODEL=gpt-4.1-mini
```

## Utilisation

Réponse entièrement locale, idéale pour vérifier le projet sans coût ni réseau :

```powershell
uv run projetcrag --offline "Quelle est la mission de ShuriX Tech ?"
```

Avec OpenAI :

```powershell
uv run projetcrag "Quels sont les deux axes majeurs de ShuriX Tech ?"
```

Un autre document peut être fourni avec `--pdf chemin/vers/document.pdf`. Les
pages citées permettent de vérifier la réponse dans le document source.
