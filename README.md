# ShuriX RAG local

Application de questions-réponses sur PDF, exécutable sans clé API ni crédit.
Elle utilise uniquement `sentence-transformers/all-MiniLM-L6-v2` pour créer des
embeddings et rechercher les passages les plus proches d'une question.

`all-MiniLM-L6-v2` est un modèle de recherche sémantique : il ne génère pas de
texte comme un LLM. L'application fournit donc une réponse extractive, appuyée
par des passages et des pages sources vérifiables. Aucune donnée n'est envoyée
à OpenAI ou à un autre service d'IA.

## Installation

Python 3.14 est requis.

```powershell
uv sync
```

Au premier lancement, les poids de `all-MiniLM-L6-v2` doivent être disponibles
dans le cache Hugging Face. Pour préparer une machine connectée :

```powershell
uv run python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
```

Les exécutions suivantes sont locales.

## Application web

```powershell
uv run streamlit run streamlit_app.py
```

Ouvrez l'adresse affichée par Streamlit. Le PDF `doc/shuri.pdf` est sélectionné
par défaut ; tout autre PDF peut être importé dans la barre latérale.

## Ligne de commande

```powershell
uv run projetcrag "Quelle est la mission de ShuriX Tech ?"
```

Pour changer de document :

```powershell
uv run projetcrag --pdf chemin/vers/document.pdf "Votre question"
```

## Structure

- `src/projetcrag/rag.py` : extraction, découpage et recherche sémantique locale.
- `streamlit_app.py` : interface prête au déploiement Streamlit.
- `agentic_rag.ipynb` : prototype initial ; l'application de production est le
  fichier Streamlit.

## Déploiement Streamlit Community Cloud

1. Publiez ce dépôt sur GitHub.
2. Dans Streamlit Community Cloud, choisissez ce dépôt et la branche voulue.
3. Indiquez `streamlit_app.py` comme fichier principal.
4. Déployez. Aucune variable secrète n'est nécessaire.

Le premier démarrage télécharge les poids du modèle si le cache n'est pas déjà
présent ; les démarrages suivants les réutilisent.
