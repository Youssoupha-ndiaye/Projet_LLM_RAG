# Projet RAG — Cybersécurité (LLM & Écosystème)

Système de **question-réponse sur 5 documents publics en français** (cybersécurité), avec
**passages cités**, réponse « je ne sais pas » hors corpus, et **évaluation chiffrée** sur 30 questions.

```
question ─► embedding (multilingual-e5-small) ─► ChromaDB (top-k) ─► prompt ─► LLM via OpenRouter ─► réponse + sources
```

## Corpus (déjà fourni : `dataset_nettoye.csv`)

| Document | Pages |
|---|---|
| Introduction à la Cybersécurité (R. Kanawati, Sorbonne Paris Nord, 2023) | 132 |
| Guide pour la bonne gouvernance de la cybersécurité (DCAF, 2019) | 104 |
| La cybersécurité dans les entités des Nations Unies (CCI, 2021) | 112 |
| Les fondamentaux de la cybersécurité (CSC, 2017) | 17 |
| Manuel de cybersécurité pour les partis politiques (NDI) | 86 |


`config.py` (paramètres communs) et `common/` (embeddings + client OpenRouter) sont partagés par tous.
**Chaque dossier contient un `README.md` qui explique chaque script ligne de logique par ligne de logique.**

## Démarrage rapide (ordre d'exécution)

```bash
pip install -r requirements.txt

python Aminatou_corpus/clean_corpus.py            #  -> data/corpus_clean.csv
python Aminatou_corpus/build_sources.py --date-consultation 2026-10-XX   # P1 -> SOURCES.md

python Amanatou_indexation/index.py --reset       #   -> chroma_db/  (télécharge e5 la 1re fois)
python Amanatou_indexation/compare_chunks.py      #   -> tableau pour justifier la taille des chunks

python Papa_eval_set/check_eval_set.py        #  -> vérifie eval_set.csv

python Ibrahima_rag/app.py                        #   -> interface : collez votre clé OpenRouter

python Youssoupha_evaluation/evaluate.py --retrieval-only          # -> sans API
python Youssoupha_evaluation/evaluate.py --model <modele:free> --judge --resume   # P5 -> complet
```

### Clé API OpenRouter (pour interroger le modèle)

1. Créez un compte sur <https://openrouter.ai> puis une clé sur <https://openrouter.ai/keys> (`sk-or-v1-...`).
2. **Soit** vous la collez dans le champ « 🔑 Clé API » de l'interface (`perso3_rag/app.py`),
   **soit** vous copiez `.env.example` en `.env` et y mettez la clé (pour les scripts en ligne de commande).
3. Modèle par défaut : `openrouter/free` (OpenRouter choisit lui-même un modèle gratuit disponible).
   Détails et conseils dans `perso3_rag/README.md`.

### Sur Google Colab

```python
!unzip -q projet_rag_cybersecurite.zip
%cd projet_rag_cybersecurite
!pip install -q -r requirements.txt
!python perso2_indexation/index.py --reset
!python perso3_rag/app.py --share        # ouvre un lien public temporaire
```

## Ce qui a été testé

Le pipeline complet (nettoyage → index ChromaDB → retrieval → prompt → client OpenRouter → interface Gradio →
évaluation avec reprise) a été exécuté avec un **faux LLM** et un backend d'embeddings « de test » (`EMBED_BACKEND=dummy`).
**Non testé ici :** le vrai modèle `multilingual-e5-small` (téléchargement Hugging Face) et un vrai appel OpenRouter —
à valider chez vous avec votre clé. Les métriques du rapport doivent venir de **votre** exécution avec e5 + un vrai LLM.
