"""
config.py — paramètres PARTAGÉS par les 5 personnes.

Tout le monde importe ce fichier : si le groupe change un paramètre
(taille des chunks, modèle d'embeddings, modèle LLM...), on ne le change
qu'ici et tout le projet reste cohérent.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Charge automatiquement le fichier .env (clé API) s'il existe.
try:
    from dotenv import load_dotenv  # pip install python-dotenv (optionnel)

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

# ----------------------------------------------------------------------------
# Chemins
# ----------------------------------------------------------------------------
P1_DIR = ROOT / "perso1_corpus"
DATA_DIR = P1_DIR / "data"
RAW_CSV = DATA_DIR / "dataset_nettoye.csv"        # fichier fourni (1 ligne = 1 page de PDF)
CLEAN_CSV = DATA_DIR / "corpus_clean.csv"         # sortie de P1 -> entrée de P2

P2_DIR = ROOT / "perso2_indexation"
CHROMA_DIR = P2_DIR / "chroma_db"                 # base vectorielle (sortie de P2)
INDEX_INFO = P2_DIR / "index_info.json"           # paramètres utilisés pour l'index

P3_DIR = ROOT / "perso3_rag"

P4_DIR = ROOT / "perso4_eval_set"
EVAL_CSV = P4_DIR / "eval_set.csv"                # sortie de P4 -> entrée de P5

P5_DIR = ROOT / "perso5_evaluation"
RESULTS_DIR = P5_DIR / "resultats"

# ----------------------------------------------------------------------------
# Découpage (P2) — valeurs par défaut, à justifier dans le rapport
# ----------------------------------------------------------------------------
CHUNK_SIZE = 600       # caractères max par chunk
CHUNK_OVERLAP = 100    # caractères de chevauchement entre 2 chunks consécutifs

# ----------------------------------------------------------------------------
# Embeddings (P2 + P3 : DOIT être le même modèle à l'indexation et à la requête)
# ----------------------------------------------------------------------------
EMBED_MODEL = "intfloat/multilingual-e5-small"    # multilingue, léger (~120 Mo)
# "e5"    = vrai modèle (à utiliser pour le rendu final)
# "dummy" = embeddings par hachage de mots, SANS téléchargement : uniquement
#           pour tester que le code tourne sans internet / sans GPU.
EMBED_BACKEND = os.getenv("EMBED_BACKEND", "e5")

COLLECTION_NAME = "cybersecurite_fr"

# ----------------------------------------------------------------------------
# Retrieval + génération (P3)
# ----------------------------------------------------------------------------
TOP_K = 5

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
# "openrouter/free" = routeur d'OpenRouter qui choisit automatiquement un modèle
# gratuit disponible. Pratique pour démarrer, mais le modèle change d'un appel à
# l'autre -> pour l'ÉVALUATION (P5), fixez un modèle précis (voir README P3).
DEFAULT_MODEL = os.getenv("OPENROUTER_MODEL", "openrouter/free")
# Modèles essayés dans l'ordre si le premier échoue (limite atteinte, retiré...).
FALLBACK_MODELS: list[str] = []
