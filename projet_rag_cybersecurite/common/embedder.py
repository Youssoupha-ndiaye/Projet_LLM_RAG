"""
common/embedder.py — transforme du texte en vecteurs (embeddings).

Deux backends :
  * "e5"    : intfloat/multilingual-e5-small via sentence-transformers.
              Ce modèle EXIGE des préfixes : "passage: " pour les documents et
              "query: " pour les questions (sinon la qualité baisse).
  * "dummy" : hachage de mots (aucun téléchargement). Sert uniquement à tester
              le pipeline hors-ligne. Ne PAS l'utiliser pour le rapport final.

Les vecteurs sont normalisés (norme 1) : avec la distance cosinus de ChromaDB,
plus la distance est petite, plus le passage est proche de la question.
"""
from __future__ import annotations

import re
import sys
import unicodedata
import zlib
from functools import lru_cache
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402


def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


class Embedder:
    def __init__(self, backend: str | None = None, model_name: str | None = None):
        self.backend = (backend or config.EMBED_BACKEND).lower()
        self.model_name = model_name or config.EMBED_MODEL
        self._model = None
        if self.backend == "e5":
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:  # message clair pour les étudiants
                raise ImportError(
                    "sentence-transformers n'est pas installé : "
                    "pip install sentence-transformers"
                ) from e
            print(f"[embedder] chargement de {self.model_name} (1er lancement = téléchargement)...")
            self._model = SentenceTransformer(self.model_name)
        elif self.backend == "dummy":
            self.dim = 1024
        else:
            raise ValueError(f"Backend inconnu : {self.backend!r} (utiliser 'e5' ou 'dummy')")

    # ------------------------------------------------------------------ dummy
    def _dummy(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            words = re.findall(r"[a-z0-9]+", _strip_accents(t.lower()))
            for w in words:
                if len(w) < 3:
                    continue
                out[i, zlib.crc32(w.encode()) % self.dim] += 1.0
        out = np.log1p(out)
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms

    # ----------------------------------------------------------------- public
    def embed_passages(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        if self.backend == "dummy":
            return self._dummy(texts)
        return self._model.encode(
            [f"passage: {t}" for t in texts],
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 64,
        )

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        if self.backend == "dummy":
            return self._dummy(texts)
        return self._model.encode([f"query: {t}" for t in texts], normalize_embeddings=True)


@lru_cache(maxsize=4)
def get_embedder(backend: str | None = None, model_name: str | None = None) -> Embedder:
    """Charge le modèle une seule fois (il est lourd) puis le réutilise."""
    return Embedder(backend, model_name)
