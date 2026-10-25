"""
PERSO 2 — index.py   (livrable : script d'indexation)
=====================================================
Construit la base vectorielle ChromaDB :

    corpus_clean.csv  ->  chunks  ->  embeddings (multilingual-e5-small)  ->  ChromaDB

Chaque chunk est stocké avec ses métadonnées (titre, auteur, URL, page...) pour
pouvoir CITER la source dans la réponse (P3).

Détails importants :
  * On calcule les embeddings nous-mêmes (pas ceux par défaut de Chroma) pour
    contrôler le modèle ET les préfixes "passage: " / "query: " d'e5.
  * Le texte embarqué = "titre du document + chunk" : cela aide à retrouver un
    passage quand la question cite le document ("selon le manuel pour les partis...").
    Le texte STOCKÉ et montré au LLM reste le chunk seul, sans le titre.
  * Distance COSINUS (hnsw:space = cosine) ; vecteurs normalisés.
  * Le backend d'embeddings utilisé est mémorisé dans la collection : P3 et P5
    interrogent ainsi automatiquement avec le même modèle.

Utilisation :
    python perso2_indexation/index.py --reset
    python perso2_indexation/index.py --size 500 --overlap 80 --reset
    EMBED_BACKEND=dummy python perso2_indexation/index.py --reset   # test hors-ligne
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import chromadb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from chunking import build_chunks  # noqa: E402
from common.embedder import get_embedder  # noqa: E402

BATCH = 256


def embedding_text(chunk: dict, with_title: bool) -> str:
    return f"{chunk['metadata']['titre']}. {chunk['text']}" if with_title else chunk["text"]


def build_collection(client, name: str, chunks: list[dict], embedder, with_title: bool = True, reset: bool = True):
    """Crée la collection et y insère tous les chunks (réutilisée par compare_chunks.py)."""
    if reset:
        try:
            client.delete_collection(name)
        except Exception:
            pass
    col = client.get_or_create_collection(
        name,
        metadata={"hnsw:space": "cosine", "embed_backend": embedder.backend, "embed_model": embedder.model_name},
    )
    for i in range(0, len(chunks), BATCH):
        part = chunks[i : i + BATCH]
        vecs = embedder.embed_passages([embedding_text(c, with_title) for c in part])
        col.add(
            ids=[c["chunk_id"] for c in part],
            documents=[c["text"] for c in part],
            metadatas=[c["metadata"] for c in part],
            embeddings=vecs.tolist(),
        )
    return col


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=str(config.CLEAN_CSV))
    ap.add_argument("--db-dir", default=str(config.CHROMA_DIR))
    ap.add_argument("--collection", default=config.COLLECTION_NAME)
    ap.add_argument("--size", type=int, default=config.CHUNK_SIZE, help="taille max d'un chunk (caractères)")
    ap.add_argument("--overlap", type=int, default=config.CHUNK_OVERLAP, help="chevauchement (caractères)")
    ap.add_argument("--backend", default=config.EMBED_BACKEND, choices=["e5", "dummy"])
    ap.add_argument("--no-title", action="store_true", help="ne pas préfixer l'embedding avec le titre du document")
    ap.add_argument("--reset", action="store_true", help="supprime et recrée la collection")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    chunks = build_chunks(df, args.size, args.overlap)
    lengths = [c["metadata"]["n_chars"] for c in chunks]
    print(f"{len(df)} pages -> {len(chunks)} chunks (taille moy. {sum(lengths)/len(lengths):.0f}, "
          f"min {min(lengths)}, max {max(lengths)} caractères)")

    embedder = get_embedder(args.backend)
    client = chromadb.PersistentClient(path=args.db_dir)
    t0 = time.time()
    col = build_collection(client, args.collection, chunks, embedder, not args.no_title, args.reset)
    print(f"Indexation terminée en {time.time() - t0:.0f} s : {col.count()} chunks dans « {args.collection} »")

    info = {
        "collection": args.collection,
        "chunk_size": args.size,
        "chunk_overlap": args.overlap,
        "n_pages": int(len(df)),
        "n_chunks": len(chunks),
        "embed_backend": embedder.backend,
        "embed_model": embedder.model_name,
        "title_in_embedding": not args.no_title,
    }
    config.INDEX_INFO.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Paramètres sauvegardés dans", config.INDEX_INFO)


if __name__ == "__main__":
    main()
