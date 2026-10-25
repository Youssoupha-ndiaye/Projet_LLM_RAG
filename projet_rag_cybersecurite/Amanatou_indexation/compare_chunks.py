"""
PERSO 2 — compare_chunks.py   (aide à JUSTIFIER le choix de taille/chevauchement)
================================================================================
Le sujet demande de « tester plusieurs tailles de chunk (ex. 300 à 800 caractères
avec chevauchement) et justifier le choix ».

Ce script construit, pour chaque configuration (taille, chevauchement), un index
temporaire EN MÉMOIRE (la vraie base n'est pas touchée) puis affiche :
  * le nombre de chunks et leur taille moyenne,
  * si le fichier d'évaluation de P4 existe : le hit@k, c'est-à-dire la part des
    questions pour lesquelles la bonne page est dans les k premiers résultats.

Pendant que P4 n'a pas fini, le script ne donne que les statistiques de
découpage : relancez-le dès que perso4_eval_set/eval_set.csv est disponible.

Le tableau obtenu (comparaison_chunks.csv + version Markdown affichée) est à
coller dans le rapport pour justifier la configuration retenue.

Utilisation :
    python perso2_indexation/compare_chunks.py
    python perso2_indexation/compare_chunks.py --configs 300:50 500:100 800:150 --k 5
"""
from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

import chromadb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from chunking import build_chunks  # noqa: E402
from common.embedder import get_embedder  # noqa: E402
from index import build_collection  # noqa: E402


def expected_ids(cell) -> set[int]:
    if pd.isna(cell) or str(cell).strip() == "":
        return set()
    return {int(x) for x in str(cell).replace(",", ";").split(";") if x.strip()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", nargs="+", default=["300:50", "500:100", "600:100", "800:150"],
                    help="liste taille:chevauchement")
    ap.add_argument("--k", type=int, default=config.TOP_K)
    ap.add_argument("--backend", default=config.EMBED_BACKEND, choices=["e5", "dummy"])
    args = ap.parse_args()

    df = pd.read_csv(config.CLEAN_CSV)
    embedder = get_embedder(args.backend)

    questions = None
    if config.EVAL_CSV.exists():
        ev = pd.read_csv(config.EVAL_CSV)
        questions = ev[ev["ids_attendus"].notna()].copy()       # on ignore les questions sans réponse
        questions["_exp"] = questions["ids_attendus"].map(expected_ids)
        print(f"Jeu d'évaluation trouvé : {len(questions)} questions avec réponse dans le corpus.")
        qvecs = embedder.embed_queries(questions["question"].tolist())
    else:
        print("eval_set.csv introuvable : seules les statistiques de découpage seront calculées.")

    client = chromadb.EphemeralClient()
    rows = []
    for cfg in args.configs:
        size, overlap = (int(x) for x in cfg.split(":"))
        chunks = build_chunks(df, size, overlap)
        lengths = [c["metadata"]["n_chars"] for c in chunks]
        row = {"taille": size, "chevauchement": overlap, "n_chunks": len(chunks),
               "taille_moyenne": round(sum(lengths) / len(lengths))}
        if questions is not None:
            col = build_collection(client, f"cmp_{uuid.uuid4().hex[:8]}", chunks, embedder)
            res = col.query(query_embeddings=qvecs.tolist(), n_results=args.k, include=["metadatas"])
            hits, mrr = 0, 0.0
            for exp, metas in zip(questions["_exp"], res["metadatas"]):
                got = [m["page_id"] for m in metas]
                rank = next((i + 1 for i, g in enumerate(got) if g in exp), None)
                if rank:
                    hits += 1
                    mrr += 1 / rank
            row[f"hit@{args.k}"] = round(hits / len(questions), 3)
            row["MRR"] = round(mrr / len(questions), 3)
        rows.append(row)
        print(row)

    out = pd.DataFrame(rows)
    out_path = Path(__file__).resolve().parent / "comparaison_chunks.csv"
    out.to_csv(out_path, index=False)
    print()
    try:
        print(out.to_markdown(index=False))      # nécessite `pip install tabulate` (optionnel)
    except ImportError:
        print(out.to_string(index=False))
    print(f"\nSauvegardé : {out_path}")


if __name__ == "__main__":
    main()
