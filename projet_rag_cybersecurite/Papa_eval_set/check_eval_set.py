"""
PERSO 4 — check_eval_set.py
===========================
Vérifie que eval_set.csv est SOLIDE avant de le donner à P5. Une question mal
construite (mauvaise page attendue, mot-clé introuvable...) fausserait toutes
les métriques sans que personne ne comprenne pourquoi.

Pour chaque question AVEC réponse :
  * les pages attendues (ids_attendus) existent dans data/corpus_clean.csv ;
  * chaque mot-clé de `mots_cles_passage` apparaît bien dans le texte de ces
    pages (preuve que la réponse est réellement dans le passage cité).
Pour chaque question SANS réponse :
  * aucun des `mots_absents` n'apparaît dans TOUT le corpus (preuve que la
    question est vraiment hors corpus).
Affiche aussi la répartition (difficulté, documents) pour vérifier l'équilibre.

Utilisation :
    python perso4_eval_set/check_eval_set.py
"""
from __future__ import annotations

import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

REQUIRED = ["id_question", "question", "reponse_reference", "titre_document", "pages_attendues",
            "ids_attendus", "mots_cles_passage", "mots_absents", "difficulte", "type_question"]


def norm(s: str) -> str:
    """Minuscules, sans accents, apostrophes et espaces normalisés (robuste aux artefacts PDF)."""
    s = unicodedata.normalize("NFKC", str(s)).lower().replace("’", "'").replace("`", "'")
    s = "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", s)


def split(cell) -> list[str]:
    return [] if pd.isna(cell) or not str(cell).strip() else [x.strip() for x in str(cell).split(";") if x.strip()]


def main() -> int:
    ev = pd.read_csv(config.EVAL_CSV)
    corpus = pd.read_csv(config.CLEAN_CSV).set_index("id")
    full_text = norm(" ".join(corpus["texte"]))
    problems: list[str] = []

    missing_cols = [c for c in REQUIRED if c not in ev.columns]
    if missing_cols:
        print("Colonnes manquantes :", missing_cols)
        return 1
    if ev["id_question"].duplicated().any():
        problems.append("id_question en double")

    for _, r in ev.iterrows():
        qid = r["id_question"]
        ids = [int(x) for x in split(r["ids_attendus"])]
        if r["difficulte"] == "sans_reponse":
            for w in split(r["mots_absents"]):
                if norm(w) in full_text:
                    problems.append(f"{qid}: « {w} » est présent dans le corpus -> la question a peut-être une réponse")
            if ids:
                problems.append(f"{qid}: une question sans réponse ne doit pas avoir d'ids_attendus")
            continue
        if not ids:
            problems.append(f"{qid}: question avec réponse mais sans ids_attendus")
            continue
        unknown = [i for i in ids if i not in corpus.index]
        if unknown:
            problems.append(f"{qid}: ids introuvables dans le corpus nettoyé : {unknown}")
            continue
        text = norm(" ".join(corpus.loc[ids, "texte"]))
        for kw in split(r["mots_cles_passage"]):
            if norm(kw) not in text:
                problems.append(f"{qid}: mot-clé « {kw} » absent des pages {ids}")

    print(f"{len(ev)} questions | difficultés : {ev['difficulte'].value_counts().to_dict()}")
    print("Documents visés :", ev["titre_document"].value_counts().to_dict())
    if problems:
        print(f"\n❌ {len(problems)} problème(s) :")
        for p in problems:
            print("  -", p)
        return 1
    print("\n✅ Jeu d'évaluation cohérent : toutes les réponses sont retrouvables dans les pages indiquées.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
