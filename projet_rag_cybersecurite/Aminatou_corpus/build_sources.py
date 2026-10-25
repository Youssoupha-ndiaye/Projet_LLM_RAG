"""
PERSO 1 — build_sources.py
==========================
Génère SOURCES.md (livrable de P1) à partir de data/corpus_clean.csv :
une fiche par document (titre, auteur, organisme, date, URL, fichier, nombre
de pages) + la date de consultation demandée par le sujet.

Utilisation :
    python perso1_corpus/build_sources.py --date-consultation 2026-10-19

⚠️ Mettez la VRAIE date à laquelle vous avez téléchargé les PDF.
   Le rapport final doit citer toutes les sources.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(config.CLEAN_CSV))
    ap.add_argument("--output", default=str(config.P1_DIR / "SOURCES.md"))
    ap.add_argument("--date-consultation", default="À RENSEIGNER (AAAA-MM-JJ)")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    lines = [
        "# SOURCES — Corpus « Cybersécurité » (documents publics en français)\n",
        "**Sous-thème retenu :** fondamentaux de la cybersécurité, bonnes pratiques "
        "(mots de passe, authentification, hameçonnage), réseaux (pare-feux, IDS, cryptographie) "
        "et gouvernance.\n",
        f"**Nombre de documents :** {df['titre'].nunique()}  |  **Pages exploitables :** {len(df)}  "
        f"|  **Caractères :** {int(df['texte'].str.len().sum()):,}\n",
        "**Traitement :** texte extrait des PDF page par page, puis nettoyé "
        "(`perso1_corpus/clean_corpus.py` : en-têtes/pieds de page, numéros de page, menus retirés).\n",
        "---\n",
    ]
    for n, (titre, g) in enumerate(df.groupby("titre", sort=False), start=1):
        r = g.iloc[0]
        lines += [
            f"## {n}. {titre}\n",
            f"- **Auteur(s) :** {r['auteur']}",
            f"- **Organisme :** {r['organisme']}",
            f"- **Date de publication :** {r['date_publication']}",
            f"- **URL :** {r['url']}",
            f"- **Fichier local :** `{r['source']}`",
            f"- **Pages conservées :** {len(g)} (pages {int(g['page'].min())} à {int(g['page'].max())})",
            f"- **Date de consultation :** {args.date_consultation}",
            "",
        ]
    Path(args.output).write_text("\n".join(lines), encoding="utf-8")
    print(f"Écrit : {args.output}")


if __name__ == "__main__":
    main()
