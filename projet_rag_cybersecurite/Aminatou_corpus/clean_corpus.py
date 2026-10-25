"""
PERSO 1 — clean_corpus.py
=========================
Objectif : partir de data/dataset_nettoye.csv (1 ligne = 1 page de PDF) et
produire data/corpus_clean.csv, un texte vraiment propre pour ne pas polluer
les chunks (en-têtes répétés, numéros de page, menus latéraux...).

Pourquoi c'est important : si l'en-tête "Introduction Attaques Pare-feux
Détection d'intrusion Authentification" est répété sur 109 pages du cours, il
sera présent dans presque tous les chunks de ce cours. Les embeddings de tous
ces chunks se ressembleront à cause de cet en-tête (et non du contenu) et le
retrieval sera moins précis.

Étapes :
  1. Normalisation Unicode (ligatures, espaces insécables, caractères de contrôle).
  2. Détection AUTOMATIQUE des lignes répétées dans un même document
     (en-têtes, pieds de page, menus) : une ligne qui apparaît sur >= 10 % des
     pages d'un document est considérée comme du bruit. Les chiffres sont
     remplacés par '#' pour reconnaître "12 / 127", "13 / 127", etc.
  3. Suppression des numéros de page isolés ("96 / 127").
  4. Recollage des mots coupés en fin de ligne ("gou- vernance" -> "gouvernance"),
     numéros de page romains, code "GE.21-14702" des pieds de page ONU,
     espaces parasites avant une apostrophe ("D ’UN" -> "D’UN").
  5. Suppression des pages (quasi) vides après nettoyage (< 60 caractères :
     diapos qui ne contenaient qu'une image, pages de titre vides...).
  6. Complétion des métadonnées manquantes ("Non précisé").

Utilisation :
    python perso1_corpus/clean_corpus.py
    python perso1_corpus/clean_corpus.py --min-chars 80 --repeat-threshold 0.15
"""
from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

PAGE_NUMBER_RE = re.compile(r"^\s*(?:page\s*)?\d+\s*(?:/|sur)\s*\d+\s*$", re.IGNORECASE)
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
HYPHEN_BREAK_RE = re.compile(r"([a-zà-ÿ]{2,})- ([a-zà-ÿ]{2,})")
ROMAN_RE = re.compile(r"^\s*(x{0,3})(ix|iv|v?i{0,3})\s*$", re.IGNORECASE)   # numéros de page romains (iv, xii...)
UN_CODE_RE = re.compile(r"\bGE\.\d{2}-\d{4,6}\b")                           # code de pied de page ONU
SPACED_APOSTROPHE_RE = re.compile(r"(\w) ([’'])(\w)")                          # "D ’UN" -> "D’UN" (artefact PDF)


def line_key(line: str) -> str:
    """Clé de comparaison d'une ligne : minuscules + chiffres -> '#'."""
    return re.sub(r"\d+", "#", re.sub(r"\s+", " ", line.strip().lower()))


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text))   # ligatures « ﬁ » -> « fi », NBSP -> espace
    text = CONTROL_RE.sub(" ", text)
    text = text.replace("\r", "\n")
    return text


def find_boilerplate(pages: list[str], threshold: float, min_pages: int = 5, max_words: int = 15) -> set[str]:
    """Retourne les clés des lignes répétées dans >= threshold des pages du document."""
    if len(pages) < min_pages:
        return set()
    counter: Counter = Counter()
    for p in pages:
        keys = {line_key(l) for l in p.split("\n") if l.strip()}
        counter.update(keys)
    limit = max(min_pages, int(threshold * len(pages)))
    return {k for k, n in counter.items() if n >= limit and 0 < len(k.split()) <= max_words}


def clean_page(text: str, boilerplate: set[str], fix_hyphens: bool = True) -> str:
    kept = []
    for line in text.split("\n"):
        if not line.strip():
            kept.append("")
            continue
        line = UN_CODE_RE.sub("", line)
        if not line.strip() or PAGE_NUMBER_RE.match(line) or ROMAN_RE.match(line) or line_key(line) in boilerplate:
            continue
        # numéro de page collé en fin de ligne : "... texte 96 / 127"
        line = re.sub(r"\s+\d+\s*/\s*\d+\s*$", "", line)
        kept.append(line.strip())
    out = "\n".join(kept)
    if fix_hyphens:
        out = HYPHEN_BREAK_RE.sub(r"\1\2", out)
    out = SPACED_APOSTROPHE_RE.sub(r"\1\2\3", out)
    out = re.sub(r"[ \t]+", " ", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=str(config.RAW_CSV))
    ap.add_argument("--output", default=str(config.CLEAN_CSV))
    ap.add_argument("--min-chars", type=int, default=60, help="pages plus courtes (après nettoyage) = supprimées")
    ap.add_argument("--repeat-threshold", type=float, default=0.10,
                    help="part de pages d'un document au-delà de laquelle une ligne est jugée répétitive")
    ap.add_argument("--no-hyphen-fix", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    print(f"Lu : {len(df)} pages, {df['titre'].nunique()} documents")

    df["texte"] = df["texte"].fillna("").map(normalize_text)
    report_lines = ["# Rapport de nettoyage du corpus\n"]
    cleaned_parts = []
    total_before = int(df["texte"].str.len().sum())

    for titre, g in df.groupby("titre", sort=False):
        boiler = find_boilerplate(g["texte"].tolist(), args.repeat_threshold)
        g = g.copy()
        g["texte"] = g["texte"].map(lambda t: clean_page(t, boiler, not args.no_hyphen_fix))
        before = len(g)
        g = g[g["texte"].str.len() >= args.min_chars]
        cleaned_parts.append(g)

        report_lines.append(f"## {titre}\n")
        report_lines.append(f"- Pages : {before} -> {len(g)} conservées ({before - len(g)} supprimées car presque vides)")
        report_lines.append(f"- Lignes répétitives détectées et supprimées : {len(boiler)}")
        for k in sorted(boiler)[:12]:
            report_lines.append(f"    - `{k[:90]}`")
        report_lines.append("")
        print(f"  {titre[:55]:55s} pages {before:4d} -> {len(g):4d} | lignes parasites : {len(boiler)}")

    out = pd.concat(cleaned_parts, ignore_index=True)
    for col, default in [("auteur", "Non précisé"), ("organisme", "Non précisé"), ("date_publication", "Non précisée")]:
        out[col] = out[col].fillna(default)
    out["nb_caracteres"] = out["texte"].str.len()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    total_after = int(out["texte"].str.len().sum())

    report_lines.insert(1, f"- Caractères avant : {total_before:,} / après : {total_after:,} "
                           f"({100 * (1 - total_after / total_before):.1f} % de bruit retiré)\n")
    report = Path(args.output).parent / "rapport_nettoyage.md"
    report.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"\nÉcrit : {args.output}  ({len(out)} pages)\nRapport : {report}")


if __name__ == "__main__":
    main()
