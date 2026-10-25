"""
PERSO 2 — chunking.py
=====================
Découpe chaque page du corpus en "chunks" (petits passages) de taille limitée.

Choix de conception (à justifier dans le rapport) :
  * On découpe PAGE PAR PAGE : un chunk ne déborde jamais sur la page suivante.
    Avantage : chaque chunk garde un numéro de page exact -> citations fiables
    et évaluation simple (le "bon passage" = la bonne page).
  * On coupe aux FRONTIÈRES DE PHRASES / de lignes quand c'est possible plutôt
    qu'au milieu d'un mot. Un chunk "Seuls les utilisateurs ayant droit ont ac"
    serait inutilisable par le LLM.
  * CHEVAUCHEMENT (overlap) : les dernières phrases d'un chunk sont répétées au
    début du suivant, pour qu'une information située à la jonction ne soit pas
    perdue.
  * Les pages courtes (diapos) restent en un seul chunk.
"""
from __future__ import annotations

import re

import pandas as pd

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?;:])\s+|\n+")


def split_units(text: str, max_len: int) -> list[str]:
    """Coupe le texte en unités (phrases/lignes) ; une unité trop longue est coupée aux espaces."""
    units = [u.strip() for u in SENTENCE_SPLIT_RE.split(text) if u and u.strip()]
    out: list[str] = []
    for u in units:
        while len(u) > max_len:
            cut = u.rfind(" ", 0, max_len)
            cut = cut if cut > 0 else max_len
            out.append(u[:cut].strip())
            u = u[cut:].strip()
        if u:
            out.append(u)
    return out


def chunk_text(text: str, size: int = 600, overlap: int = 100) -> list[str]:
    """Assemble les unités en chunks de <= `size` caractères avec `overlap` caractères de recouvrement."""
    if overlap >= size:
        raise ValueError("overlap doit être strictement inférieur à size")
    units = split_units(text, size)
    chunks: list[str] = []
    cur: list[str] = []

    def cur_len() -> int:
        return len(" ".join(cur))

    for u in units:
        if cur and cur_len() + 1 + len(u) > size:
            chunks.append(" ".join(cur))
            # on garde en tête du prochain chunk les dernières unités (<= overlap caractères)
            keep: list[str] = []
            total = 0
            for x in reversed(cur):
                if total + len(x) + 1 > overlap:
                    break
                keep.insert(0, x)
                total += len(x) + 1
            cur = keep
        cur.append(u)
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def slugify(s: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", s.lower())
    return s.strip("-")[:30]


def build_chunks(df: pd.DataFrame, size: int, overlap: int) -> list[dict]:
    """Transforme le corpus (1 ligne = 1 page) en liste de chunks avec métadonnées."""
    records: list[dict] = []
    for _, row in df.iterrows():
        for i, ch in enumerate(chunk_text(str(row["texte"]), size, overlap)):
            records.append(
                {
                    "chunk_id": f"{slugify(row['titre'])}-p{int(row['page']):03d}-c{i}",
                    "text": ch,
                    "metadata": {
                        "titre": str(row["titre"]),
                        "auteur": str(row["auteur"]),
                        "organisme": str(row["organisme"]),
                        "date_publication": str(row["date_publication"]),
                        "source": str(row["source"]),
                        "url": str(row["url"]),
                        "page": int(row["page"]),
                        "page_id": int(row["id"]),      # identifiant de la page dans le CSV (utilisé par l'évaluation)
                        "chunk_index": i,
                        "n_chars": len(ch),
                    },
                }
            )
    return records


if __name__ == "__main__":  # mini test : python perso2_indexation/chunking.py
    demo = "Phrase une. Phrase deux est un peu plus longue que la première. " * 12
    for c in chunk_text(demo, 200, 50):
        print(len(c), "|", c[:60], "...")
