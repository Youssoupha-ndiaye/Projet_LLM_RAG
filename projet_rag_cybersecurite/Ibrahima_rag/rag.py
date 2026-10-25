"""
PERSO 3 — rag.py   (livrable : script RAG)
==========================================
Pipeline complet de question-réponse :

    question ──► embedding ──► ChromaDB (top-k passages) ──► prompt ──► LLM (OpenRouter)
                                                                           │
                         réponse + passages cités ◄───────────────────────┘

Trois fonctions à retenir :
  * RAG.retrieve()      : cherche les k passages les plus proches de la question.
  * build_messages()    : construit le prompt (consignes + passages numérotés).
  * RAG.answer()        : enchaîne tout et renvoie réponse + sources.

Le prompt IMPOSE au modèle de :
  1. répondre uniquement à partir des passages fournis,
  2. citer les passages avec [1], [2]...,
  3. répondre « Je ne sais pas » si l'information n'y est pas.

Utilisation en ligne de commande :
    export OPENROUTER_API_KEY="sk-or-..."
    python perso3_rag/rag.py "Quelles sont les trois clés d'un bon mot de passe ?"
    python perso3_rag/rag.py "..." --k 8 --model openrouter/free
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import chromadb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from common.embedder import get_embedder  # noqa: E402
from common.openrouter_client import OpenRouterError, chat  # noqa: E402

ABSTENTION_SENTENCE = "Je ne sais pas : l'information n'est pas présente dans les documents fournis."

SYSTEM_PROMPT = f"""Tu es un assistant spécialisé en cybersécurité. Tu réponds en français.

RÈGLES STRICTES :
1. Réponds UNIQUEMENT à partir des passages numérotés fournis dans le contexte. N'utilise aucune connaissance extérieure, même si tu connais la réponse.
2. Cite les passages utilisés avec leur numéro entre crochets, par exemple [1] ou [2][3], à la fin de chaque affirmation.
3. Si les passages ne contiennent pas l'information nécessaire pour répondre, réponds exactement : « {ABSTENTION_SENTENCE} » et rien d'autre.
4. Sois concis (6 phrases maximum) et factuel. N'invente aucun chiffre, nom ou définition."""

_ABSTAIN_RE = re.compile(r"je ne sais pas|ne (?:sais|peux) pas répondre|pas (?:présente?|mentionnée?|disponible)s? dans les (?:documents|passages)", re.I)


def is_abstention(answer: str) -> bool:
    """Vrai si la réponse du modèle est un « je ne sais pas » (utilisé par P5)."""
    return bool(_ABSTAIN_RE.search(answer or ""))


def build_messages(question: str, passages: list[dict]) -> list[dict]:
    """Construit les messages envoyés au LLM : consignes + passages numérotés + question."""
    blocs = []
    for i, p in enumerate(passages, start=1):
        m = p["metadata"]
        blocs.append(f"[{i}] (Document : {m['titre']} — page {m['page']})\n{p['text']}")
    context = "\n\n".join(blocs)
    user = f"PASSAGES :\n{context}\n\nQUESTION : {question}\n\nRÉPONSE (avec citations [n]) :"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def cited_indexes(answer: str, n_passages: int) -> list[int]:
    """Numéros de passages (1..n) cités dans la réponse, ex. 'texte [1][3]' -> [1, 3]."""
    found = {int(x) for x in re.findall(r"\[(\d{1,2})\]", answer or "")}
    return sorted(i for i in found if 1 <= i <= n_passages)


class RAG:
    def __init__(self, db_dir: str | Path = config.CHROMA_DIR, collection: str = config.COLLECTION_NAME):
        if not Path(db_dir).exists():
            raise FileNotFoundError(
                f"Base vectorielle introuvable ({db_dir}). Lancez d'abord : python perso2_indexation/index.py --reset"
            )
        client = chromadb.PersistentClient(path=str(db_dir))
        try:
            self.col = client.get_collection(collection)
        except Exception as e:
            raise FileNotFoundError(
                f"Collection « {collection} » introuvable. Lancez : python perso2_indexation/index.py --reset"
            ) from e
        meta = self.col.metadata or {}
        # On réutilise EXACTEMENT le modèle d'embeddings utilisé à l'indexation.
        self.embedder = get_embedder(meta.get("embed_backend"), meta.get("embed_model"))

    def retrieve(self, question: str, k: int = config.TOP_K) -> list[dict]:
        vec = self.embedder.embed_queries([question])
        res = self.col.query(query_embeddings=vec.tolist(), n_results=k,
                             include=["documents", "metadatas", "distances"])
        return [
            {"text": d, "metadata": m, "distance": float(dist), "score": round(1 - float(dist), 4)}
            for d, m, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
        ]

    def answer(self, question: str, api_key: str, model: str | None = None, k: int = config.TOP_K,
               fallbacks: list[str] | None = None, temperature: float = 0.0) -> dict:
        """Retourne {answer, model, passages, cited, abstained}. Lève OpenRouterError en cas d'échec LLM."""
        passages = self.retrieve(question, k)
        out = chat(build_messages(question, passages), api_key=api_key, model=model,
                   fallbacks=fallbacks, temperature=temperature)
        return {
            "question": question,
            "answer": out["content"],
            "model": out["model"],
            "passages": passages,
            "cited": cited_indexes(out["content"], len(passages)),
            "abstained": is_abstention(out["content"]),
        }


def format_sources(result: dict) -> str:
    """Texte lisible des sources (pour la console). Les passages cités sont marqués ✔."""
    lines = []
    for i, p in enumerate(result["passages"], start=1):
        m = p["metadata"]
        mark = "✔" if i in result["cited"] else " "
        lines.append(f"{mark} [{i}] {m['titre']} — p.{m['page']} (score {p['score']:.2f})")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("question")
    ap.add_argument("--k", type=int, default=config.TOP_K)
    ap.add_argument("--model", default=config.DEFAULT_MODEL)
    ap.add_argument("--api-key", default=config.OPENROUTER_API_KEY, help="sinon variable OPENROUTER_API_KEY")
    ap.add_argument("--show-passages", action="store_true", help="affiche aussi le texte des passages")
    ap.add_argument("--retrieval-only", action="store_true", help="n'appelle pas le LLM (test de la recherche)")
    args = ap.parse_args()

    rag = RAG()
    if args.retrieval_only:
        for i, p in enumerate(rag.retrieve(args.question, args.k), start=1):
            m = p["metadata"]
            print(f"[{i}] {m['titre']} p.{m['page']} (score {p['score']:.2f})\n    {p['text'][:200]}...\n")
        return
    try:
        res = rag.answer(args.question, args.api_key, args.model, args.k)
    except OpenRouterError as e:
        sys.exit(f"Erreur LLM : {e}")
    print("\nRÉPONSE (modèle : %s)\n%s\n\nSOURCES\n%s" % (res["model"], res["answer"], format_sources(res)))
    if args.show_passages:
        print("\n" + json.dumps([p["text"] for p in res["passages"]], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
