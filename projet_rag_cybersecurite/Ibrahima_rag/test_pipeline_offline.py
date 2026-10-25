"""
PERSO 3 — test_pipeline_offline.py
==================================
Vérifie que TOUT le pipeline fonctionne SANS clé API ni internet pour le LLM :
l'appel à OpenRouter est remplacé par une fausse réponse. Utile pour déboguer
avant de consommer vos requêtes gratuites.

Ce que le test contrôle :
  1. la recherche (retrieval) renvoie bien k passages avec métadonnées ;
  2. le prompt contient les passages numérotés et les consignes anti-hallucination ;
  3. les citations [n] de la réponse sont bien reliées aux passages ;
  4. la détection du « je ne sais pas » fonctionne ;
  5. l'interface (app.py) sait traiter une question.

Utilisation :
    python perso3_rag/test_pipeline_offline.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import rag as rag_module  # noqa: E402
from rag import RAG, build_messages, cited_indexes, is_abstention  # noqa: E402


def fake_chat(messages, api_key, model=None, fallbacks=None, temperature=0.0, **kw):
    """Fausse réponse du LLM : cite le passage [1] (ou s'abstient si le contexte est vide)."""
    assert "UNIQUEMENT" in messages[0]["content"], "le prompt système doit imposer de rester dans les passages"
    assert "[1]" in messages[1]["content"], "les passages doivent être numérotés"
    return {"content": "Selon le document, voir le premier passage [1][2].", "model": "modele/factice"}


def main() -> None:
    rag = RAG()
    q = "Quelles sont les trois clés d'un bon mot de passe ?"

    passages = rag.retrieve(q, k=5)
    assert len(passages) == 5 and all("page" in p["metadata"] for p in passages)
    print("1) retrieval OK — meilleur passage :", passages[0]["metadata"]["titre"], "p.", passages[0]["metadata"]["page"])

    msgs = build_messages(q, passages)
    assert msgs[1]["content"].count("[") >= 5
    print("2) prompt OK —", len(msgs[1]["content"]), "caractères de contexte")

    assert cited_indexes("A [1] puis B [3][3] et [9]", 5) == [1, 3]
    assert is_abstention("Je ne sais pas : l'information n'est pas présente dans les documents fournis.")
    assert not is_abstention("Les trois clés sont la longueur, le caractère aléatoire et l'unicité [1].")
    print("3-4) citations et détection d'abstention OK")

    with mock.patch.object(rag_module, "chat", fake_chat):
        res = rag.answer(q, api_key="cle-factice", k=5)
        assert res["cited"] == [1, 2] and res["model"] == "modele/factice"
        print("5) RAG.answer OK :", res["answer"])

        import app  # l'interface

        answer, rows, details, used = app.ask("cle-factice", "openrouter/free", 5, q)
        assert rows and len(rows[0]) == 6
        print("6) interface OK :", used)

    print("\n✅ Tous les tests passent. Il ne reste qu'à tester avec une vraie clé API.")


if __name__ == "__main__":
    main()
