"""
PERSO 3 — app.py   (interface pour interroger le modèle)
========================================================
Petite application web (Gradio) pour poser des questions au RAG :

  1. vous collez votre CLÉ API OpenRouter dans le champ prévu
     (elle reste en mémoire le temps de la session, elle n'est PAS enregistrée) ;
  2. vous choisissez le modèle (par défaut `openrouter/free` = modèle gratuit
     choisi automatiquement ; le bouton « Actualiser » liste les modèles gratuits
     du moment) ;
  3. vous posez votre question : la réponse s'affiche avec les passages cités.

Lancement :
    python perso3_rag/app.py                 # ouvre http://127.0.0.1:7860
    python perso3_rag/app.py --share         # lien public temporaire (utile sur Google Colab)

Prérequis : la base vectorielle a été construite par P2 (python perso2_indexation/index.py --reset).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import gradio as gr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from common.openrouter_client import OpenRouterError, list_free_models  # noqa: E402
from rag import RAG  # noqa: E402

_rag: RAG | None = None


def get_rag() -> RAG:
    """Charge la base + le modèle d'embeddings une seule fois."""
    global _rag
    if _rag is None:
        _rag = RAG()
    return _rag


def index_status() -> str:
    if not config.INDEX_INFO.exists():
        return "⚠️ Index introuvable : lancez d'abord `python perso2_indexation/index.py --reset`."
    i = json.loads(config.INDEX_INFO.read_text(encoding="utf-8"))
    note = "  \n⚠️ Index de **test** (embeddings `dummy`) : reconstruisez avec `--backend e5` pour de vrais résultats." \
        if i.get("embed_backend") == "dummy" else ""
    return (f"Index prêt : **{i['n_chunks']} passages** · chunks de {i['chunk_size']} caractères "
            f"(chevauchement {i['chunk_overlap']}) · embeddings `{i['embed_backend']}`{note}")


def refresh_models(api_key: str, current: str):
    try:
        models = list_free_models(api_key)
    except OpenRouterError as e:
        return gr.Dropdown(), f"❌ {e}"
    value = current if current in models else models[0]
    return gr.Dropdown(choices=models, value=value), f"✅ {len(models)} modèles gratuits trouvés."


def ask(api_key: str, model: str, k: int, question: str):
    api_key = (api_key or config.OPENROUTER_API_KEY or "").strip()
    question = (question or "").strip()
    empty_tbl: list = []
    if not question:
        return "Posez une question.", empty_tbl, "", ""
    if not api_key:
        return "🔑 Collez votre clé API OpenRouter dans le champ de gauche.", empty_tbl, "", ""
    try:
        res = get_rag().answer(question, api_key, model=(model or config.DEFAULT_MODEL), k=int(k))
    except FileNotFoundError as e:
        return f"⚠️ {e}", empty_tbl, "", ""
    except OpenRouterError as e:
        return f"❌ {e}", empty_tbl, "", ""

    rows, details = [], []
    for i, p in enumerate(res["passages"], start=1):
        m = p["metadata"]
        rows.append([i, "✔" if i in res["cited"] else "", m["titre"], m["page"], p["score"],
                     p["text"][:160].replace("\n", " ") + "…"])
        details.append(f"**[{i}] {m['titre']} — page {m['page']}** (score {p['score']:.2f})\n\n{p['text']}\n\n---")
    badge = "ℹ️ *Le système a répondu qu'il ne sait pas.*\n\n" if res["abstained"] else ""
    return badge + res["answer"], rows, "\n\n".join(details), f"Modèle ayant répondu : `{res['model']}`"


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="RAG Cybersécurité") as demo:
        gr.Markdown("# 🛡️ Assistant RAG — Cybersécurité\nPosez une question : la réponse est générée "
                    "**uniquement** à partir des 5 documents du corpus, avec les passages cités.")
        gr.Markdown(index_status())
        with gr.Row():
            with gr.Column(scale=1):
                api_key = gr.Textbox(label="🔑 Clé API OpenRouter", type="password",
                                     placeholder="sk-or-v1-...", info="Créez-la sur openrouter.ai/keys")
                model = gr.Dropdown(label="Modèle", choices=[config.DEFAULT_MODEL], value=config.DEFAULT_MODEL,
                                    allow_custom_value=True,
                                    info="openrouter/free = modèle gratuit choisi automatiquement")
                refresh = gr.Button("🔄 Actualiser la liste des modèles gratuits")
                k = gr.Slider(1, 10, value=config.TOP_K, step=1, label="Nombre de passages (top-k)")
                status = gr.Markdown("")
            with gr.Column(scale=2):
                question = gr.Textbox(label="Votre question", lines=2,
                                      placeholder="Ex. : Quelles sont les trois clés d'un bon mot de passe ?")
                go = gr.Button("Interroger", variant="primary")
                answer = gr.Markdown(label="Réponse")
                used_model = gr.Markdown("")
                sources = gr.Dataframe(headers=["#", "Cité", "Document", "Page", "Score", "Extrait"],
                                       interactive=False, wrap=True, label="Passages récupérés (✔ = cité dans la réponse)")
                with gr.Accordion("Texte complet des passages", open=False):
                    passages = gr.Markdown("")
        gr.Examples(
            examples=["Quelles sont les trois clés d'un bon mot de passe ?",
                      "Quelle différence entre un virus et un ver informatique ?",
                      "Qu'est-ce que la Convention de Budapest ?",
                      "En quoi consiste le modèle Zero Trust ?"],
            inputs=[question])
        refresh.click(refresh_models, [api_key, model], [model, status])
        for trigger in (go.click, question.submit):
            trigger(ask, [api_key, model, k, question], [answer, sources, passages, used_model])
    return demo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true", help="crée un lien public temporaire (Colab)")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()
    build_ui().launch(share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()
