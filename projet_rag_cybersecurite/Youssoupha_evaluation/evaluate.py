"""
PERSO 5 — evaluate.py   (livrable : script d'évaluation)
========================================================
Lance le système RAG sur les 30 questions de P4 et calcule les métriques demandées.

MÉTRIQUES
---------
A. RETRIEVAL (sans LLM, gratuit, instantané) — questions AVEC réponse
   * Hit@k    : la bonne page est-elle dans les k passages récupérés ?  (1 ou 0)
   * Recall@k : part des pages attendues retrouvées (utile quand il y en a plusieurs)
   * MRR      : 1 / rang du premier bon passage (récompense un bon classement)

B. GÉNÉRATION (appelle le LLM) — nécessite une clé API OpenRouter
   * Gestion des questions SANS réponse : le système dit-il « je ne sais pas » ?
     (taux d'abstention correcte) et, à l'inverse, s'abstient-il à tort sur les
     questions qui ont une réponse ? (taux d'abstention abusive)
   * Fidélité (--judge) : un LLM « juge » lit les passages + la réponse et dit si
     chaque affirmation est appuyée par les passages (1 appel LLM par question).
   * Exactitude (--judge) : la réponse est-elle correcte par rapport à la
     réponse de référence ? (dans le même appel du juge)

⚠️ LIMITES DES MODÈLES GRATUITS : OpenRouter limite le nombre de requêtes/jour
des modèles « :free ». 30 questions = 30 appels (+30 avec --judge). Le script
SAUVEGARDE après chaque question et REPREND là où il s'est arrêté (--resume) :
vous pouvez donc le relancer le lendemain si la limite est atteinte.

UTILISATION
-----------
    # 1) Retrieval seul (aucune clé API nécessaire)
    python perso5_evaluation/evaluate.py --retrieval-only

    # 2) Évaluation complète, avec modèle FIXE (reproductible) et juge LLM
    export OPENROUTER_API_KEY="sk-or-..."
    python perso5_evaluation/evaluate.py --model <id_du_modele:free> --judge --resume

    # tester sur 5 questions seulement
    python perso5_evaluation/evaluate.py --judge --limit 5

SORTIES (dossier resultats/)
----------------------------
    results.csv          une ligne par question (réponse, sources, métriques, diagnostic)
    metrics.json         métriques globales + par difficulté
    erreurs_a_classer.csv  cas problématiques avec une catégorie PROPOSÉE
                           (à vérifier et corriger à la main pour le rapport)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "perso3_rag"))
import config  # noqa: E402
from common.openrouter_client import OpenRouterError, chat  # noqa: E402
from rag import RAG, build_messages, is_abstention  # noqa: E402

JUDGE_SYSTEM = """Tu es un évaluateur rigoureux de systèmes de question-réponse. Tu reçois : une QUESTION, des PASSAGES (le seul contexte autorisé), la RÉPONSE du système et une RÉPONSE DE RÉFÉRENCE.
Réponds UNIQUEMENT par un objet JSON, sans texte autour, de la forme :
{"fidele": 0 ou 1, "correct": 0 ou 1, "commentaire": "une phrase"}
- "fidele" = 1 si TOUTES les affirmations de la réponse sont appuyées par les PASSAGES (aucune information inventée ou venant d'ailleurs). Une réponse du type « je ne sais pas » est fidèle.
- "correct" = 1 si la réponse contient l'essentiel de la RÉPONSE DE RÉFÉRENCE sans contradiction. Pour une question dont la référence dit que l'information est absente du corpus, "correct" = 1 uniquement si le système dit qu'il ne sait pas."""


def parse_ids(cell) -> list[int]:
    return [] if pd.isna(cell) or not str(cell).strip() else [int(x) for x in str(cell).split(";") if x.strip()]


def judge(question: str, passages: list[dict], answer: str, reference: str, api_key: str, model: str | None) -> dict:
    ctx = "\n\n".join(f"[{i}] {p['text']}" for i, p in enumerate(passages, 1))
    user = (f"QUESTION : {question}\n\nPASSAGES :\n{ctx}\n\nRÉPONSE DU SYSTÈME :\n{answer}\n\n"
            f"RÉPONSE DE RÉFÉRENCE :\n{reference}")
    out = chat([{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}],
               api_key=api_key, model=model, temperature=0.0, max_tokens=300)
    m = re.search(r"\{.*\}", out["content"], re.S)
    if not m:
        raise ValueError(f"réponse du juge non exploitable : {out['content'][:120]}")
    data = json.loads(m.group(0))
    return {"fidele": int(data.get("fidele", 0)), "correct": int(data.get("correct", 0)),
            "juge_commentaire": str(data.get("commentaire", ""))}


def has_text(v) -> bool:
    """Vrai si la cellule contient du texte (les cellules vides relues depuis un CSV valent NaN)."""
    return isinstance(v, str) and v.strip() != ""


def propose_error_category(r: dict) -> str | None:
    """Catégorie d'erreur PROPOSÉE automatiquement (à relire à la main)."""
    if has_text(r.get("erreur_llm")):
        return "erreur technique LLM (limite de requêtes, modèle indisponible...) : relancer avec --resume"
    sans_rep = r["difficulte"] == "sans_reponse"
    if sans_rep and not r.get("abstention"):
        return "question sans réponse mal gérée (le système répond au lieu de s'abstenir)"
    if not sans_rep and r.get("hit") == 0:
        return "mauvais chunk (retrieval : la bonne page n'est pas dans le top-k)"
    if not sans_rep and r.get("abstention") == 1:
        return "abstention abusive (le système dit « je ne sais pas » alors que la réponse existe)"
    if r.get("fidele") == 0:
        return "hallucination / réponse non appuyée par les passages"
    if r.get("correct") == 0:
        return "réponse incorrecte ou incomplète malgré un bon retrieval (génération)"
    return None


def summarize(df: pd.DataFrame, k: int) -> dict:
    def mean(s):
        s = pd.to_numeric(s, errors="coerce").dropna()
        return round(float(s.mean()), 3) if len(s) else None

    ans = df[df["difficulte"] != "sans_reponse"]
    noans = df[df["difficulte"] == "sans_reponse"]
    gen = df[df["reponse"].notna()]
    m = {
        "n_questions": int(len(df)),
        "n_avec_reponse": int(len(ans)),
        "n_sans_reponse": int(len(noans)),
        "k": k,
        f"hit@{k}": mean(ans["hit"]),
        f"recall@{k}": mean(ans["recall"]),
        "MRR": mean(ans["rr"]),
        "retrieval_par_difficulte": {d: {"hit": mean(g["hit"]), "n": int(len(g))} for d, g in ans.groupby("difficulte")},
    }
    if len(gen):
        m["abstention_correcte_sans_reponse"] = mean(gen[gen["difficulte"] == "sans_reponse"]["abstention"])
        m["abstention_abusive_avec_reponse"] = mean(gen[gen["difficulte"] != "sans_reponse"]["abstention"])
        if "fidele" in gen:
            m["fidelite"] = mean(gen["fidele"])
            m["exactitude"] = mean(gen["correct"])
            m["exactitude_par_difficulte"] = {d: mean(g["correct"]) for d, g in gen.groupby("difficulte")}
    return m


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k", type=int, default=config.TOP_K)
    ap.add_argument("--model", default=config.DEFAULT_MODEL, help="FIXEZ un modèle précis pour un résultat reproductible")
    ap.add_argument("--judge-model", default=None, help="modèle du juge (par défaut : le même que --model)")
    ap.add_argument("--api-key", default=config.OPENROUTER_API_KEY)
    ap.add_argument("--judge", action="store_true", help="active la fidélité + l'exactitude par LLM juge")
    ap.add_argument("--retrieval-only", action="store_true", help="métriques de retrieval uniquement (pas d'API)")
    ap.add_argument("--resume", action="store_true", help="reprend results.csv sans refaire les questions déjà traitées")
    ap.add_argument("--limit", type=int, default=None, help="n'évalue que les N premières questions")
    ap.add_argument("--sleep", type=float, default=3.0, help="pause entre deux questions (secondes)")
    args = ap.parse_args()

    if not args.retrieval_only and not args.api_key:
        sys.exit("Clé API absente : définissez OPENROUTER_API_KEY ou utilisez --retrieval-only.")

    ev = pd.read_csv(config.EVAL_CSV)
    if args.limit:
        ev = ev.head(args.limit)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    res_path = config.RESULTS_DIR / "results.csv"
    done: dict[str, dict] = {}
    if args.resume and res_path.exists():
        for r in pd.read_csv(res_path).to_dict("records"):
            if args.retrieval_only or (has_text(r.get("reponse")) and not has_text(r.get("erreur_llm"))):
                done[r["id_question"]] = r
        print(f"Reprise : {len(done)} question(s) déjà traitée(s).")

    rag = RAG()
    rows: list[dict] = []
    for n, q in enumerate(ev.to_dict("records"), start=1):
        if q["id_question"] in done:
            rows.append(done[q["id_question"]])
            continue
        expected = parse_ids(q["ids_attendus"])
        passages = rag.retrieve(q["question"], args.k)
        got = [p["metadata"]["page_id"] for p in passages]
        row = dict(q)
        row["pages_recuperees"] = ";".join(f"{p['metadata']['titre'][:22]}:p{p['metadata']['page']}" for p in passages)
        if expected:
            rank = next((i + 1 for i, g in enumerate(got) if g in expected), None)
            row["hit"] = int(rank is not None)
            row["rr"] = round(1 / rank, 3) if rank else 0.0
            row["recall"] = round(len(set(got) & set(expected)) / len(set(expected)), 3)

        if not args.retrieval_only:
            try:
                out = chat(build_messages(q["question"], passages),
                           api_key=args.api_key, model=args.model, temperature=0.0)
                row.update(reponse=out["content"], modele_utilise=out["model"], abstention=int(is_abstention(out["content"])))
                if args.judge:
                    row.update(judge(q["question"], passages, out["content"], q["reponse_reference"],
                                     args.api_key, args.judge_model or args.model))
                row["erreur_llm"] = ""
            except (OpenRouterError, ValueError, json.JSONDecodeError) as e:
                row["erreur_llm"] = str(e)[:200]
                print(f"  ⚠️ {q['id_question']} : {e}")
            time.sleep(args.sleep)

        rows.append(row)
        seen = {r["id_question"] for r in rows}        # sauvegarde incrémentale (on garde aussi les anciens résultats)
        pd.DataFrame(rows + [d for qid, d in done.items() if qid not in seen]).to_csv(res_path, index=False)
        print(f"[{n}/{len(ev)}] {q['id_question']} {q['difficulte']:<12} "
              f"hit={row.get('hit', '-')} abst={row.get('abstention', '-')} fidele={row.get('fidele', '-')} correct={row.get('correct', '-')}")

    df = pd.DataFrame(rows)
    for col in ("reponse", "abstention", "fidele", "correct", "erreur_llm"):
        if col not in df:
            df[col] = None
    df.to_csv(res_path, index=False)

    metrics = summarize(df, args.k)
    metrics["modele"] = args.model
    (config.RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    errs = []
    for r in df.to_dict("records"):
        cat = propose_error_category(r) if not args.retrieval_only or r.get("hit") == 0 else None
        if cat:
            errs.append({"id_question": r["id_question"], "difficulte": r["difficulte"], "question": r["question"],
                         "categorie_proposee": cat, "pages_attendues": r.get("pages_attendues"),
                         "pages_recuperees": r.get("pages_recuperees"), "reponse_systeme": r.get("reponse"),
                         "reponse_reference": r["reponse_reference"], "categorie_finale (à remplir)": "", "analyse (à remplir)": ""})
    pd.DataFrame(errs).to_csv(config.RESULTS_DIR / "erreurs_a_classer.csv", index=False)

    print("\n=== MÉTRIQUES ===")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"\n{len(errs)} cas à analyser -> {config.RESULTS_DIR / 'erreurs_a_classer.csv'}")


if __name__ == "__main__":
    main()
