**Mission :** lancer le système sur les 30 questions, calculer les métriques, classer les erreurs, rédiger le rapport.
**Livrables :** `evaluate.py` et `RAPPORT.md`. **Échéance :** jeudi 23h59. **Dépend de :** `eval_set.csv` (P4) et du système (P2 + P3).

## Contenu du dossier

```
Youssoupha_evaluation/
├── evaluate.py           # (livrable) lance l'évaluation et calcule les métriques
├── RAPPORT.md            # (livrable) modèle de rapport pré-rempli, à compléter avec VOS résultats
└── resultats/            # créé par evaluate.py
    ├── results.csv             # 1 ligne par question : réponse, passages, hit, fidélité...
    ├── metrics.json            # métriques globales et par difficulté
    └── erreurs_a_classer.csv   # cas à analyser, avec une catégorie PROPOSÉE
```

## Déroulé conseillé

```bash
# 1) Retrieval seul : gratuit, instantané, aucune clé API
python Youssoupha_evaluation/evaluate.py --retrieval-only

# 2) Test sur 5 questions avec le LLM (vérifier que tout marche avant de consommer les requêtes gratuites)
export OPENROUTER_API_KEY="sk-or-v1-..."
python Youssoupha_evaluation/evaluate.py --model <id_modele:free> --judge --limit 5

# 3) Évaluation complète ; --resume reprend où l'on s'est arrêté si la limite quotidienne est atteinte
python perso5_evaluation/evaluate.py --model <id_modele:free> --judge --resume
```
⚠️ **Fixez le modèle** (`--model`) pour un résultat reproductible ; avec `openrouter/free` le modèle change à chaque appel.
Précisez dans le rapport le modèle d'embeddings, la taille de chunk, le top-k **et le modèle LLM** utilisés.
Budget de requêtes : 30 (réponses) + 30 (juge) = **60 appels** ; si votre quota gratuit journalier est plus bas, utilisez `--resume` le lendemain.

## Les métriques (et comment les expliquer)

**A. Qualité du retrieval** (25 questions avec réponse ; sans LLM)
| Métrique | Définition | Lecture |
|---|---|---|
| **Hit@k** | 1 si au moins une page attendue est parmi les k passages récupérés, sinon 0 ; moyenne sur les questions | « Dans x % des cas, le bon passage est dans le top-k. » |
| **Recall@k** | part des pages attendues retrouvées (utile si une réponse est sur 2 pages) | complète le hit pour les questions multi-pages |
| **MRR** | moyenne de 1/rang du premier bon passage | 1 = toujours en 1er ; 0,5 ≈ souvent 2ᵉ |

**B. Génération** (avec LLM)
| Métrique | Définition |
|---|---|
| **Fidélité** (`--judge`) | Le juge LLM lit passages + réponse : toutes les affirmations sont-elles appuyées par les passages ? (1/0) → mesure l'**hallucination** |
| **Exactitude** (`--judge`) | La réponse correspond-elle à la réponse de référence ? (1/0) |
| **Abstention correcte** | Parmi les 5 questions sans réponse, part où le système dit « je ne sais pas » (idéal : 100 %) |
| **Abstention abusive** | Parmi les 25 questions avec réponse, part où il s'abstient à tort (idéal : 0 %) |

Le juge est un LLM : il peut se tromper et, utilisant le même modèle que le générateur, être trop indulgent. **Relisez à la main** au moins
les cas signalés et un échantillon des autres, et dites-le dans le rapport (c'est une limite méthodologique à mentionner).

## Explication du code `evaluate.py`

| Fonction / bloc | Rôle |
|---|---|
| `main()` — boucle principale | Pour chaque question : `rag.retrieve()` → compare les `page_id` récupérés aux `ids_attendus` → hit, rang (rr), recall. Puis, sans `--retrieval-only`, appelle le LLM avec **le même prompt que l'application** (`build_messages`) et détecte l'abstention. Sauvegarde `results.csv` **après chaque question**. |
| `judge()` | Envoie question + passages + réponse + référence à un LLM avec un prompt qui exige un JSON `{"fidele":0/1,"correct":0/1,"commentaire":…}` ; extrait et parse le JSON. |
| `summarize()` | Calcule les moyennes globales et par difficulté → `metrics.json`. |
| `propose_error_category()` | Propose une catégorie d'erreur : **mauvais chunk** (bonne page absente du top-k), **hallucination** (fidélité = 0), **question sans réponse mal gérée**, **abstention abusive**, **réponse incorrecte malgré un bon retrieval**, **erreur technique**. |
| Reprise (`--resume`) | Relit `results.csv`, saute les questions déjà réussies, retente celles en erreur (ex. limite 429). |

## Analyse des erreurs (le cœur du rapport)

Ouvrez `resultats/erreurs_a_classer.csv`. Pour chaque ligne : lisez la question, les pages attendues vs récupérées et la réponse du système,
remplissez `categorie_finale` et `analyse`. Puis comptez les erreurs par catégorie dans le rapport. Questions à vous poser :
- *Mauvais chunk* : la page attendue est-elle trop pauvre (diapo), ou la question trop éloignée du vocabulaire du document ? Le chunk est-il trop gros/petit ?
- *Hallucination* : quel passage a été ignoré ? Le prompt est-il assez strict ?
- *Sans réponse mal gérée* : le modèle a-t-il répondu avec sa connaissance générale ?

Pensez à faire **varier un paramètre** (k = 3 / 5 / 8, ou taille de chunks) et à comparer : c'est ce qui permet d'écrire « ce que nous changerions ».

## `RAPPORT.md`
Plan pré-rempli (corpus, architecture, choix techniques, protocole d'évaluation, résultats, erreurs, améliorations, limites, sources).
Les zones **`À COMPLÉTER`** concernent les chiffres : **n'inscrivez jamais de valeur que vous n'avez pas obtenue vous-mêmes.**

## À dire à l'oral
« Sur 25 questions avec réponse, la bonne page est dans le top-5 dans X % des cas (MRR Y). Sur les 5 questions hors corpus, le système
s'abstient Z fois. La fidélité mesurée par un LLM juge est W %, que nous avons vérifiée à la main sur N cas. Les erreurs se répartissent en…
et nous changerions… »
