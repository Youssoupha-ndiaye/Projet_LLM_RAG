**Mission :** interroger la base (top-k), construire un prompt qui force le modèle à répondre uniquement à partir des passages
(sinon « je ne sais pas »), appeler un LLM et afficher les sources.
**Livrables :** `rag.py` (+ `app.py`, l'interface). **Échéance :** mercredi 23h59. **Dépend de :** l'index de P2.

## Contenu du dossier

```
Ibrahima_rag/
├── rag.py                    # (livrable) pipeline RAG + utilisation en ligne de commande
├── app.py                    # interface web : vous collez votre clé API OpenRouter et vous interrogez le modèle
└── test_pipeline_offline.py  # teste tout SANS clé API (LLM simulé)
```
(le client OpenRouter lui-même est dans `common/openrouter_client.py`)

## 1. Interroger le modèle : clé API OpenRouter + interface

### Obtenir la clé
1. Compte gratuit sur <https://openrouter.ai> → <https://openrouter.ai/keys> → **Create Key** → copiez `sk-or-v1-...`.
2. **Ne la partagez pas et ne la mettez jamais sur GitHub.** Elle ne s'enregistre nulle part dans ce code.

### Lancer l'interface
```bash
python perso2_indexation/index.py --reset     # une seule fois (si pas déjà fait)
python perso3_rag/app.py                      # puis ouvrez http://127.0.0.1:7860
python perso3_rag/app.py --share              # lien public temporaire (Google Colab)
```
Dans la page :
1. **🔑 Clé API OpenRouter** : collez la clé (champ masqué, gardée en mémoire le temps de la session).
2. **Modèle** : laissez `openrouter/free`, ou cliquez **🔄 Actualiser** pour lister les modèles gratuits du moment et en choisir un
   (vous pouvez aussi taper un identifiant à la main).
3. **Nombre de passages (top-k)** : 5 par défaut.
4. Écrivez la question → **Interroger**. S'affichent : la réponse avec citations `[1] [2]`, le modèle qui a répondu, et le tableau des
   passages récupérés (✔ = cité dans la réponse) avec document, page et score. Un accordéon donne le texte complet des passages.

### Quel modèle choisir sur OpenRouter ?
- **`openrouter/free`** (par défaut) : un « routeur » d'OpenRouter qui choisit lui-même un modèle gratuit disponible. C'est le plus simple
  et il résiste mieux quand un modèle gratuit disparaît. **Inconvénient :** le modèle change d'un appel à l'autre.
- **Pour l'évaluation (P5), fixez un modèle précis** (un identifiant se terminant par `:free`, trouvé via le bouton Actualiser ou sur
  <https://openrouter.ai/models?pricing=free>) : sinon vos résultats ne sont pas reproductibles. Préférez un modèle récent, de taille
  importante et « instruct », bon en français ; testez-en 2 ou 3 sur quelques questions.
- La liste des modèles gratuits **change souvent** (des modèles sont retirés sans préavis) : ne figez pas un identifiant la veille du rendu
  sans l'avoir retesté. Vous pouvez fournir des modèles de secours via `FALLBACK_MODELS` dans `config.py`.
- **Limites** : les modèles gratuits sont limités en nombre de requêtes par jour (de l'ordre de quelques dizaines sans crédit, plus si
  vous avez rechargé votre compte — vérifiez les chiffres actuels sur la documentation OpenRouter). Prévoyez-le pour l'évaluation.

## 2. Explication du code

### `rag.py`
| Élément | Ce qu'il fait |
|---|---|
| `SYSTEM_PROMPT` | Les consignes : répondre **uniquement** à partir des passages, **citer** `[n]`, et si l'info manque répondre **exactement** « Je ne sais pas : l'information n'est pas présente dans les documents fournis. » Température 0 pour des réponses stables. |
| `RAG.__init__()` | Ouvre ChromaDB et charge **le même modèle d'embeddings que celui de l'indexation** (lu dans les métadonnées de la collection). |
| `RAG.retrieve(question, k)` | Embedding de la question (préfixe `query: `) → les k chunks les plus proches (distance cosinus) avec texte, métadonnées et score = 1 − distance. |
| `build_messages()` | Fabrique la conversation : message système + « PASSAGES : [1] (Document : …, page …) texte… [2] … QUESTION : … ». |
| `RAG.answer()` | Enchaîne `retrieve` → `build_messages` → `chat()` (OpenRouter) et renvoie la réponse, le modèle utilisé, les passages, les numéros **cités**, et un booléen `abstained`. |
| `cited_indexes()` | Repère les `[1]`, `[3]`… présents dans la réponse : ce sont les « passages utilisés » à afficher. |
| `is_abstention()` | Détecte un « je ne sais pas » (réutilisé par P5 pour mesurer le comportement hors corpus). |

En ligne de commande :
```bash
export OPENROUTER_API_KEY="sk-or-v1-..."
python perso3_rag/rag.py "Quelles sont les trois clés d'un bon mot de passe ?"
python perso3_rag/rag.py "..." --k 8 --show-passages
python perso3_rag/rag.py "..." --retrieval-only        # n'appelle pas le LLM : teste la recherche seule
```

### `common/openrouter_client.py` (partagé)
`chat()` appelle `POST https://openrouter.ai/api/v1/chat/completions` (format compatible OpenAI). Il gère : clé refusée (401), crédits
(402), **limite de débit (429 → attend puis réessaie)**, réponses vides (certains modèles), et bascule sur les modèles de secours.
`list_free_models()` interroge `GET /models` et ne garde que les modèles dont le prix d'entrée **et** de sortie est 0.

### `app.py`
Interface Gradio : `ask()` appelle `RAG.answer()` et met en forme réponse + tableau de sources ; `refresh_models()` alimente la liste
déroulante ; `index_status()` affiche en haut les paramètres de l'index (et avertit si l'index est un index de test `dummy`).

### `test_pipeline_offline.py`
Vérifie sans clé : retrieval, structure du prompt, citations, détection d'abstention, interface. À lancer avant de consommer vos requêtes.

## 3. Pistes d'amélioration (« pour aller plus loin », non obligatoires)
- Reformuler la question avant la recherche ; récupérer 10 passages puis re-classer (reranking).
- Filtrer par document avec les métadonnées (`where={"titre": ...}` dans `col.query`).
- Recherche hybride (mots-clés + vecteurs) : utile pour les sigles (NIDS, PPR, RSA…).
- Afficher un seuil : si le meilleur score est très bas, s'abstenir sans même appeler le LLM.

## À dire à l'oral
« La question est vectorisée avec le même modèle que le corpus, on récupère les 5 chunks les plus proches, on les numérote dans un prompt
qui interdit d'utiliser autre chose et impose de citer et de dire "je ne sais pas" si l'information manque. Le LLM est appelé via
OpenRouter ; l'interface affiche la réponse et les passages utilisés avec document et page. »
