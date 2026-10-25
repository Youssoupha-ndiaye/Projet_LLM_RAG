**Mission :** découper le corpus en chunks, calculer leurs embeddings, les stocker dans ChromaDB avec leurs métadonnées.
**Livrable :** `index.py`. **Échéance :** mardi 23h59. **Entrée :** `perso1_corpus/data/corpus_clean.csv` (déjà fourni).

## Contenu du dossier

```
perso2_indexation/
├── chunking.py          # logique de découpage (fonctions réutilisées par les autres scripts)
├── index.py             # (livrable) construit la base ChromaDB
├── compare_chunks.py    # teste plusieurs tailles de chunks pour JUSTIFIER le choix
├── chroma_db/           # créé par index.py (base vectorielle) — ne pas committer
└── index_info.json      # créé par index.py : paramètres utilisés (taille, modèle...)
```

## Explication du code

### `chunking.py`
Un **chunk** est un petit passage de texte (≈ 600 caractères). On ne peut pas donner 400 pages au LLM : on lui donne seulement
les quelques chunks les plus proches de la question.

| Fonction | Ce qu'elle fait |
|---|---|
| `split_units(text, max_len)` | Coupe le texte en phrases/lignes (après `. ! ? ; :` ou retour à la ligne). Une phrase plus longue que `max_len` est coupée à un espace. |
| `chunk_text(text, size, overlap)` | Assemble ces unités jusqu'à `size` caractères. Quand un chunk est plein, le suivant **recommence avec les dernières unités du précédent** (≤ `overlap` caractères) : c'est le chevauchement. |
| `build_chunks(df, size, overlap)` | Applique `chunk_text` **page par page** et attache à chaque chunk ses métadonnées : titre, auteur, organisme, date, URL, **page**, `page_id`, index du chunk. |

Choix à défendre : **découpage page par page** (un chunk garde un numéro de page exact → citations fiables, évaluation simple),
**coupure aux phrases** (jamais au milieu d'un mot), **chevauchement** (une info située à la jonction n'est pas perdue).

### `index.py`
```
corpus_clean.csv → build_chunks → embeddings e5 → ChromaDB (distance cosinus)
```

| Élément | Explication |
|---|---|
| `get_embedder()` | Charge `intfloat/multilingual-e5-small` (multilingue, ~120 Mo). Ce modèle **exige** les préfixes `passage: ` (documents) et `query: ` (questions) : c'est géré dans `common/embedder.py`. |
| `embedding_text()` | Le texte *embarqué* = « titre du document + chunk ». Ça aide quand la question mentionne le document. Le texte *stocké* et montré au LLM reste le chunk seul. Option `--no-title` pour comparer. |
| `build_collection()` | Crée la collection (`hnsw:space = cosine`), calcule les embeddings par lots de 256 et insère `ids`, `documents`, `metadatas`, `embeddings`. |
| `index_info.json` | Mémorise taille, chevauchement, modèle : P3 et P5 s'en servent et vous pouvez le citer dans le rapport. |

Le **modèle d'embeddings est mémorisé dans la collection** : P3 interroge automatiquement avec le même modèle
(mélanger deux modèles donnerait des résultats absurdes).

```bash
python perso2_indexation/index.py --reset                      # valeurs de config.py (600 / 100)
python perso2_indexation/index.py --size 500 --overlap 80 --reset
```
Au premier lancement, le modèle e5 est téléchargé (internet nécessaire). Comptez quelques minutes sur CPU.
`EMBED_BACKEND=dummy` permet de tester le code sans télécharger (résultats **non représentatifs**).

### `compare_chunks.py` — de quoi justifier le choix
Construit, pour chaque couple `taille:chevauchement`, un index temporaire en mémoire, puis affiche nombre de chunks, taille moyenne et,
si `perso4_eval_set/eval_set.csv` existe (il est fourni), **hit@k** et **MRR** (la bonne page est-elle dans les k résultats, et à quel rang ?).

```bash
python perso2_indexation/compare_chunks.py --configs 300:50 500:100 600:100 800:150 --k 5
```
Collez le tableau obtenu dans le rapport, puis **reportez la configuration gagnante dans `config.py`**
(`CHUNK_SIZE`, `CHUNK_OVERLAP`) et relancez `index.py --reset`. Interprétation type : des chunks courts sont plus précis
mais perdent le contexte ; des chunks longs gardent le contexte mais diluent l'information utile et consomment le prompt du LLM.

## Points de vigilance

- Après tout changement de taille/overlap/modèle : `index.py --reset`.
- Ne committez pas `chroma_db/` (déjà dans `.gitignore`) : chacun la reconstruit.
- Vous pouvez aussi tester un autre modèle (`paraphrase-multilingual-MiniLM-L12-v2`) : changez `EMBED_MODEL` dans `config.py`
  **et supprimez les préfixes e5** dans `common/embedder.py` (ils ne concernent que e5).

## À dire à l'oral

« On découpe page par page en chunks de N caractères avec M de chevauchement, coupés aux phrases. On a comparé plusieurs tailles
sur 25 questions : [résultat]. Les embeddings viennent de multilingual-e5-small, stockés dans ChromaDB avec le titre, l'URL
et la page de chaque chunk pour pouvoir citer. »
