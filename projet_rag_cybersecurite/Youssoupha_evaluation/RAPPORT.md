# Rapport — Système RAG sur la cybersécurité

*Activité pratique « LLM & Écosystème » — Groupe : À COMPLÉTER (noms) — Date : À COMPLÉTER*

> Les zones **À COMPLÉTER** attendent VOS résultats. N'écrivez aucun chiffre que vous n'avez pas obtenu en exécutant le projet.

## 1. Objectif et corpus

Nous avons construit un assistant de question-réponse (RAG) sur la **cybersécurité**, à partir de documents publics en français.
Le système répond **uniquement** à partir des passages retrouvés, cite ses sources et dit « je ne sais pas » quand l'information est absente.

| # | Document | Auteur / organisme | Année | Pages utilisées |
|---|---|---|---|---|
| 1 | Introduction à la Cybersécurité | R. Kanawati, Université Sorbonne Paris Nord | 2023 | 101 |
| 2 | Guide pour la bonne gouvernance de la cybersécurité | DCAF | 2019 | 99 |
| 3 | La cybersécurité dans les entités des Nations Unies | Corps commun d'inspection de l'ONU | 2021 | 112 |
| 4 | Les fondamentaux de la cybersécurité | CSC | 2017 | 17 |
| 5 | Manuel de cybersécurité pour les partis politiques | National Democratic Institute (NDI) | — | 80 |

URL et dates de consultation : voir `perso1_corpus/SOURCES.md`. Nettoyage : suppression des en-têtes/pieds de page répétés, numéros de page,
pages vides (451 → 409 pages) ; limites : quelques artefacts d'extraction PDF subsistent (À COMPLÉTER si vous en avez observé d'autres).

## 2. Architecture

```
PDF → texte nettoyé (P1) → chunks + embeddings → ChromaDB (P2)
question → embedding → top-k passages → prompt → LLM (OpenRouter) → réponse + citations (P3)
```

## 3. Choix techniques et justification

**Découpage (P2).** Chunks de À COMPLÉTER caractères, chevauchement À COMPLÉTER, découpés par page et aux frontières de phrases.
Comparaison réalisée avec `compare_chunks.py` :

| Taille | Chevauchement | Nb chunks | Hit@5 | MRR |
|---|---|---|---|---|
| À COMPLÉTER | | | | |

Justification : À COMPLÉTER (compromis précision / contexte, taille du prompt).

**Embeddings.** `intfloat/multilingual-e5-small` (multilingue, léger, préfixes `query:` / `passage:`). **Base :** ChromaDB, distance cosinus,
métadonnées par chunk (titre, URL, page).

**Génération (P3).** Modèle : À COMPLÉTER (identifiant OpenRouter exact), température 0, top-k = À COMPLÉTER. Le prompt impose :
(1) répondre seulement avec les passages, (2) citer `[n]`, (3) répondre « Je ne sais pas » sinon.

## 4. Protocole d'évaluation

- **Jeu de 30 questions** (P4) : 7 faciles, 14 moyennes, 4 difficiles, 5 sans réponse dans le corpus (À ADAPTER si vous l'avez modifié).
- **Retrieval** : Hit@k, Recall@k, MRR (page attendue vs pages récupérées).
- **Génération** : fidélité et exactitude par LLM juge (modèle : À COMPLÉTER), abstention correcte/abusive.
- Vérification manuelle de N = À COMPLÉTER réponses pour contrôler le juge.

## 5. Résultats

| Métrique | Valeur |
|---|---|
| Hit@5 (25 questions avec réponse) | À COMPLÉTER |
| Recall@5 | À COMPLÉTER |
| MRR | À COMPLÉTER |
| Fidélité (juge LLM) | À COMPLÉTER |
| Exactitude (juge LLM) | À COMPLÉTER |
| Abstention correcte (5 questions sans réponse) | À COMPLÉTER |
| Abstention abusive (25 questions avec réponse) | À COMPLÉTER |

Par difficulté : À COMPLÉTER (`metrics.json`). Commentaire : À COMPLÉTER.

## 6. Analyse des erreurs

| Catégorie | Nb | Exemple (id question) | Cause probable |
|---|---|---|---|
| Mauvais chunk (retrieval) | | | |
| Hallucination / non fidèle | | | |
| Question sans réponse mal gérée | | | |
| Abstention abusive | | | |
| Réponse incorrecte malgré bon retrieval | | | |

Analyse détaillée de 2 à 3 cas représentatifs : À COMPLÉTER.

## 7. Ce que nous changerions

À COMPLÉTER à partir de vos observations (pistes : taille de chunk, top-k plus grand + reranking, recherche hybride pour les sigles,
reformulation de la question, seuil de score pour s'abstenir, meilleur modèle LLM, indexation des tableaux/formules mal extraits).

## 8. Limites

- Corpus restreint (5 documents) ; extraction PDF imparfaite (diapositives, formules).
- Juge LLM imparfait (possible indulgence) ; jeu de 30 questions limité statistiquement.
- Modèles gratuits : disponibilité et limites de requêtes variables (À COMPLÉTER si vous avez été limités).

## 9. Sources

Voir `perso1_corpus/SOURCES.md` (toutes les sources du corpus doivent y figurer et être citées ici).
Outils : Python, ChromaDB, sentence-transformers (multilingual-e5-small), OpenRouter, Gradio.
