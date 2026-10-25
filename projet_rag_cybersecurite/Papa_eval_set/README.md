**Mission :** écrire ~30 questions avec réponse de référence et passage source attendu : faciles, difficiles et sans réponse dans le corpus.
**Livrable :** `eval_set.csv`. **Échéance :** jeudi 23h59. **Dépend de :** le corpus de P1 (sert ensuite à P5).

## Contenu du dossier

```
Papa_eval_set/
├── eval_set.csv          # (livrable) 30 questions : 25 avec réponse + 5 sans réponse
└── check_eval_set.py     # vérifie que chaque question est cohérente avec le corpus
```

## Le fichier `eval_set.csv`

Un brouillon complet de **30 questions** est fourni. Chaque réponse de référence a été rédigée à partir du texte des pages indiquées, et
`check_eval_set.py` confirme que les mots-clés attendus figurent bien dans ces pages. **Votre travail : relire, corriger, reformuler
dans vos mots et compléter** (c'est vous qui devez pouvoir défendre chaque question à l'oral).

| Colonne | Rôle |
|---|---|
| `id_question` | Q01…Q30 |
| `question` | Question posée au système |
| `reponse_reference` | Réponse attendue (utilisée par le juge LLM de P5) |
| `titre_document` | Document source attendu |
| `pages_attendues` | Numéro(s) de page dans le PDF (lisible par un humain) |
| `ids_attendus` | Identifiant(s) `id` des pages dans `corpus_clean.csv` — **c'est ce que Youssoupha compare** au retrieval. Plusieurs valeurs séparées par `;` |
| `mots_cles_passage` | Mots/expressions qui DOIVENT apparaître dans les pages attendues (contrôle automatique) |
| `mots_absents` | Pour les questions sans réponse : mots qui ne doivent apparaître nulle part dans le corpus |
| `difficulte` | `facile`, `moyen`, `difficile` ou `sans_reponse` |
| `type_question` | définition, factuelle, comparaison, calcul, procédure, multi-passages, hors_corpus |

### Répartition du brouillon
- **7 faciles** (un fait dans un passage), **14 moyennes**, **4 difficiles** (comparaison NIDS/HIDS, calcul d'erreur d'IDS, certificats
  contre l'attaque de l'homme du milieu, question multi-documents), **5 sans réponse**.
- Couverture : Kanawati 10, Manuel NDI 5, Fondamentaux CSC 3, Guide DCAF 3, Rapport ONU 3, 1 question multi-documents, 5 hors corpus.
- Les 5 questions **sans réponse** portent sur des sujets proches du thème mais absents du corpus (WPA3, rapport IBM, pfSense,
  LockBit, Zero Trust) : elles testent si le système **s'abstient** au lieu d'inventer.

## Explication de `check_eval_set.py`

Lancez-le **à chaque modification** :
```bash
python Papa_eval_set/check_eval_set.py
```
| Étape du script | Ce qu'il contrôle |
|---|---|
| `norm()` | Compare sans tenir compte des accents, majuscules, apostrophes `’`/`'` ni espaces multiples (robuste aux artefacts PDF). |
| Question avec réponse | Les `ids_attendus` existent dans `corpus_clean.csv` ; **chaque mot-clé** figure dans le texte de ces pages. |
| Question sans réponse | **Aucun** des `mots_absents` n'apparaît dans tout le corpus ; pas d'`ids_attendus`. |
| Bilan | Répartition par difficulté et par document pour vérifier l'équilibre. Code de sortie 1 s'il y a un problème. |

## Comment ajouter ou modifier une question
1. Trouvez la page : `grep` dans `perso1_corpus/data/corpus_clean.csv` (ou ouvrez-le dans Excel/pandas) et notez la colonne **`id`**.
2. Ajoutez une ligne : question, réponse de référence **recopiée/résumée du passage**, `ids_attendus`, 1 à 3 `mots_cles_passage`.
3. Lancez `check_eval_set.py`.
4. Si une question a sa réponse sur **plusieurs pages**, mettez tous les ids (`383;245`) : P5 compte un *hit* si **au moins une** est retrouvée
   et calcule le *recall* sur l'ensemble.

## Bonnes pratiques (pour que l'évaluation soit juste)
- Une question = une seule idée ; évitez de recopier la phrase du document mot pour mot (sinon le retrieval est trop facile).
- Mélangez les formulations (« Qu'est-ce que… », « Pourquoi… », « Quelle différence… »).
- Gardez aussi des questions **qui échoueront peut-être** : c'est ce qui rend l'analyse d'erreurs intéressante.
- Ne choisissez pas les questions en regardant les résultats du système (sinon biais).
- Évitez les questions dont la réponse se trouve dans une page très pauvre en texte (formules illisibles après extraction PDF).

## À dire à l'oral
« On a écrit 30 questions : 7 faciles, 14 moyennes, 4 difficiles (comparaisons, calculs, multi-documents) et 5 sans réponse, avec pour chacune
la réponse de référence, le document et la page attendus. Un script vérifie automatiquement que la réponse est bien dans la page citée. »
