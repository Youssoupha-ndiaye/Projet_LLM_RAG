**Mission :** fournir à toute l'équipe un corpus propre, avec ses métadonnées, et documenter les sources.
**Livrables :** dossier `data/` + `SOURCES.md`. **Échéance :** lundi 23h59.

## Contenu du dossier

```
perso1_corpus/
├── clean_corpus.py        # nettoie le CSV  -> data/corpus_clean.csv
├── build_sources.py       # génère SOURCES.md
├── SOURCES.md             # (livrable) fiche de chaque document
└── data/
    ├── dataset_nettoye.csv      # fichier de départ : 1 ligne = 1 page de PDF (451 pages)
    ├── corpus_clean.csv         # (livrable) corpus nettoyé : 409 pages utiles
    └── rapport_nettoyage.md     # ce qui a été supprimé, document par document
```

Colonnes du CSV : `id` (identifiant unique de la page), `titre`, `auteur`, `organisme`, `date_publication`,
`source` (nom du PDF), `page`, `url`, `texte`.

## Explication du code

### `clean_corpus.py`
Pourquoi nettoyer ? Dans le cours de Kanawati, l'en-tête *« Introduction Attaques Pare-feux Détection d'intrusion Authentification »*
est répété sur ~110 pages. Sans nettoyage, il se retrouve dans presque tous les chunks, rend leurs embeddings artificiellement
semblables et dégrade la recherche.

| Fonction | Ce qu'elle fait |
|---|---|
| `normalize_text()` | Normalise l'Unicode (NFKC) : ligatures, espaces insécables, caractères de contrôle. |
| `line_key()` | Fabrique une « clé » de ligne : minuscules + tous les chiffres remplacés par `#`. Ainsi `12 / 127` et `13 / 127` ont la même clé. |
| `find_boilerplate()` | Compte, pour **un même document**, dans combien de pages chaque ligne apparaît. Une ligne présente dans ≥ 10 % des pages (et ≤ 15 mots) = en-tête / pied de page / menu → à supprimer. |
| `clean_page()` | Supprime ces lignes, les numéros de page (`96 / 127`, `xii`), le code ONU `GE.21-14702`, recolle les mots coupés (`gou- vernance` → `gouvernance`) et corrige `D ’UN` → `D’UN`. |
| `main()` | Applique le tout document par document, supprime les pages de moins de 60 caractères (diapos qui ne contenaient qu'une image), remplace les métadonnées manquantes par « Non précisé », écrit `corpus_clean.csv` et `rapport_nettoyage.md`. |

Résultat sur ce corpus : **451 → 409 pages**, ~5,5 % de caractères de bruit retirés.
Options : `--min-chars 80` (plus strict), `--repeat-threshold 0.15` (moins agressif), `--no-hyphen-fix`.

### `build_sources.py`
Lit `corpus_clean.csv` et écrit `SOURCES.md` : titre, auteur, organisme, date de publication, URL, fichier, pages conservées,
**date de consultation**. Lancez-le avec la vraie date où vous avez téléchargé les PDF :

```bash
python perso1_corpus/build_sources.py --date-consultation 2026-10-19
```

## À faire à la main (ce que le code ne peut pas deviner)

- [ ] Renseigner la **date de consultation** réelle dans `SOURCES.md`.
- [ ] Compléter les auteurs/dates « Non précisé » si vous les trouvez dans les PDF (ex. auteur du guide DCAF).
- [ ] Ouvrir `data/rapport_nettoyage.md` et vérifier que les lignes supprimées sont bien du bruit.
- [ ] Parcourir quelques pages de `corpus_clean.csv` : certains artefacts d'extraction PDF subsistent
      (ex. `ˆ` dans « co ˆut »). Ils sont rares et sans incidence majeure, mais à mentionner comme **limite** dans le rapport.
- [ ] Vérifier que chaque URL du CSV est accessible.

## À dire à l'oral

« On a gardé 5 documents publics en français sur la cybersécurité, 451 pages. On a détecté automatiquement les lignes
répétées par document, supprimé les numéros de page et les pages vides : 409 pages exploitables. Chaque page garde ses
métadonnées pour pouvoir citer la source. »
