"""Decontamination des donnees synthetiques de DODa-500K (Partie 2).

Retire des lignes synthetiques (doda_grounded, exploratory) :
  1. les lignes avec un anglais ou un darija vide ;
  2. les lignes avec moins de 70 % de lettres arabes (lettres seulement) ;
  3. les paires identiques (anglais normalise + darija) ;
  4. les lignes dont l'anglais normalise est dans Test B, Val B ou TerjamaBench.

Usage (racine du depot) :
    python scripts/decontaminate_synthetic.py
    python scripts/decontaminate_synthetic.py --jsonl data/darija_parallel.jsonl --out data/synthetic

TerjamaBench est charge depuis Hugging Face (jeu protege : variable HF_TOKEN
requise), comme dans evaluate_mt.py. Option --terjama CHEMIN.csv pour lire un
fichier local a la place.

Sortie : data/synthetic/doda_grounded.csv et data/synthetic/exploratory.csv
(colonnes eng, darija_ar). Ces fichiers sont gros : ne pas les ajouter a Git.
"""
import argparse
import os
import re

import pandas as pd

MODES = ["doda_grounded", "exploratory"]


def norm(s):
    # Meme normalisation que make_splits.py
    s = re.sub(r"[^\w\s]", " ", str(s).lower())
    return re.sub(r"\s+", " ", s).strip()


def ratio_arabe(t):
    # Lettres seulement : chiffres et ponctuation ne comptent pas
    lettres = [c for c in str(t) if c.isalpha()]
    if not lettres:
        return 0.0
    return sum("\u0600" <= c <= "\u06FF" for c in lettres) / len(lettres)


def english_set(path, col):
    return set(pd.read_csv(path)[col].map(norm))


def terjama_set(path):
    if path is None:
        # Meme chargement que evaluate_mt.py (colonne "English", split "test")
        from datasets import load_dataset
        return set(norm(x) for x in load_dataset("atlasia/TerjamaBench")["test"]["English"])
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    key = cols.get("english") or cols.get("eng")
    if key is None:
        raise SystemExit(f"Colonne anglaise introuvable dans {path}. Colonnes : {list(df.columns)}")
    return set(df[key].map(norm))


ap = argparse.ArgumentParser()
ap.add_argument("--jsonl", default="data/darija_parallel.jsonl")
ap.add_argument("--terjama", default=None, help="CSV local (sinon Hugging Face)")
ap.add_argument("--test_b", default="data/splits/B/test.csv")
ap.add_argument("--val_b", default="data/splits/B/val.csv")
ap.add_argument("--out", default="data/synthetic")
args = ap.parse_args()

test_b = english_set(args.test_b, "eng")
val_b = english_set(args.val_b, "eng")
terjama = terjama_set(args.terjama)
print(f"Phrases anglaises : Test B {len(test_b)} | Val B {len(val_b)} | TerjamaBench {len(terjama)}")

df = pd.read_json(args.jsonl, lines=True)
os.makedirs(args.out, exist_ok=True)

for mode in MODES:
    d = df[df["generation_mode"] == mode].copy()
    print(f"\n=== {mode} ===")
    print("Lignes au depart                    :", len(d))

    d["eng_n"] = d["english"].map(norm)
    d["darija_arabic"] = d["darija_arabic"].fillna("").astype(str)

    n = len(d)
    d = d[(d["eng_n"] != "") & (d["darija_arabic"].str.strip() != "")]
    print("Retirees : anglais ou darija vide   :", n - len(d))

    n = len(d)
    d = d[d["darija_arabic"].map(ratio_arabe) >= 0.7]
    print("Retirees : moins de 70 % arabe      :", n - len(d))

    n = len(d)
    d = d.drop_duplicates(["eng_n", "darija_arabic"])
    print("Retirees : paires identiques        :", n - len(d))

    # Recouvrement avec les ensembles interdits (une ligne peut etre dans plusieurs)
    in_test = d["eng_n"].isin(test_b)
    in_val = d["eng_n"].isin(val_b)
    in_terj = d["eng_n"].isin(terjama)
    print("Lignes dans Test B                  :", int(in_test.sum()))
    print("Lignes dans Val B                   :", int(in_val.sum()))
    print("Lignes dans TerjamaBench            :", int(in_terj.sum()))
    bad = in_test | in_val | in_terj
    print("Retirees : total (union des 3)      :", int(bad.sum()))
    d = d[~bad]

    # Verification finale
    assert not d["eng_n"].isin(test_b | val_b | terjama).any()

    print("Lignes finales                      :", len(d))
    print("Phrases anglaises uniques           :", d["eng_n"].nunique())

    out = d.rename(columns={"english": "eng", "darija_arabic": "darija_ar"})[["eng", "darija_ar"]]
    path = os.path.join(args.out, f"{mode}.csv")
    out.to_csv(path, index=False)
    print("Ecrit :", path)
