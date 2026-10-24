"""Partie 1 : score de small_A sur les phrases de Test A vues / non vues dans Train A.

Usage (depuis la racine du dépôt) :
    python scripts/analyze_leak.py results/small_A_20261023_221426_test_A.csv
"""
import re
import sys

import pandas as pd
from sacrebleu.metrics import BLEU, CHRF

N_BOOTSTRAP = 1000


def norm(s):
    """Même normalisation que scripts/make_splits.py."""
    s = re.sub(r"[^\w\s]", " ", str(s).lower())
    return re.sub(r"\s+", " ", s).strip()


results = pd.read_csv(sys.argv[1])               # english, reference, hypothesis
train = pd.read_csv("data/splits/A/train.csv")   # eng, darija_ar

train_eng = set(train["eng"].map(norm))
results["seen"] = results["english"].map(norm).isin(train_eng)

for seen, part in results.groupby("seen"):
    hyps = part["hypothesis"].tolist()
    refs = [part["reference"].tolist()]
    b = BLEU().corpus_score(hyps, refs, n_bootstrap=N_BOOTSTRAP)
    c = CHRF(word_order=2).corpus_score(hyps, refs, n_bootstrap=N_BOOTSTRAP)
    label = "vues" if seen else "non vues"
    n_unique = part["english"].map(norm).nunique()
    print(f"{label:9s} | {len(part):5d} lignes, {n_unique:5d} phrases uniques")
    print(f"          | {b}")
    print(f"          | {c}")
