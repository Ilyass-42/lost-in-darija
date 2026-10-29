"""Test des phrases communes : small_A et small_B sur les memes phrases.

Usage (depuis la racine du depot) :
    python scripts/analyze_common.py RESULTS_A.csv RESULTS_B.csv

RESULTS_A : CSV de small_A sur Test A (colonnes english, reference, hypothesis)
RESULTS_B : CSV de small_B sur Test B (memes colonnes)
"""
import re
import sys

import numpy as np
import pandas as pd
from sacrebleu.metrics import BLEU, CHRF


def norm(s):
    # Meme normalisation que make_splits.py : minuscules, ponctuation -> espace
    s = re.sub(r"[^\w\s]", " ", str(s).lower())
    return re.sub(r"\s+", " ", s).strip()


def load_results(path):
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    print(path, "-> colonnes :", list(df.columns))
    df["hypothesis"] = df["hypothesis"].fillna("").astype(str)
    df["eng_n"] = df["english"].map(norm)
    return df


def load_train(path):
    return set(pd.read_csv(path)["eng"].map(norm))


res_a = load_results(sys.argv[1])
res_b = load_results(sys.argv[2])
train_a = load_train("data/splits/A/train.csv")
train_b = load_train("data/splits/B/train.csv")

# 1. Phrases anglaises dans Test A et Test B, et dans aucun des deux Train
common = (set(res_a["eng_n"]) & set(res_b["eng_n"])) - train_a - train_b
print("\nPhrases communes (Test A et Test B, hors Train A et B) :", len(common))

# 2. On garde les lignes qui ont la MEME reference darija dans A et dans B
a = res_a[res_a["eng_n"].isin(common)][["eng_n", "reference", "hypothesis"]]
b = res_b[res_b["eng_n"].isin(common)][["eng_n", "reference", "hypothesis"]]
print("Lignes dans A :", len(a), "| lignes dans B :", len(b))
m = a.merge(b, on=["eng_n", "reference"], suffixes=("_a", "_b"))
n = len(m)
print("Lignes avec la meme reference dans A et B :", n)
if n == 0:
    sys.exit("Aucune ligne commune : arret.")

ref = m["reference"].tolist()
hyp_a = np.array(m["hypothesis_a"].tolist(), dtype=object)
hyp_b = np.array(m["hypothesis_b"].tolist(), dtype=object)
ref_arr = np.array(ref, dtype=object)

chrf = CHRF(word_order=2)  # chrF++
bleu = BLEU()              # BLEU ordre 4


def score(metric, hyps, refs):
    return metric.corpus_score(list(hyps), [list(refs)])


# 3. Scores des deux modeles sur les memes lignes
print("\n          chrF++    BLEU   4-grammes")
for name, hyp in (("small_A", hyp_a), ("small_B", hyp_b)):
    c = score(chrf, hyp, ref_arr)
    bl = score(bleu, hyp, ref_arr)
    print(f"{name}  {c.score:7.2f}  {bl.score:6.2f}   {bl.totals[3]}")

# 4. Bootstrap apparie de la difference A - B (chrF++)
rng = np.random.default_rng(12345)
diffs = []
for _ in range(1000):
    idx = rng.integers(0, n, n)
    d = score(chrf, hyp_a[idx], ref_arr[idx]).score - score(chrf, hyp_b[idx], ref_arr[idx]).score
    diffs.append(d)
low, high = np.percentile(diffs, [2.5, 97.5])
print(f"\nDifference chrF++ (A - B) : {np.mean(diffs):.2f}  IC 95 % [{low:.2f} ; {high:.2f}]")
print("Si l'intervalle contient 0, la difference n'est pas distincte du hasard du test.")
