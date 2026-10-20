import re
import pandas as pd


def norm(s):
    """Normalize an English sentence (same rule as make_splits.py)."""
    s = re.sub(r"[^\w\s]", " ", str(s).lower())
    return re.sub(r"\s+", " ", s).strip()


def leak(train, val, test):
    """Count shared normalized English sentences between the sets."""
    tr, va, te = set(train["eng_n"]), set(val["eng_n"]), set(test["eng_n"])
    return len(te & tr), len(va & tr), len(te & va)


for name in ["A", "B"]:
    parts = [pd.read_csv(f"data/splits/{name}/{s}.csv") for s in ["train", "val", "test"]]
    for p in parts:
        p["eng_n"] = p["eng"].map(norm)
    print(name, [len(p) for p in parts], leak(*parts))