"""Measure English-sentence overlap between Train, Val and Test.

Normalization: lowercase + strip spaces.
Percentages are computed on unique English sentences.
"""
import sys
import pandas as pd


def norm_set(df: pd.DataFrame, col: str = "eng") -> set:
    return set(df[col].astype(str).str.lower().str.strip())


def overlap(a: set, b: set) -> int:
    return len(a & b)


def main(data_dir: str = "data") -> None:
    tr = norm_set(pd.read_csv(f"{data_dir}/Train.csv"))
    va = norm_set(pd.read_csv(f"{data_dir}/Val.csv"))
    te = norm_set(pd.read_csv(f"{data_dir}/Test.csv"))

    print(f"Unique sentences  Train={len(tr)}  Val={len(va)}  Test={len(te)}")
    print(f"Test ∩ Train: {overlap(te, tr)} / {len(te)} = {overlap(te, tr) / len(te):.2%}")
    print(f"Val  ∩ Train: {overlap(va, tr)} / {len(va)} = {overlap(va, tr) / len(va):.2%}")
    print(f"Test ∩ Val  : {overlap(te, va)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data")