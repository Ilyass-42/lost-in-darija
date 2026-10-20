import pandas as pd
import re
from datasets import load_dataset
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from pathlib import Path


SEED = 42


df = pd.read_csv("data/doda_human.csv")
def norm(s):
    """Normalize an English sentence for grouping.

    Lowercase the text. Replace punctuation with a space.
    Replace multiple spaces with one space. Remove spaces at the start and end.
    Example: "Thank you!!" -> "thank you"
    """
    s = re.sub(r"[^\w\s]", " ", str(s).lower())
    return re.sub(r"\s+", " ", s).strip()

df["eng_n"] = df["english"].map(norm)
df = df.drop_duplicates(subset=["eng_n","darija_arabic"]) # removes identical pairs.


tb = load_dataset("atlasia/TerjamaBench")  
# Remove every TerjamaBench English sentence from the data before the split.
# Reason: TerjamaBench is our independent benchmark. The model must not see its sentences.

tb_eng = {norm(x) for x in tb["test"]["English"]}

in_tb = df["eng_n"].isin(tb_eng)

print(f"Lignes retirées: {in_tb.sum()}")
print(f"Phrases: {df.loc[in_tb, 'eng_n'].nunique()}")


df = df[~in_tb]
print(len(df), df["eng_n"].nunique())

a_train, a_tmp = train_test_split(df,test_size=0.2, random_state=SEED)
a_val, a_test = train_test_split(a_tmp,test_size=0.5, random_state=SEED)


print("A :", len(a_train), len(a_val), len(a_test))

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
train_idx, tmp_idx = next(gss.split(df,groups=df["eng_n"]))
b_train, b_tmp = df.iloc[train_idx], df.iloc[tmp_idx]

gss = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=SEED)
val_idx, test_idx = next(gss.split(b_tmp, groups=b_tmp["eng_n"]))
b_val, b_test = b_tmp.iloc[val_idx], b_tmp.iloc[test_idx]

print(f"B: {len(b_train)}, {len(b_val)},{len(b_test)}")


def leak(train, val, test):
    tr, va, te = set(train["eng_n"]), set(val["eng_n"]), set(test["eng_n"])
    return len(te & tr), len(va & tr), len(te & va)


print(f"A fuite (Test∩Train, Val∩Train, Test∩Val): {leak(a_train, a_val, a_test)}")
print(f"B fuite (Test∩Train, Val∩Train, Test∩Val): {leak(b_train, b_val, b_test)}")

assert leak(b_train, b_val, b_test) == (0,0,0), "Split N has leakage" 

print("A Test unique:", a_test["eng_n"].nunique())
print("A Val unique:", a_val["eng_n"].nunique())

for name, parts in {"A": (a_train, a_val, a_test), "B": (b_train, b_val, b_test)}.items():
    out = Path(f"data/splits/{name}")
    out.mkdir(parents=True, exist_ok=True)
    for split, part in zip(["train", "val", "test"], parts):
        part = part.rename(columns={"english": "eng", "darija_arabic": "darija_ar"})
        part[["eng", "darija_ar"]].to_csv(out / f"{split}.csv", index=False)
