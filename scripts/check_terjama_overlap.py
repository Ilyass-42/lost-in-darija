import re
import pandas as pd
from datasets import load_dataset

def norm(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"[^\w\s]", " ", s)      # ponctuation -> espace
    return re.sub(r"\s+", " ", s).strip()  # espaces multiples -> un seul

tb = load_dataset("atlasia/TerjamaBench")   # demande ton token HF
print(tb)                                     # affiche les splits et les colonnes

tb_eng = set()
for split in tb.values():
    tb_eng |= {norm(x) for x in split["English"]}

for name in ["Train", "Val", "Test"]:
    df = pd.read_csv(f"data/{name}.csv")
    common = {norm(x) for x in df["eng"]} & tb_eng
    print(f"TerjamaBench ∩ {name}: {len(common)} / {len(tb_eng)}")
    print("  exemples :", list(common)[:5])