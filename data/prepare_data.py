import re
from datasets import load_dataset
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

def is_arabic(text):
    if not text or len(text)==0:
        return False
    arabic_chars = len(re.findall(r'[\u0600-\u06FF]', text))
    return arabic_chars/len(text) > 0.5


def load_and_prepare(subset_name):
    dataset = load_dataset("atlasia/darija_english", subset_name)
    df = dataset["train"].to_pandas()
    df.rename(columns={"darija":"darija_ar","english":"eng"},inplace=True)
    df = df[["eng", "darija_ar"]]
    df = df[df["darija_ar"].apply(is_arabic)]
    return df

train_path = Path(__file__).parent/"Train_v5_backup.csv" #data/Train.csv
val_path = Path(__file__).parent/"Val_v5_backup.csv"

df_comments = load_and_prepare("comments")
df_web_data = load_and_prepare("web_data")
df_train = pd.read_csv(train_path)
df_val = pd.read_csv(val_path)

df_merged = pd.concat([df_train,df_comments,df_web_data,df_val],ignore_index=True)

df_merged.drop_duplicates(subset=["eng","darija_ar"],inplace=True)

df_new_train,df_new_val = train_test_split(df_merged,test_size=0.10,random_state=42)


output_train = Path(__file__).parent/"Train.csv"
output_val = Path(__file__).parent/"Val.csv"


df_new_train.to_csv(output_train,index = False)
df_new_val.to_csv(output_val,index = False)


