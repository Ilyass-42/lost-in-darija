import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from torch.utils.data import DataLoader, Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ar")
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-ar")

model = model.to(device)


class DarijaDataset(Dataset):
    def __init__(self,csv_path,tokenizer):
        df = pd.read_csv(csv_path)
        self.sources = df["eng"].tolist()
        self.targets = df["darija_ar"].tolist()
        self.tokenizer = tokenizer
    def __len__(self):
        return len(self.sources)
    def __getitem__(self, index):
        source = ">>ary<< " + self.sources[index]
        target = self.targets[index]
        source = self.tokenizer(source,return_tensors="pt",padding=False,truncation=True)
        target = self.tokenizer(target,return_tensors="pt",padding=False,truncation=True)
        return source,target