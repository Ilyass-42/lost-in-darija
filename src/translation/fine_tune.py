import pandas as pd
from transformers import MarianTokenizer, MarianMTModel
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ar")
model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-ar")
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
        return {
            "input_ids" :source["input_ids"].squeeze(),
            "attention_mask" :source["attention_mask"].squeeze(),
            "labels" : target["input_ids"].squeeze()
        }
    
train_path = "./data/Train.csv"
Darija_Dataset = DarijaDataset(train_path,tokenizer) 

dataloader = DataLoader(Darija_Dataset,batch_size=16,shuffle=True)

lr = 2e-5
optimizer = AdamW(model.parameters(),lr=lr)
num_epoch = 4

model.train()
for epoch in range(num_epoch):
    for batch in dataloader:
        optimizer.zero_grad()
        output = model(input_ids = batch["input_ids"].to(device) ,
                    attention_mask = batch["attention_mask"].to(device), 
                    labels = batch["labels"].to(device))
        loss = output.loss
        loss.backward()
        optimizer.step()
    print(f"Epoch : {epoch} - Loss {loss.item()}")


model.save_pretrained("./models/fine_tuned_marian")
tokenizer.save_pretrained("./models/fine_tuned_marian")
