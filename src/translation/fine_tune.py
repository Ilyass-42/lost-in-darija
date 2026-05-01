import pandas as pd
from transformers import MarianTokenizer, MarianMTModel, DataCollatorForSeq2Seq,get_cosine_schedule_with_warmup
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from torch.amp import autocast, GradScaler
import os

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
        self.tokenized_source = self.tokenizer([">>ary<< "+source for source in self.sources],return_tensors=None,padding=False,truncation=True)
        self.tokenized_target = self.tokenizer(self.targets,return_tensors=None,padding=False,truncation=True)

    def __len__(self):
        return len(self.sources)
    def __getitem__(self, index):
        return {
            "input_ids":self.tokenized_source["input_ids"][index],
            "labels":self.tokenized_target["input_ids"][index],
            "attention_mask":self.tokenized_source["attention_mask"][index]
            }

device_type = device.type
pin_memory = (device_type=="cuda")


train_path = "./data/Train.csv"
Darija_Dataset = DarijaDataset(train_path,tokenizer) 

val_path = "./data/Val.csv"
Darija_Dataset_Validation = DarijaDataset(val_path,tokenizer)

collator = DataCollatorForSeq2Seq(tokenizer, model=model)
dataloader = DataLoader(Darija_Dataset, batch_size=16, shuffle=True, collate_fn=collator,num_workers=min(4,os.cpu_count()),pin_memory=pin_memory)
dataloader_validation = DataLoader(Darija_Dataset_Validation, batch_size=16, collate_fn=collator,num_workers=min(4,os.cpu_count()),pin_memory=pin_memory)


lr = 2e-5
optimizer = AdamW(model.parameters(),lr=lr)
num_epoch = 8
best_val_loss = float("inf")

num_training_steps = num_epoch * len(dataloader)
num_warmup_steps   = int(0.06 * num_training_steps)

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps
)


run_name = datetime.now().strftime("fine_tune_v4_%Y%m%d_%H%M%S")
writer = SummaryWriter(f"runs/{run_name}")
global_step = 0



scaler = GradScaler(device_type=device_type)

model.train()
for epoch in range(num_epoch):
    sum_train_loss = 0
    num_batch_train = 0
    for batch in dataloader:
        num_batch_train +=1
        global_step +=1

        optimizer.zero_grad()
        with autocast(device_type=device_type):
            output = model(input_ids = batch["input_ids"].to(device) ,
                        attention_mask = batch["attention_mask"].to(device), 
                        labels = batch["labels"].to(device))
            loss = output.loss
        sum_train_loss += loss.item()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        writer.add_scalar("Loss/train_batch",loss.item(),global_step)
        writer.add_scalar("LR",scheduler.get_last_lr()[0],global_step)
    avg_train_loss = sum_train_loss / num_batch_train
    writer.add_scalar("Loss/train_epoch", avg_train_loss, epoch)
    print(f"Epoch : {epoch} - Train Loss {avg_train_loss}")
    
    model.eval()
    
    sum_val_loss = 0
    num_batch_val = 0
    for batch in dataloader_validation:
        num_batch_val +=1
        with torch.no_grad():
            with autocast(device_type=device_type):
                val_output = model(input_ids = batch["input_ids"].to(device) ,
                            attention_mask = batch["attention_mask"].to(device), 
                            labels = batch["labels"].to(device))
                val_loss = val_output.loss.item()
            sum_val_loss += val_loss

    
    avg_val_loss = sum_val_loss/num_batch_val
    writer.add_scalar("Loss/val_epoch", avg_val_loss, epoch)
    print(f"Epoch : {epoch} - Validation Loss {avg_val_loss}")
    if avg_val_loss <= best_val_loss:
        best_val_loss = avg_val_loss
        model.save_pretrained("./models/fine_tuned_marian")
        tokenizer.save_pretrained("./models/fine_tuned_marian")
    model.train()

writer.close()
