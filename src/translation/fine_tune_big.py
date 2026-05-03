import pandas as pd
from transformers import MarianTokenizer, MarianMTModel, DataCollatorForSeq2Seq,get_cosine_schedule_with_warmup
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from torch.amp import autocast, GradScaler
import os
from peft import LoraConfig, TaskType, get_peft_model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-tc-big-en-ar")
model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-tc-big-en-ar")
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

accumulation_steps = 4
batch_size = 4

collator = DataCollatorForSeq2Seq(tokenizer, model=model)
dataloader = DataLoader(Darija_Dataset, batch_size=batch_size, shuffle=True, collate_fn=collator,num_workers=min(4,os.cpu_count()),pin_memory=pin_memory)
dataloader_validation = DataLoader(Darija_Dataset_Validation, batch_size=batch_size, collate_fn=collator,num_workers=min(4,os.cpu_count()),pin_memory=pin_memory)

lora_config = LoraConfig(
    task_type = TaskType.SEQ_2_SEQ_LM,
    r = 16,
    lora_alpha = 32,
    target_modules = ["q_proj","k_proj","v_proj","out_proj"],
    lora_dropout = 0.1,
    bias="none"
)

model = get_peft_model(model,lora_config)
model.print_trainable_parameters()


lr = 2e-5
optimizer = AdamW(filter(lambda p:p.requires_grad, model.parameters()),lr=lr)
num_epoch = 8
best_val_loss = float("inf")

num_training_steps = num_epoch * len(dataloader) // accumulation_steps
num_warmup_steps   = int(0.06 * num_training_steps)

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps
)


run_name = datetime.now().strftime("fine_tune_big_v2_%Y%m%d_%H%M%S")
writer = SummaryWriter(f"runs/{run_name}")
global_step = 0


try:
    scaler = GradScaler(device_type=device_type)
except TypeError:
    scaler = GradScaler()


model.train()
for epoch in range(num_epoch):
    sum_train_loss = 0
    num_batch_train = 0
    for batch_index,batch in enumerate(dataloader):
        num_batch_train +=1
        

        with autocast(device_type=device_type):
            output = model(input_ids = batch["input_ids"].to(device) ,
                        attention_mask = batch["attention_mask"].to(device), 
                        labels = batch["labels"].to(device))
            loss = output.loss
            sum_train_loss += loss.item()
            loss = loss / accumulation_steps
        scaler.scale(loss).backward()
        if (batch_index+1) % accumulation_steps ==0:
            scaler.step(optimizer)
            optimizer.zero_grad()
            scaler.update()
            scheduler.step()
            global_step +=1

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
        model.save_pretrained("./models/fine_tuned_marian_big")
        tokenizer.save_pretrained("./models/fine_tuned_marian_big")
    model.train()

writer.close()
