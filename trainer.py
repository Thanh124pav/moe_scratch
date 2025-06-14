import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, AutoModelForSeq2SeqLM
from tqdm import tqdm
import wandb
from safetensors.torch import save_file

wandb.init(project="moe", name="moe-decoder-mt") 
os.environ["WANDB_API_KEY"] = "e1ca972bcd5ce8fed1316c6115941ba2e37addaf"


from tokenized_data.load_data import load_data

class Trainer:
    def __init__(self, model, data_collator,config):
        self.model = model
        self.data_collator = data_collator
        self.run_name = config.run_name
        self.batch_size = config.batch_size
        self.path_data = config.path_data
        self.folder = config.folder
        self.num_epochs = config.num_epochs
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def train_model(self):
        print("Starting data loading...")
        train_dataset, eval_dataset, _ = load_data(self.path_data)
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Eval dataset size: {len(eval_dataset)}")
        
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model size: {total_params / 1e6} M")

        train_dataloader = DataLoader(train_dataset, collate_fn=self.data_collator, batch_size=self.batch_size, shuffle=True)
        eval_dataloader = DataLoader(eval_dataset, collate_fn=self.data_collator, batch_size=self.batch_size, shuffle=False)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5)
        tokens_seen = 0
        global_step = 0
        total_loss = 0
        for epoch in range(self.num_epochs):
            for i, batch in enumerate(tqdm(train_dataloader)):
                optimizer.zero_grad()
                self.model.train()

                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                outputs = self.model(input_ids, labels=labels, attention_mask=attention_mask)

                loss = outputs.loss
                train_batches = len(train_dataloader) // self.batch_size
                wandb.log({"train/loss": loss.item(), "train/step": i + epoch * train_batches})
                total_loss += loss.item()
                loss.backward()
                optimizer.step()

                tokens_seen += input_ids.shape[1]
                global_step += 1
                if global_step %20 == 0: 
                    avg_loss = total_loss / 200
                    print(f"Epoch {epoch}, Step {i}, Loss: {avg_loss}")
                    total_loss = 0
                    self.model.eval()
                    eval_loss = 0
                    for batch in eval_dataloader:
                        input_ids = batch['input_ids'].to(self.device)
                        attention_mask = batch['attention_mask'].to(self.device)
                        labels = batch['labels'].to(self.device)
                        with torch.no_grad():
                            outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                            eval_loss += outputs.loss.item()
                    eval_loss /= len(eval_dataloader)
                    print(f"Epoch {epoch}, Eval Loss: {eval_loss}")
                    wandb.log({"eval/loss": eval_loss, "eval/step": i + epoch * len(eval_dataloader)})
                    self.model.train()
                    if eval_loss < best_eval_loss:
                        best_eval_loss = eval_loss
                        save_file(self.model.state_dict(), f"{self.folder}/model.safetensors")
                        print(f"Model saved to {self.folder}")


