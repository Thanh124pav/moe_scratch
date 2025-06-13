import os
import torch
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, AutoModelForSeq2SeqLM


from tokenized_data.load_data import load_data

class Trainer:
    def __init__(self, model, data_collator, dir_name, batch_size, path_data, info_model):
        self.model = model
        self.data_collator = data_collator
        self.dir_name = dir_name
        self.batch_size = batch_size
        self.path_data = path_data
        self.info_model = info_model

    def train_model(self):
        print("Starting data loading...")
        train_dataset, eval_dataset, _ = load_data(self.path_data)
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Eval dataset size: {len(eval_dataset)}")
        
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model size: {total_params / 1e6} M")

        print("Setting up training arguments...")
        training_args = Seq2SeqTrainingArguments(
            self.dir_name,
            report_to="wandb",
            do_train=True,
            do_eval=True,
            num_train_epochs=7,
            learning_rate=1e-5,
            warmup_ratio=0.05,
            weight_decay=0.01,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            logging_dir='./log',
            group_by_length=True,
            save_strategy="epoch",
            save_total_limit=3,
            eval_strategy="steps",
            eval_steps=200, 
            fp16=True,
            remove_unused_columns=True,
        )

        print("Initializing trainer...")
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=self.data_collator,
        )
        
        print("Starting training...")
        trainer.train() 
        print("Training completed, saving model...")
        trainer.save_model(self.info_model["folder"])
        return self.model

