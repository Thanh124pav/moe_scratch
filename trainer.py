import os
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_metric
from torch.utils.data import DataLoader
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, AutoModelForSeq2SeqLM
from tqdm import tqdm
import wandb
from safetensors.torch import save_file


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

        print("Setting up training arguments...")
        training_args = Seq2SeqTrainingArguments(
            self.run_name,
            report_to="wandb",
            do_train=True,
            do_eval=True,
            num_train_epochs=7,
            learning_rate=1e-3,
            warmup_ratio=0.05,
            weight_decay=0.01,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            logging_dir='./log',
            group_by_length=True,
            save_strategy="epoch",
            save_total_limit=3,
            eval_strategy="steps",
            eval_steps=1000, 
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
            compute_metrics=compute_metrics,
        )
        
        print("Starting training...")
        trainer.train() 
        print("Training completed, saving model...")
        trainer.save_model(self.folder)
        return self.model

def compute_metrics(eval_preds, tokenizer=None):
    """
    Hàm tính BLEU, ROUGE, BERTScore cho dịch máy.
    eval_preds: dict hoặc tuple (predictions, label_ids)
    tokenizer: tokenizer dùng để decode
    """
    predictions, label_ids = eval_preds[0], eval_preds[1]
    if isinstance(predictions, tuple):  # For some models
        predictions = predictions[0]
    # Chuyển sang list nếu là tensor
    if hasattr(predictions, 'tolist'):
        predictions = predictions.tolist()
    if hasattr(label_ids, 'tolist'):
        label_ids = label_ids.tolist()
    # Loại bỏ -100
    pred_str = [tokenizer.decode([t for t in pred if t != -100 and t != tokenizer.pad_token_id], skip_special_tokens=True) for pred in predictions]
    label_str = [tokenizer.decode([t for t in label if t != -100 and t != tokenizer.pad_token_id], skip_special_tokens=True) for label in label_ids]
    # BLEU
    bleu = load_metric('sacrebleu')
    bleu_score = bleu.compute(predictions=pred_str, references=[[l] for l in label_str])["score"]
    # ROUGE
    rouge = load_metric('rouge')
    rouge_score = rouge.compute(predictions=pred_str, references=label_str, use_stemmer=True)
    # BERTScore (nếu cần, cài bert-score)
    try:
        import bert_score
        P, R, F1 = bert_score.score(pred_str, label_str, lang="vi")
        bertscore = {
            "bertscore_precision": P.mean().item(),
            "bertscore_recall": R.mean().item(),
            "bertscore_f1": F1.mean().item(),
        }
    except ImportError:
        bertscore = {}
    # Tổng hợp
    result = {
        "bleu": bleu_score,
        "rouge1": rouge_score["rouge1"].mid.fmeasure,
        "rouge2": rouge_score["rouge2"].mid.fmeasure,
        "rougeL": rouge_score["rougeL"].mid.fmeasure,
    }
    result.update(bertscore)
    return result
if __name__ == "__main__":
    wandb.init(project="moe", name="moe-decoder-mt") 