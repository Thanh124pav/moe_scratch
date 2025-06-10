import torch
import os
t = torch.rand(10, 10).cuda()
print(t.device) # should be CUDA
print("PyTorch:", torch.__version__)  
print("CUDA runtime trong wheel:", torch.version.cuda)  
print("Tên GPU:", torch.cuda.get_device_name(0))  
print("Compute Capability:", torch.cuda.get_device_capability(0))  
print("CUDA khả dụng:", torch.cuda.is_available())  
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments
from tokenizer import tokenizer, vocab_size 
os.environ["WANDB_API_KEY"] = "e1ca972bcd5ce8fed1316c6115941ba2e37addaf"

from moe_decoder import MoEDecoderModel
from load_data import train_dataset, eval_dataset

config = {
    "vocab_size": vocab_size, 
    "embed_dim": 256, 
    "block_size": 1024, 
    "n_heads": 8, 
    "n_experts": 8, 
    "top_k": 2,
    "n_layers": 4,
}
batch_size = 4
model = MoEDecoderModel(config).to('cuda')
data_collator = DataCollatorForSeq2Seq(
    tokenizer, 
    model=model, 
    return_tensors="pt", 
    padding="max_length", 
    max_length = config["block_size"], 
    label_pad_token_id=-100
)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model size: {total_params / 1e6} M")

training_args = Seq2SeqTrainingArguments(
    "moe_gpu_v3/",
    report_to="wandb",
    do_train=True,
    do_eval=True,
    num_train_epochs=7,
    learning_rate=1e-8,
    warmup_ratio=0.05,
    weight_decay=0.01,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    logging_dir='./log',
    group_by_length=True,
    save_strategy="epoch",
    save_total_limit=3,
    eval_strategy="steps",
    eval_steps=2000, 
    fp16=True,
    remove_unused_columns=True,
)

info_model = {
    "folder": "d_tokR_wiki",
}
if __name__ == "__main__": 
    model.train()
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset= eval_dataset,
        data_collator=data_collator,
    )
    trainer.train() 
    trainer.save_model(info_model["folder"])

