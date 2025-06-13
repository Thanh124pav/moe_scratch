import os
import torch
from transformers import PretrainedConfig, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments, AutoModelForSeq2SeqLM
vit5_model = AutoModelForSeq2SeqLM.from_pretrained("VietAI/vit5-base")
t = torch.rand(10, 10).cuda()
print(t.device) # should be CUDA
print("PyTorch:", torch.__version__)  
print("CUDA runtime trong wheel:", torch.version.cuda)  
print("Tên GPU:", torch.cuda.get_device_name(0))  
print("Compute Capability:", torch.cuda.get_device_capability(0))  
print("CUDA khả dụng:", torch.cuda.is_available())  


from model.moe.decoder_only import MoEDecoderModel
from tokenized_data.load_data import train_dataset, eval_dataset
from tokenized_data.tokenizer import tokenizer, vocab_size 

os.environ["WANDB_API_KEY"] = "e1ca972bcd5ce8fed1316c6115941ba2e37addaf"

config = PretrainedConfig(
    vocab_size=tokenizer.vocab_size,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
    decoder_start_token_id=tokenizer.pad_token_id,  # hoặc bos_token_id nếu có
    embed_dim = 256,
    block_size = 1024,
    n_layers = 4,
    n_heads = 8,
    n_experts = 8,
    top_k_experts = 2,
    teacher_model = vit5_model, 
)

batch_size = 2
model = MoEDecoderModel(config).to('cuda')
data_collator = DataCollatorForSeq2Seq(
    tokenizer, 
    model=model, 
    return_tensors="pt", 
    padding="max_length", 
    max_length = config.block_size, 
    label_pad_token_id=-100
)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model size: {total_params / 1e6} M")

training_args = Seq2SeqTrainingArguments(
    "d_6layers_tokL_wiki/",
    report_to="wandb",
    do_train=True,
    do_eval=True,
    num_train_epochs=7,
    learning_rate=1e-5,
    warmup_ratio=0.05,
    weight_decay=0.01,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    logging_dir='./log',
    group_by_length=True,
    save_strategy="epoch",
    save_total_limit=3,
    eval_strategy="steps",
    eval_steps=200, 
    fp16=True,
    remove_unused_columns=True,
)

info_model = {
    "folder": "d_tokL_wiki",
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

