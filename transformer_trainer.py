from transformers import PretrainedConfig, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments
from safetensors.torch import save_file
import torch
import os

from tokenized_data.tokenizer import tokenizer
from model.transformer.decoder_only import DecoderOnly, kaiming_init_weights
# from model.transformer.transformer_model import TransformerModel , kaiming_init_weights
from tokenized_data.load_ed_data import train_dataset, eval_dataset

os.environ["WANDB_API_KEY"] = "e1ca972bcd5ce8fed1316c6115941ba2e37addaf" 
info_model = {
    "folder": "td_tokR_wiki"
}
batch_size = 2

config = PretrainedConfig(
    vocab_size=tokenizer.vocab_size,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
    decoder_start_token_id=tokenizer.pad_token_id,  # hoặc bos_token_id nếu có
    embed_dim = 256,
    block_size = 1024,
    n_layers = 8,
    n_heads = 8,
)
model = DecoderOnly(config).to("cuda")
# model = TransformerModel(config).to("cuda")
model.apply(kaiming_init_weights)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model size: {total_params / 1e6} M")
data_collator = DataCollatorForSeq2Seq(
    tokenizer, 
    model=model, 
    return_tensors="pt", 
    padding="max_length", 
    max_length = 1024, 
    label_pad_token_id=-100
)
if __name__ == "__main__":
    training_args = Seq2SeqTrainingArguments(
        "td_tokR_wiki/",
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

    model.train()
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset= eval_dataset,
        data_collator=data_collator,
    )
    trainer.train() 
    os.makedirs(info_model["folder"], exist_ok=True)
    save_file(info_model["folder"] + "/model.safetensors")
