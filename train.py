import os
import torch
from transformers import PretrainedConfig, DataCollatorForSeq2Seq
import wandb
t = torch.rand(10, 10).cuda()
print(t.device) # should be CUDA
print("PyTorch:", torch.__version__)  
print("CUDA runtime trong wheel:", torch.version.cuda)  
print("Tên GPU:", torch.cuda.get_device_name(0))  
print("Compute Capability:", torch.cuda.get_device_capability(0))  
print("CUDA khả dụng:", torch.cuda.is_available())  


from model.moe.decoder_only import MoEDecoderModel
from tokenized_data.tokenizer import tokenizer
from trainer import Trainer
wandb.init(project="moe-wandb-demo")

os.environ["WANDB_API_KEY"] = "e1ca972bcd5ce8fed1316c6115941ba2e37addaf"

config = PretrainedConfig(
    vocab_size=tokenizer.vocab_size,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
    decoder_start_token_id=tokenizer.pad_token_id,  # VietAI/vit5 dùng pad_token làm start token
    embed_dim = 256,    # Tăng embedding dimension
    block_size = 1024,
    n_layers = 4,       # Tăng số layers
    n_heads = 8,
    n_experts = 8,
    top_k_experts = 2, 
    run_name = "decoder_mt",
    folder = "d_tokR_mt",
    num_epochs = 7,
    batch_size = 4,
    path_data = "data/machine_translation/tokenized_right/",
)

model = MoEDecoderModel(config).to('cuda')
data_collator = DataCollatorForSeq2Seq(
    tokenizer, 
    model=model, 
    return_tensors="pt", 
    padding="max_length", 
    max_length = config.block_size, 
    label_pad_token_id=-100
)

trainer = Trainer(model,data_collator,config)
trainer.train_model()   


