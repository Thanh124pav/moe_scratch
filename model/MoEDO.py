import os
import torch
from transformers import PretrainedConfig, DataCollatorForSeq2Seq, AutoTokenizer
import wandb
t = torch.rand(10, 10).cuda()
print(t.device) # should be CUDA
print("PyTorch:", torch.__version__)  
print("CUDA runtime trong wheel:", torch.version.cuda)  
print("Tên GPU:", torch.cuda.get_device_name(0))  
print("Compute Capability:", torch.cuda.get_device_capability(0))  
print("CUDA khả dụng:", torch.cuda.is_available())  

from compute_parmeters import ModelParameterCount
from moe.decoder_only import MoEDecoderModel
from model.moe.deepseek_decoder_only import DeepSeekMoEDecoderModel
# from tokenized_data.tokenizer import tokenizer
tokenizer = AutoTokenizer.from_pretrained('tokenized_data/do_tokenizer')
tokenizer.padding_size = 'left'


os.environ["WANDB_API_KEY"] = "e1ca972bcd5ce8fed1316c6115941ba2e37addaf"

config = PretrainedConfig(
    vocab_size=tokenizer.vocab_size + 1,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
    decoder_start_token_id=tokenizer.pad_token_id,  # VietAI/vit5 dùng pad_token làm start token
    embed_dim = 256,    
    block_size = 1024,
    n_layers = 4,     
    n_heads = 8,
    n_experts = 8,
    top_k_experts = 2, 
    run_name = "decoder_as",
    folder = "saved_model/test_do_tokL_as",
    num_epochs = 7,
    batch_size = 2,
    path_data = "data/summarization/do/",
)

FGconfig = PretrainedConfig(
    vocab_size=tokenizer.vocab_size + 1,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
    decoder_start_token_id=tokenizer.pad_token_id,  # VietAI/vit5 dùng pad_token làm start token
    embed_dim = 256,    
    block_size = 1024,
    n_layers = 4,     
    n_heads = 8,
    n_experts = 16,
    top_k_experts = 4, 
    run_name = "decoder_as",
    folder = "saved_model/test_do_tokL_as",
    num_epochs = 7,
    batch_size = 2,
    path_data = "data/summarization/do/",
)

# model = MoEDecoderModel(config).to('cuda')
# model = DeepSeekMoEDecoderModel(config).to('cuda')

if __name__ == "__main__":
    ModelParameterCount(MoEDecoderModel, config)
    ModelParameterCount(DeepSeekMoEDecoderModel, FGconfig)


