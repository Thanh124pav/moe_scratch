from transformers import PretrainedConfig
from transformers import AutoTokenizer
import torch 
from model.moe.decoder_only import MoEDecoderModel
tokenizer = AutoTokenizer.from_pretrained("VietAI/vit5-base")
config = PretrainedConfig(
    vocab_size=tokenizer.vocab_size,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
    decoder_start_token_id=tokenizer.pad_token_id,  # VietAI/vit5 dùng pad_token làm start token
    embed_dim = 512,    # Tăng embedding dimension
    block_size = 1024,
    n_layers = 6,       # Tăng số layers
    n_heads = 8,
    n_experts = 8,
    top_k_experts = 2, 
)

model = MoEDecoderModel(config).to('cuda')

res = model.generate(torch.tensor([[1, 2, 3]]).to('cuda'), max_length=10)
print(res)
print(tokenizer.decode(res[0]))