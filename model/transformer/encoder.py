import torch.nn as nn 
import torch 
from transformers.modeling_outputs import BaseModelOutput
from .MHA import MultiHeadAttention 

class EBlock(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.attn = MultiHeadAttention(dim, n_heads) 
        self.drop1 = nn.Dropout(0.1)
        self.norm1 = nn.LayerNorm(dim) 
        self.ff = nn.Sequential(
            nn.Linear(dim, dim*4), 
            nn.ReLU(), 
            nn.Linear(4*dim, dim), 
            nn.Dropout(0.2)
        )
        self.norm2 = nn.LayerNorm(dim)
    def forward(self, x): 
        h, _ = self.attn(x, x, x)
        h = self.drop1(h) 
        x = self.norm1(x + h)
        ff_out = self.ff(x) 
        return self.norm2(x + ff_out), 0

class Encoder(nn.Module):
    def __init__(self, config, shared): 
        super().__init__()
        dim = config.embed_dim 
        n_heads = config.n_heads
        block_size = config.block_size
        n_layers = config.n_layers
        self.embed_tokens = shared 
        self.drop = nn.Dropout(0.1)
        self.pos_embed = nn.Embedding(block_size, dim)
        self.layers = nn.ModuleList(
            [EBlock(dim, n_heads) for _ in range(n_layers)]
        )
    def forward(self, input_ids):
        B, T = input_ids.shape 
        tok_emb = self.embed_tokens(input_ids) 
        pos_emb = self.pos_embed(torch.arange(T, device = input_ids.device))
        x = tok_emb + pos_emb 
        x = self.drop(x)
        for layer in self.layers: 
            x, _ = layer(x) 
        return BaseModelOutput(last_hidden_state=x), 0