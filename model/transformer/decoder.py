import torch.nn as nn
import torch 

from .MHA import MultiHeadAttention 

class DBlock(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.self_attn = MultiHeadAttention(dim, n_heads)
        self.drop1 = nn.Dropout(0.1)
        self.norm1 = nn.LayerNorm(dim)
        self.cross_attn = MultiHeadAttention(dim, n_heads)
        self.drop2 = nn.Dropout(0.1)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim*4), 
            nn.ReLU(), 
            nn.Linear(dim*4, dim),
            nn.Dropout(0.1)
        )
        self.drop3 = nn.Dropout(0.1)
        self.norm3 = nn.LayerNorm(dim)
        

    def forward(self, x, enc, past_key_values=None, use_cache=False):
        sa_out, next_kv = self.self_attn(x, x, x, past_key_values=past_key_values, use_cache=use_cache)
        sa_out = self.drop1(sa_out)
        x = self.norm1(x + sa_out)
        ca_out, _ = self.cross_attn(x, enc, enc)
        ca_out = self.drop2(ca_out)
        x = self.norm2(x + ca_out)
        ff_out = self.ff(x)
        x = self.norm3(x + ff_out)
        return x, 0, next_kv # 0 thay cho lb_loss

class Decoder(nn.Module):
    def __init__(self, config, shared_embedding):
        super().__init__()
        embed_dim = config.embed_dim 
        block_size = config.block_size 
        n_layers = config.n_layers 
        n_heads = config.n_heads 
        self.embed = shared_embedding
        self.drop = nn.Dropout(0.1)
        self.pos_embed = nn.Embedding(block_size, embed_dim)
        self.layers = nn.ModuleList([DBlock(embed_dim, n_heads) for _ in range(n_layers)])
    def forward(self, decoder_input_ids, enc_out, past_key_values = None, use_cache = False, attention_mask = None, **kwargs ):
        B, T = decoder_input_ids.shape
        tok_emb = self.embed(decoder_input_ids)
        pos_emb = self.pos_embed(torch.arange(T, device = decoder_input_ids.device))
        y = tok_emb + pos_emb
        y = self.drop(y)
        next_kvs = []
        for i, layer in enumerate(self.layers):
            past = past_key_values[i] if past_key_values else None
            y, _, kv = layer(y, enc_out, past, use_cache)
            if use_cache:
                next_kvs.append(kv)
        if use_cache is None: 
            next_kvs = None
        return y, 0, next_kvs