import torch.nn as nn
import torch 

from model.moe.MultiHeadAttention import MultiHeadAttention 
from model.moe.MoE import SparseMoE

class DecoderBlock(nn.Module):
    def __init__(self, dim, n_heads, n_experts, top_k):
        super().__init__()
        self.self_attn = MultiHeadAttention(dim, n_heads)
        self.drop1 = nn.Dropout(0.1)
        self.norm1 = nn.LayerNorm(dim)
        self.cross_attn = MultiHeadAttention(dim, n_heads)
        self.drop2 = nn.Dropout(0.1)
        self.norm2 = nn.LayerNorm(dim)
        self.moe = SparseMoE(dim, n_experts, top_k)
        self.drop3 = nn.Dropout(0.1)
        self.norm3 = nn.LayerNorm(dim)
        

    def forward(self, x, enc, past_key_values=None, use_cache=False):
        sa_out, next_kv = self.self_attn(x, x, x, past_key_values=past_key_values, use_cache=use_cache)
        sa_out = self.drop1(sa_out)
        x = self.norm1(x + sa_out)
        ca_out, _ = self.cross_attn(x, enc, enc)
        ca_out = self.drop2(ca_out)
        x = self.norm2(x + ca_out)
        moe_out, lb = self.moe(x)
        moe_out = self.drop3(moe_out)
        x = self.norm3(x + moe_out)
        return x, lb, next_kv

class Decoder(nn.Module):
    main_input_name = "decoder_input_ids"
    def __init__(self, config, shared_embedding):
        super().__init__()
        embed_dim = config.embed_dim 
        vocab_size = config.vocab_size
        block_size = config.block_size 
        n_layers = config.n_layers 
        n_heads = config.n_heads 
        n_experts = config.n_experts 
        top_k = config.top_k_experts
        self.embed = shared_embedding
        self.drop = nn.Dropout(0.1)
        self.pos_embed = nn.Embedding(block_size, embed_dim)
        self.layers = nn.ModuleList([DecoderBlock(embed_dim, n_heads, n_experts, top_k) for _ in range(n_layers)])
        
    def forward(self, decoder_input_ids, enc_out, past_key_values=None, use_cache=False, 
                attention_mask=None, **kwargs):
        B, T = decoder_input_ids.shape
        tok_emb = self.embed(decoder_input_ids)
        pos_emb = self.pos_embed(torch.arange(T, device = decoder_input_ids.device))
        y = tok_emb + pos_emb
        y = self.drop(y)
        next_kvs = []
        lb_loss = 0
        
        for i, layer in enumerate(self.layers):
            past = past_key_values[i] if past_key_values else None
            y, lb, kv = layer(y, enc_out, past, use_cache)
            lb_loss += lb
            if use_cache:
                next_kvs.append(kv)
        if use_cache is None: 
            next_kvs = None
            
        return y, lb_loss, next_kvs