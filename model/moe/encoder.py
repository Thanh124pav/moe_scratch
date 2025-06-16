import torch.nn as nn
import torch
from transformers.modeling_outputs import BaseModelOutput

from .MultiHeadAttention import MultiHeadAttention 
from .MoE import SparseMoE

class EncoderBlock(nn.Module):
    def __init__(self, dim, n_heads, n_experts, top_k):
        super().__init__()
        self.attn = MultiHeadAttention(dim, n_heads)
        self.norm1 = nn.LayerNorm(dim)
        self.drop1 = nn.Dropout(0.1)
        self.moe = SparseMoE(dim, n_experts, top_k)
        self.norm2 = nn.LayerNorm(dim)
        self.drop2 = nn.Dropout(0.1)

    def forward(self, x, attention_mask = None):
        B, T, _ = x.shape
        # Causal mask: (1, 1, T, T)
        causal_mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(0)
        if attention_mask is not None:
            # attention_mask: (B, T) -> (B, 1, 1, T)
            attn_mask = attention_mask.unsqueeze(1).unsqueeze(1)
            # Kết hợp causal mask và attention mask (broadcast)
            mask = (causal_mask.bool() & attn_mask.bool()).to(x.device)
        else:
            mask = causal_mask
        h, _ = self.attn(x, x, x, mask = mask)
        h = self.drop1(h)
        x = self.norm1(x + h)
        moe_out, lb = self.moe(x)
        return self.norm2(x + moe_out), lb


class Encoder(nn.Module):
    main_input_name = "input_ids"
    def __init__(self, config, shared_embedding):
        embed_dim = config.embed_dim 
        vocab_size = config.vocab_size
        block_size = config.block_size 
        n_layers = config.n_layers 
        n_heads = config.n_heads 
        n_experts = config.n_experts 
        top_k = config.top_k_experts
        super().__init__()
        self.embed = shared_embedding
        self.drop = nn.Dropout(0.1)
        self.pos_embed = nn.Embedding(block_size, embed_dim)
        self.layers = nn.ModuleList([EncoderBlock(embed_dim, n_heads, n_experts, top_k) for _ in range(n_layers)])
    def forward(self, input_ids, attention_mask = None, **kwargs):
        B, T = input_ids.shape 
        tok_emb = self.embed(input_ids)
        pos_emb = self.pos_embed(torch.arange(T, device = input_ids.device))
        x = tok_emb + pos_emb
        x = self.drop(x)

        lb_loss = 0
        for layer in self.layers:
            x, lb = layer(x, attention_mask = attention_mask) 
            lb_loss += lb 
        return BaseModelOutput(last_hidden_state=x), lb_loss
    