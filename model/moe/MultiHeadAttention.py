import torch.nn as nn 
import torch
import torch.nn.functional as F

block_size = 1024
def scaled_dot_product_attention(q, k, v, mask=None):
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32, device=q.device))
    # (B, n_heads, T, d_head) @ (B, n_heads, d_head, T') => (B, n_heads, T, T')
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    attn = F.softmax(scores, dim=-1) 
    output = torch.matmul(attn, v) # (B, n_heads, T, T') @ (B, n_heads, T', d_head) => (B, n_heads, T, d_head)
    return output, attn 

class MultiHeadAttention(nn.Module):
    '''Return MHA output and (k,v)'''
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.k_cache = None  # (B, num_heads, T', d_head)
        self.v_cache = None # # (B, num_heads, T', d_head)
    def clear_cache(self):
        self.k_cache = None 
        self.v_cache = None
    def forward(self, query, key, value, past_key_values=None, use_cache=False, mask = None):
        '''Return MHA output and (k,v)'''
        B, T, C = query.size()
        def shape(x): return x.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2) # (B, T, C) => (B, T, num_heads, head_dim) => (B, num_heads, T, d_head)
        def unshape(x): return x.transpose(1, 2).contiguous().view(B, T, C) # (B, num_heads, T, head_dim) => (B, T, num_heads, head_dim) => (B, T, C)

        q = shape(self.q_proj(query)) # (B, num_heads, T, d_head)
        k = shape(self.k_proj(key)) # (B, num_heads, T, d_head)
        v = shape(self.v_proj(value)) # (B, num_heads, T, d_head)

        if use_cache:
            if past_key_values is not None:
                past_k, past_v = past_key_values
                k_cache = torch.cat([past_k, k], dim = 2)
                v_cache = torch.cat([past_v, v], dim = 2)
                if k_cache.size(2) > block_size: 
                    k_cache = k_cache[:, :, 1:, :]
                    v_cache = v_cache[:, :, 1:, :]
                k = k_cache # (B, num_heads, T + T', d_head)
                v = v_cache # (B, num_heads, T + T', d_head)
        # causal_mask = torch.tril(torch.ones(q.size(2), k.size(2))).to(q.device)
        output, _ = scaled_dot_product_attention(q, k, v, mask) # (B, n_heads, T, d_head)
        output = unshape(output) # (B, T, C)
        next_kv = (k, v) if use_cache else None
        return self.out_proj(output), next_kv  #  (B, T, C) , (B, num_heads, T + T', d_head) 