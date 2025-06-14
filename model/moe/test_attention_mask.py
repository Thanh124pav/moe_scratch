import torch
import torch.nn as nn
import torch.nn.functional as F

# Hàm attention đơn giản với mask
def scaled_dot_product_attention(q, k, v, mask=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / (q.size(-1) ** 0.5)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    attn = F.softmax(scores, dim=-1)
    output = torch.matmul(attn, v)
    return output, attn

# Block đơn giản chỉ để test mask
class SimpleBlock(nn.Module):
    def __init__(self, embed_dim, n_heads):
        super().__init__()
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads

    def forward(self, x, attention_mask=None):
        B, T, C = x.shape
        def shape(x):
            return x.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, n_heads, T, head_dim)
        q = shape(self.q_proj(x))
        k = shape(self.k_proj(x))
        v = shape(self.v_proj(x))
        # Causal mask
        causal_mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(0)
        if attention_mask is not None:
            attn_mask = attention_mask[:, None, None, :]  # (B, 1, 1, T)
            mask = (causal_mask.bool() & attn_mask.bool()).to(x.device)
        else:
            mask = causal_mask
        out, attn = scaled_dot_product_attention(q, k, v, mask)
        return out, attn, mask

# Hàm test
if __name__ == "__main__":
    torch.manual_seed(42)
    B, T, C = 2, 4, 8
    n_heads = 2
    x = torch.randn(B, T, C)
    # attention_mask: batch 1 có 3 token thực, batch 2 có 2 token thực
    attention_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])
    block = SimpleBlock(C, n_heads)
    out, attn, mask = block(x, attention_mask=attention_mask)
    print("Output shape:", out.shape)
    print("Attention shape:", attn.shape)
    print("Mask shape:", mask.shape)
    print("Mask[0,0]:\n", mask[0,0].int())
    print("Mask[1,0]:\n", mask[1,0].int())
    print("Test kết hợp causal và attention mask thành công!") 