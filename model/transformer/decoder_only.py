import torch.nn as nn
from torch.nn import init
import torch 
import torch.nn.functional as F
from transformers.modeling_outputs import Seq2SeqLMOutput

from .MHA import MultiHeadAttention

class Block(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.self_attn = MultiHeadAttention(dim, n_heads)
        self.drop1 = nn.Dropout(0.1)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim*4), 
            nn.ReLU(), 
            nn.Linear(dim*4, dim),
            nn.Dropout(0.1)
        )
        

    def forward(self, x, past_key_values=None, use_cache=False):
        sa_out, next_kv = self.self_attn(x, x, x, past_key_values=past_key_values, use_cache=use_cache)
        sa_out = self.drop1(sa_out)
        x = self.norm1(x + sa_out)
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x, 0, next_kv # 0 thay cho lb_loss

class DecoderOnly(nn.Module):
    def __init__(self, config):
        super().__init__()
        embed_dim = config.embed_dim
        vocab_size = config.vocab_size
        block_size = config.block_size 
        n_layers = config.n_layers 
        n_heads = config.n_heads
        self.eos_token_id = config.eos_token_id
        self.pad_token_id = config.pad_token_id
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(block_size, embed_dim)
        self.drop = nn.Dropout(0.1)
        self.layers = nn.ModuleList(
            [Block(embed_dim, n_heads) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
    def forward(self, input_ids, labels=None, past_key_values = None, use_cache = False, **kwargs): 
        B, T = input_ids.shape 
        tok_emb = self.embed(input_ids)
        pos_emb = self.pos_embed(torch.arange(T, device = input_ids.device))
        x = tok_emb + pos_emb 
        x = self.drop(x)
        next_kvs = []
        for i, layer in enumerate(self.layers):
            past = past_key_values[i] if past_key_values else None
            x, _, kv = layer(x, past, use_cache)
            if use_cache:
                next_kvs.append(kv)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if labels is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.reshape(B*T, C)
            labels = labels.reshape(B*T)
            loss = F.cross_entropy(logits, labels, ignore_index=-100)
        if use_cache:
            return Seq2SeqLMOutput(logits=logits, loss=loss), next_kvs
        return Seq2SeqLMOutput(logits=logits, loss=loss)
    
    @torch.no_grad
    def generate(self, input_ids, context_length,  max_new_tokens = 128):
        self.eval()
        batch_size = input_ids.shape[0]
        generated = torch.zeros((batch_size, max_new_tokens), dtype=torch.long, device=input_ids.device)
        finish = torch.zeros((batch_size,), device = input_ids.device) 
        idx_cond = input_ids[:, -context_length : ] 
        past_kv = None
        for i in range(max_new_tokens):
            if past_kv is None: 
                input_step = idx_cond
            else: 
                input_step = idx_cond[:, -1:].contiguous()
            output, next_kvs = self(input_step, use_cache=True, past_kv=past_kv)
            logits = output.logits[:, -1, :]
            probs = F.softmax(logits, dim = -1) 
            idx_next = torch.multinomial(probs, num_samples=1)
            generated[:, i] = idx_next.squeeze(-1)
            idx_cond = torch.cat((idx_cond, idx_next), dim = -1)
            past_kv = next_kvs
            check_end = (idx_next == self.eos_token_id).int()
            finish = torch.logical_or(finish, check_end)
            if finish.all():
                break
        return generated
        
def kaiming_init_weights(m):
    if isinstance (m, (nn.Linear)):
        init.kaiming_normal_(m.weight)