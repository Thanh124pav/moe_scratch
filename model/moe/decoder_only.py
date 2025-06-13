from transformers.modeling_outputs import Seq2SeqLMOutput
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torch

from .MultiHeadAttention import MultiHeadAttention
from .MoE import SparseMoE

# hyperparameters

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Block(nn.Module):
    def __init__(self, n_embed, n_heads, num_experts, top_k):
        # n_embed: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        self.mha = MultiHeadAttention(n_embed, n_heads)
        self.smoe = SparseMoE(n_embed, num_experts, top_k)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x, past_kv = None, use_cache = False):
        mha_out, next_kv = self.mha(x,x,x, past_kv, use_cache)
        x = self.ln1(x + mha_out)
        out_moe, lb_loss = self.smoe(x)
        x = self.ln2(x + out_moe)
        return x, lb_loss, next_kv

lb_weight = 0.01
#Finally putting it all together to crease a sparse mixture of experts language model
class MoEDecoderModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        vocab_size = config.vocab_size
        embed_dim = config.embed_dim
        block_size = config.block_size
        n_heads = config.n_heads
        n_experts = config.n_experts
        top_k = config.top_k_experts
        n_layers = config.n_layers
        self.eos_token_id = config.eos_token_id
        self.token_embedding_table = nn.Embedding(vocab_size, embed_dim )
        self.position_embedding_table = nn.Embedding(block_size, embed_dim)
        self.blocks = nn.ModuleList([Block(embed_dim,  n_heads, n_experts , top_k) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(embed_dim) # final layer norm
        self.lm_head = nn.Linear(embed_dim, vocab_size)
        


    def forward(self, input_ids, labels=None, attention_mask=None, past_kv=None, use_cache=False):
        B, T = input_ids.shape
        tok_emb = self.token_embedding_table(input_ids) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        total_lb_loss = 0
        
        if past_kv is None:
            past_kv = [None] * len(self.blocks)
        next_kvs = []
        for i, block in enumerate(self.blocks):
            x, lb_loss, next_kv = block(x, past_kv[i], use_cache)
            if use_cache:
                next_kvs.append(next_kv)
            total_lb_loss += lb_loss
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        # Tính toán các loss
        loss = None
        
        if labels is not None:
            B, T, C = logits.shape
            logits_flat = logits.reshape(B*T, C)
            labels_flat = labels.reshape(B*T)
            loss = F.cross_entropy(logits_flat, labels_flat) + lb_weight * total_lb_loss
            

                
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


