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

    def forward(self, x, past_kv=None, use_cache=False, attention_mask=None):
        B, T, C = x.shape
        mask = None
        if not use_cache:
            # Causal mask: (1, 1, T, T)
            causal_mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(0)
            if attention_mask is not None:
                # attention_mask: (B, T) -> (B, 1, 1, T)
                attn_mask = attention_mask.unsqueeze(1).unsqueeze(1)
                # Kết hợp causal mask và attention mask (broadcast)
                mask = (causal_mask.bool() & attn_mask.bool()).to(x.device)
            else:
                mask = causal_mask
        mha_out, next_kv = self.mha(x, x, x, past_kv, use_cache, mask)
        x = self.ln1(x + mha_out)
        out_moe, lb_loss = self.smoe(x)
        x = self.ln2(x + out_moe)
        #print(f"moe_out: {x}")
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
        #print(input_ids)
        ##print(labels)
        B, T = input_ids.shape
        tok_emb = self.token_embedding_table(input_ids) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        total_lb_loss = 0
        
        if past_kv is None:
            past_kv = [None] * len(self.blocks)
        next_kvs = []
        for i, block in enumerate(self.blocks):
            x, lb_loss, next_kv = block(x, past_kv[i], use_cache, attention_mask)
            if use_cache:
                next_kvs.append(next_kv)
            total_lb_loss += lb_loss
        x = self.ln_f(x) # (B,T,C)
        ##print(f"x: {x}")
        logits = self.lm_head(x) # (B,T,vocab_size)

        # Tính toán các loss
        loss = None
        
        if labels is not None:
            B, T, C = logits.shape
            labels = labels[...,1:, :].contiguos()
            logits = logits[...,:-1, :].contiguos()
            logits_2d = logits.view(B*T, -1)
            labels_2d = labels.view(B*T)
            ##print(f"logits_2d: {logits_2d}")
            ##print(f"labels_2d: {labels_2d}")
            loss = F.cross_entropy(logits_2d, labels_2d, ignore_index=-100) + lb_weight * total_lb_loss
            #print(f"loss: {loss}")
        if use_cache: 
            return Seq2SeqLMOutput(logits=logits, loss=loss), next_kvs
        return Seq2SeqLMOutput(logits=logits, loss=loss)
    
    @torch.no_grad
    def generate(self, input_ids, max_length=None, max_new_tokens=64, context_length=None, 
                 pad_token_id=None, eos_token_id=None,top_k=5, top_p=0.6, temperature=1.5, **kwargs):

        self.eval()
        batch_size = input_ids.shape[0]
        
        # Handle parameters
        if context_length is None:
            context_length = input_ids.shape[1]  # Use all input
        
        generated = torch.zeros((batch_size, max_new_tokens), dtype=torch.long, device=input_ids.device)
        finish = torch.zeros((batch_size,), device=input_ids.device, dtype=torch.bool) 
        idx_cond = input_ids[:, -context_length:] 
        past_kv = None
        
        for i in range(max_new_tokens):
            if past_kv is None: 
                input_step = idx_cond
            else: 
                input_step = idx_cond[:, -1:].contiguous()
                
            output, next_kvs = self(input_step, use_cache=True, past_kv=past_kv)
            logits = output.logits[:, -1, :]/temperature

            # Top-k sampling
            if top_k > 0:
                top_k = min(top_k, logits.size(-1))
                values, indices = torch.topk(logits, top_k)
                mask = torch.full_like(logits, float('-inf'))
                mask.scatter_(1, indices, values)
                logits = mask

            # Top-p (nucleus) sampling
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_mask = cumulative_probs > top_p
                # Shift mask right to keep at least one token
                sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
                sorted_mask[..., 0] = 0
                indices_to_remove = sorted_mask.scatter(1, sorted_indices, sorted_mask)
                logits = logits.masked_fill(indices_to_remove, float('-inf'))


            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            generated[:, i] = idx_next.squeeze(-1)
            idx_cond = torch.cat((idx_cond, idx_next), dim=-1)
            past_kv = next_kvs
            
            check_end = (idx_next.squeeze(-1) == eos_token_id)
            if isinstance(check_end, bool):
                if check_end:
                    break
            else:
                finish = torch.logical_or(finish, check_end)
                if finish.all():
                    break
                
        return generated

def kaiming_init_weights(m):
    if isinstance (m, (nn.Linear)):
        init.kaiming_normal_(m.weight)


