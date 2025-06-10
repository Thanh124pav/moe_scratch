import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutput
from transformers import PreTrainedModel, PretrainedConfig, AutoTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import Seq2SeqLMOutput
import os
import evaluate
import numpy as np 
from tqdm import tqdm
from datasets import load_from_disk

os.environ["WANDB_API_KEY"] = "e1ca972bcd5ce8fed1316c6115941ba2e37addaf"
torch.manual_seed(1337)

tokenizer = AutoTokenizer.from_pretrained("VietAI/vit5-base")
vocab_size = tokenizer.vocab_size
n_embed = 256
num_heads = 8
num_experts = 8
num_layers = 2
top_k = 2
dropout = 0.1


# output, attn_weights
def scaled_dot_product_attention(q, k, v, mask=None):
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32, device=q.device))
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    attn = F.softmax(scores, dim=-1)
    return torch.matmul(attn, v), attn

# output, next (k, v)
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, past_kv=None, use_cache=False):
        B, T, C = query.size()
        def shape(x): return x.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        def unshape(x): return x.transpose(1, 2).contiguous().view(B, T, C)

        q = shape(self.q_proj(query))
        k = shape(self.k_proj(key))
        v = shape(self.v_proj(value))

        if past_kv is not None:
            pk, pv = past_kv
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)

        output, _ = scaled_dot_product_attention(q, k, v)
        output = unshape(output)
        next_kv = (k, v) if use_cache else None
        return self.out_proj(output), next_kv

# === MoE Components ===
class Expert(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.ReLU(),
            nn.Linear(4 * dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class NoisyTopkRouter(nn.Module):
    def __init__(self, dim, num_experts, top_k):
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        self.linear = nn.Linear(dim, num_experts)
        self.noise_linear = nn.Linear(dim, num_experts)

    def forward(self, x):
        logits = self.linear(x)
        noise = torch.randn_like(logits) * F.softplus(self.noise_linear(x))
        noisy_logits = logits + noise
        top_logits, indices = noisy_logits.topk(self.top_k, dim=-1)
        mask = torch.full_like(noisy_logits, float('-inf')).scatter(-1, indices, top_logits)
        probs = F.softmax(mask, dim=-1)

        prob_mean = probs.mean(dim=(0, 1))
        expert_mask = torch.zeros_like(probs).scatter(-1, indices, 1.0)
        prob_count = expert_mask.mean(dim=(0, 1))
        lb_loss = self.num_experts * (prob_mean * prob_count).sum()

        return probs, indices, lb_loss

class SparseMoE(nn.Module):
    def __init__(self, dim, num_experts, top_k):
        super().__init__()
        self.router = NoisyTopkRouter(dim, num_experts, top_k)
        self.experts = nn.ModuleList([Expert(dim) for _ in range(num_experts)])

    def forward(self, x):
        B, T, D = x.shape
        probs, indices, lb_loss = self.router(x)
        out = torch.zeros_like(x)
        flat_x = x.view(-1, D)
        flat_probs = probs.view(-1, probs.size(-1))

        for i, expert in enumerate(self.experts):
            mask = (indices == i).any(dim=-1).view(-1)
            selected = torch.nonzero(mask).squeeze(-1)
            if selected.numel() > 0:
                expert_in = flat_x[selected]
                expert_out = expert(expert_in)
                weights = flat_probs[selected, i].unsqueeze(1)
                out.view(-1, D).index_add_(0, selected, expert_out * weights)

        return out.view(B, T, D), lb_loss

# === Transformer Blocks ===
class EncoderLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = MultiHeadAttention(dim, num_heads)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.moe = SparseMoE(dim, num_experts, top_k)

    def forward(self, x):
        h, _ = self.attn(x, x, x)
        x = self.norm1(x + h)
        moe_out, lb = self.moe(x)
        return self.norm2(x + moe_out), lb

class DecoderLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.self_attn = MultiHeadAttention(dim, num_heads)
        self.cross_attn = MultiHeadAttention(dim, num_heads)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.moe = SparseMoE(dim, num_experts, top_k)

    def forward(self, x, enc, past_kv=None, use_cache=False):
        sa_out, next_kv = self.self_attn(x, x, x, past_kv=past_kv, use_cache=use_cache)
        x = self.norm1(x + sa_out)
        ca_out, _ = self.cross_attn(x, enc, enc)
        x = self.norm2(x + ca_out)
        moe_out, lb = self.moe(x)
        x = self.norm3(x + moe_out)
        return x, lb, next_kv

    
lb_weight = 0.01
# === Full Encoder-Decoder Model ===
class EncoderDecoderMoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed = nn.Embedding(config.vocab_size, n_embed)
        self.enc_layers = nn.ModuleList([EncoderLayer(n_embed) for _ in range(num_layers)])
        self.dec_layers = nn.ModuleList([DecoderLayer(n_embed) for _ in range(num_layers)])
        self.out_proj = nn.Linear(n_embed, config.vocab_size)
    def forward(self, input_ids=None, decoder_input_ids=None, labels=None,
                attention_mask=None, past_key_values=None, use_cache=False, **kwargs):
        lb_loss = 0 
        x = self.embed(input_ids)
        for layer in self.enc_layers:
            x, lb = layer(x)
            lb_loss += lb 
            enc_out = x   
        
        if decoder_input_ids is None:
            decoder_start_token_id = tokenizer.pad_token_id
            if labels is not None:
                decoder_input_ids = labels.new_zeros(labels.shape)
                decoder_input_ids[:, 1:] = labels[:, :-1]
                decoder_input_ids[:, 0] = decoder_start_token_id
                decoder_input_ids = torch.where(
                    decoder_input_ids == -100,
                    tokenizer.pad_token_id,
                    decoder_input_ids
                )
            else: 
                #print("generate")
                batch_size = input_ids.shape[0]
                decoder_input_ids = input_ids.new_full((batch_size, 1), decoder_start_token_id)
        #print(f"decoder_input_ids is None? {decoder_input_ids == None}")
        y = self.embed(decoder_input_ids)
        next_past = []
        for i, layer in enumerate(self.dec_layers):
            past = past_key_values[i] if past_key_values else None
            y, lb, kv = layer(y, enc_out, past, use_cache)
            lb_loss += lb
            if use_cache:
                next_past.append(kv)

        logits = self.out_proj(y)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
            loss += lb_weight*lb_loss
        return Seq2SeqLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=tuple(next_past) if use_cache else None,
        )
    def generate(self, input_ids, context_length, max_new_tokens=256):
        self.eval()
        batch_size = input_ids.shape[0]
        generated = torch.zeros((batch_size, max_new_tokens), dtype=torch.long, device=input_ids.device)
        finish = torch.zeros((batch_size,), device=input_ids.device, dtype=torch.bool)
        # 1. Lưu cache
        past_key_values = None

        for i in range(max_new_tokens):
            if i == 0:
                idx_cond = input_ids[:, -context_length:]
            else:
                idx_cond = input_ids[:, -1:]  # Chỉ truyền token mới nhất

            with torch.no_grad():
                output = self(
                    idx_cond,
                    past_key_values=past_key_values,
                    use_cache=True,
                    decoder_input_ids=input_ids if i == 0 else None  # Tùy forward bạn xử lý decoder_input_ids/inputs thế nào
                )
                logits = output.logits
                logits = logits[:, -1, :]
                past_key_values = output.past_key_values

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            generated[:, i] = idx_next.squeeze(-1)
            input_ids = torch.cat((input_ids, idx_next), dim=-1)
            finish = torch.logical_or(finish, (idx_next.squeeze(-1) == tokenizer.eos_token_id))
            if finish.all():
                break

        return generated


def kaiming_init_weights(m):
    if isinstance (m, (nn.Linear)):
        init.kaiming_normal_(m.weight)

config = PretrainedConfig(
    is_encoder_decoder = True,
    vocab_size=tokenizer.vocab_size,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
    decoder_start_token_id=tokenizer.pad_token_id  # hoặc bos_token_id nếu có
)


model = EncoderDecoderMoE(config).to("cuda")
model.apply(kaiming_init_weights)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model size: {total_params / 1e6} M")
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, return_tensors="pt", 
                                       padding="max_length", max_length = 1024, label_pad_token_id=-100)

if __name__ == "__main__":
    # ---------- Train ---------------
    training_args = Seq2SeqTrainingArguments("moe_ed_scratch_v3/",
                                        report_to="wandb",
                                        do_train=True,
                                        do_eval=True,
                                        num_train_epochs=7,
                                        learning_rate=1e-5,
                                        warmup_ratio=0.05,
                                        weight_decay=0.01,
                                        per_device_train_batch_size=16,
                                        per_device_eval_batch_size=16,
                                        logging_dir='./log',
                                        group_by_length=True,
                                        save_strategy="epoch",
                                        save_total_limit=3,
                                        eval_strategy="steps",
                                        eval_steps=200, 
                                        fp16=True,
                                        remove_unused_columns=True,
                                        )

    from load_data import train_dataset, eval_dataset, test_dataset
    model.train()
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset= eval_dataset,
        data_collator=data_collator,
    )
    trainer.train() 

    trainer.save_model("moe_ed_scratch_saved")
    torch.save(model.state_dict(), "moe_ed_scratch_saved/pytorch_model.bin")

    metric = evaluate.load("rouge") 
    predictions = []
    references = []
    context_length = 128
    test_dataloader = torch.utils.data.DataLoader(test_dataset, collate_fn=data_collator, batch_size=16, drop_last = True)
    texts = []
    labels_list = []
    for i, batch in enumerate(tqdm(test_dataloader)):
        outputs = model.generate(
            input_ids=batch['input_ids'].to('cuda'),
            max_new_tokens=context_length,
        )
        texts.append(outputs)
        labels_list.append(batch['labels'])

    with open("generated_test_texts.txt", "w", encoding="utf-8") as f:
        for outputs in texts:
            decoded = [tokenizer.decode(out, clean_up_tokenization_spaces=False, skip_special_tokens=True) for out in outputs]
            for line in decoded:
                f.write(line.strip() + "\n")
    print("Saved to generated_test_texts.txt")
    for outputs, labels in zip(texts, labels_list):
        with tokenizer.as_target_tokenizer():
            outputs = [tokenizer.decode(out, clean_up_tokenization_spaces=False, skip_special_tokens=True) for out in outputs]

            labels = np.where(labels != -100,  labels, tokenizer.pad_token_id)
            actuals = [tokenizer.decode(out, clean_up_tokenization_spaces=False, skip_special_tokens=True) for out in labels]
        predictions.extend(outputs)
        references.extend(actuals)

    import json
    results = dict(metric.compute(predictions=predictions, references=references).items() )
    with open("rouge_test_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=True, indent=4)
    print("Saved to rouge_test_results.json")

    print(results)