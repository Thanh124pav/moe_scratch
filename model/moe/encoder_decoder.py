import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutput
from transformers import PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import Seq2SeqLMOutput


from .encoder import Encoder
from .decoder import Decoder


# === Full Encoder-Decoder Model ===
lb_weight = 0.01
class EncoderDecoderMoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        vocab_size = config.vocab_size
        embed_dim = config.embed_dim
        self.decoder_start_token_id = config.decoder_start_token_id
        self.eos_token_id = config.eos_token_id
        self.pad_token_id = config.pad_token_id
        self.shared_embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoder = Encoder(config, self.shared_embedding)  
        self.decoder = Decoder(config, self.shared_embedding)
        
        self.apply(kaiming_init_weights)


    def forward(self, input_ids=None, decoder_input_ids=None, labels=None,
                past_key_values=None, use_cache=False, return_dict = True,
                encoder_outputs=None, src_mask=None, tgt_mask=None):

        if decoder_input_ids is None:
            decoder_start_token_id = self.decoder_start_token_id
            if labels is not None:
                decoder_input_ids = labels.new_zeros(labels.shape)
                decoder_input_ids[:, 1:] = labels[:, :-1]
                decoder_input_ids[:, 0] = self.decoder_start_token_id
                decoder_input_ids = torch.where(
                    decoder_input_ids == -100,
                    self.pad_token_id,
                    decoder_input_ids
                )
                # Tạo decoder_attention_mask bằng cách dịch trái attention_mask của labels
                decoder_attention_mask = (labels != -100) & (labels != self.pad_token_id)
                decoder_attention_mask = decoder_attention_mask.long()
                shifted_decoder_attention_mask = decoder_attention_mask.new_zeros(decoder_attention_mask.shape)
                shifted_decoder_attention_mask[:, 1:] = decoder_attention_mask[:, :-1]
                shifted_decoder_attention_mask[:, 0] = 1  # Vị trí đầu tiên luôn là real (start token)
                decoder_attention_mask = shifted_decoder_attention_mask
            else: 
                #print("generate")
                batch_size = input_ids.shape[0]
                decoder_input_ids = input_ids.new_full((batch_size, 1), self.decoder_start_token_id)
        lb_loss = 0   
        if encoder_outputs is not None:
            #print(encoder_outputs)
            if isinstance(encoder_outputs, tuple):
                # print("Tuple!")
                enc_out = encoder_outputs[0].last_hidden_state
                lb_loss += encoder_outputs[1]
            else:
                enc_out = encoder_outputs.last_hidden_state
        else:
            assert input_ids is not None, "input_ids must be provided if encoder_outputs is None"
            lb_loss = 0
            enc_out, lb = self.encoder(input_ids, src_mask)
            enc_out = enc_out.last_hidden_state
            lb_loss += lb
            encoder_outputs = BaseModelOutput(last_hidden_state=enc_out)

        
        y, lb, next_kvs = self.decoder(decoder_input_ids, enc_out, past_key_values, use_cache, attention_mask = tgt_mask)
        lb_loss += lb
        logits = torch.matmul(y, self.shared_embedding.weight.t())
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
            loss += lb_weight * lb_loss
        return Seq2SeqLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=tuple(next_kvs) if use_cache else None,
            encoder_last_hidden_state= (
                encoder_outputs[0].last_hidden_state if isinstance(encoder_outputs, tuple)
                else encoder_outputs.last_hidden_state
            )
        )
    def get_encoder(self):
        return self.encoder
    def get_decoder(self):
        return self.decoder
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
                    decoder_input_ids=input_ids if i == 0 else None  
                )
                logits = output.logits
                logits = logits[:, -1, :]
                past_key_values = output.past_key_values

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            generated[:, i] = idx_next.squeeze(-1)
            input_ids = torch.cat((input_ids, idx_next), dim=-1)
            finish = torch.logical_or(finish, (idx_next.squeeze(-1) == self.eos_token_id))
            if finish.all():
                break

        return generated

def kaiming_init_weights(m):
    if isinstance (m, (nn.Linear)):
        init.kaiming_normal_(m.weight)

