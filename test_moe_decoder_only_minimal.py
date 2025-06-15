import torch
from model.moe.decoder_only import MoEDecoderModel

# Dummy config, sửa lại cho đúng với model của bạn nếu cần
class DummyConfig:
    vocab_size = 32100
    embed_dim = 512
    block_size = 128
    n_heads = 8
    n_experts = 4
    top_k_experts = 2
    n_layers = 6
    eos_token_id = 1

if __name__ == '__main__':
    config = DummyConfig()
    model = MoEDecoderModel(config).cuda()
    B, T = 2, 16  # batch size, seq len
    input_ids = torch.randint(0, config.vocab_size, (B, T)).cuda()
    labels = input_ids.clone()
    labels[:, :8] = -100  # mask nửa đầu
    attention_mask = torch.ones_like(input_ids).cuda()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        print('Loss:', outputs.loss)
        print('Logits NaN:', torch.isnan(outputs.logits).any())
        print('Output shape:', outputs.logits.shape) 