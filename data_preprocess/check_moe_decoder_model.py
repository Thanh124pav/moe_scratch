import torch
from model.moe.decoder_only import MoEDecoderModel
from transformers import AutoTokenizer

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

def check_moe_decoder_model(model, tokenizer, device='cuda'):
    model.eval()
    # Tạo batch input_ids, labels, attention_mask ngẫu nhiên
    B, T = 2, 16
    input_ids = torch.randint(0, tokenizer.vocab_size, (B, T)).to(device)
    labels = input_ids.clone()
    labels[:, :8] = -100  # mask nửa đầu
    attention_mask = torch.ones_like(input_ids).to(device)
    print('input_ids:', input_ids[0])
    print('labels:', labels[0])
    print('attention_mask:', attention_mask[0])
    print('Số lượng token tính loss:', (labels[0] != -100).sum().item())
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        print('Loss:', outputs.loss)
        print('Logits NaN:', torch.isnan(outputs.logits).any())
        pred_ids = outputs.logits.argmax(-1)
        print('Predicted token ids:', pred_ids[0])
        # Decode
        print('Decoded input:', tokenizer.decode(input_ids[0], skip_special_tokens=False))
        print('Decoded label:', tokenizer.decode([i for i in labels[0].tolist() if i != -100], skip_special_tokens=False))
        print('Decoded pred:', tokenizer.decode(pred_ids[0], skip_special_tokens=False))
    # Sinh thử một câu
    test_text = "Tôi yêu AI"
    bos_id = tokenizer.bos_token_id or tokenizer.cls_token_id
    sep_id = tokenizer.convert_tokens_to_ids('<sep>')
    test_input = [bos_id] + tokenizer.encode(test_text, add_special_tokens=False) + [sep_id]
    test_tensor = torch.tensor([test_input], dtype=torch.long, device=device)
    with torch.no_grad():
        gen_ids = model.generate(test_tensor, max_length=32)[0].tolist()
    gen_tokens = gen_ids[len(test_input):]
    if tokenizer.eos_token_id in gen_tokens:
        eos_idx = gen_tokens.index(tokenizer.eos_token_id)
        gen_tokens = gen_tokens[:eos_idx]
    print('Generated:', tokenizer.decode(gen_tokens, skip_special_tokens=True))

if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained('VietAI/vit5-base')
    config = DummyConfig()
    model = MoEDecoderModel(config).cuda()
    check_moe_decoder_model(model, tokenizer) 