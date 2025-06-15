import streamlit as st
import torch
from transformers import AutoTokenizer
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

@st.cache_resource
def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained('VietAI/vit5-base')
    config = DummyConfig()
    model = MoEDecoderModel(config)
    # model.load_state_dict(torch.load('path_to_your_model.pt'))  # Nếu có checkpoint
    model.eval()
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

def generate_text(input_text, model, tokenizer, max_length=64, sep_token='<sep>'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Chuẩn hóa input cho decoder-only: [BOS] src <sep>
    bos_id = tokenizer.bos_token_id or tokenizer.cls_token_id
    sep_id = tokenizer.convert_tokens_to_ids(sep_token)
    input_ids = tokenizer.encode(input_text, add_special_tokens=False)
    input_ids = [bos_id] + input_ids + [sep_id]
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        output_ids = model.generate(input_tensor, max_length=max_length)[0].tolist()
    # Cắt phần input đi, chỉ lấy phần sinh ra
    gen_start = len(input_ids)
    gen_tokens = output_ids[gen_start:]
    # Nếu có eos thì cắt tại đó
    if tokenizer.eos_token_id in gen_tokens:
        eos_idx = gen_tokens.index(tokenizer.eos_token_id)
        gen_tokens = gen_tokens[:eos_idx]
    return tokenizer.decode(gen_tokens, skip_special_tokens=True)

st.title('MoE Decoder-only Machine Translation Demo')
user_input = st.text_input('Nhập văn bản nguồn:', '')
if st.button('Dịch/Generate'):
    if user_input.strip() == '':
        st.warning('Vui lòng nhập văn bản!')
    else:
        with st.spinner('Đang sinh kết quả...'):
            result = generate_text(user_input, model, tokenizer)
        st.success('Kết quả:')
        st.write(result) 