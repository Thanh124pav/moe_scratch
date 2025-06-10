from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("VietAI/vit5-base")
vocab_size = tokenizer.vocab_size 