from transformers import AutoTokenizer

ed_tokenizer = AutoTokenizer.from_pretrained("VietAI/vit5-base")
vocab_size = ed_tokenizer.vocab_size 
# sep_token = '<sep>'
# if ed_tokenizer.sep_token is None:
#     ed_tokenizer.sep_token = sep_token
#     ed_tokenizer.add_special_tokens({'additional_special_tokens': [sep_token]})
# ed_tokenizer.save_pretrained("tokenized_data/do_tokenizer") 