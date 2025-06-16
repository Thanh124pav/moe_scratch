from datasets import load_metric
from safetensors.torch import load_file
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# tokenizer = AutoTokenizer.from_pretrained('tokenized_data/do_tokenizer')
# from tokenized_data.tokenizer import tokenizer
from train import model, config, data_collator
from tokenized_data.load_data import load_data 

from summarization_metrics import test_summarization 

metric = load_metric("rouge") 
#model.load_state_dict(load_file( config.folder + "/model.safetensors"))
tokenizer = AutoTokenizer.from_pretrained("VietAI/vit5-base")  
model = AutoModelForSeq2SeqLM.from_pretrained("VietAI/vit5-base").to('cuda')
_, _, test_dataset = load_data(config.path_data)
context_length = 1024
batch_test = 4
file_name = "t_wiki_wiki1024"
test_dataset = test_dataset.select(torch.arange(4))

test_summarization(file_name, test_dataset, 
                   data_collator, batch_test, 
                   context_length, model, 
                   tokenizer, metric
                   )