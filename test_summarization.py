from datasets import load_metric
from safetensors.torch import load_file


from tokenized_data.tokenizer import tokenizer
from trainer import model, config, data_collator
from tokenized_data.load_data import test_dataset

from summarization_metrics import test_summarization 

metric = load_metric("rouge") 
model.load_state_dict(load_file( config.folder + "/model.safetensors"))

context_length = 1024
batch_test = 4
file_name = "t_wiki_wiki1024"

test_summarization(file_name, test_dataset, 
                   data_collator, batch_test, 
                   context_length, model, 
                   tokenizer, metric
                   )