
from datasets import load_metric
from safetensors.torch import load_file


from tokenized_data.tokenizer import tokenizer
from transformer_trainer import model, info_model, data_collator
from tokenized_data.load_data import test_dataset
from translation_metrics import test_translation 

# model.load_state_dict(load_file( info_model["folder"] + "/model.safetensors"))

context_length = 1024
batch_test = 4
file_name = "td_mt"

test_translation(file_name, test_dataset, 
                   data_collator, batch_test, 
                   context_length, model, 
                   tokenizer
                   )

