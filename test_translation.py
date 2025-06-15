from transformers import AutoTokenizer
from datasets import load_metric
from safetensors.torch import load_file


from train import model, config, data_collator
from tokenized_data.load_data import load_data
from translation_metrics import test_translation 

model.load_state_dict(load_file( config.folder + "/model.safetensors"))
tokenizer = AutoTokenizer.from_pretrained("tokenized_data/do_tokenizer")
_, _, test_dataset = load_data(config.path_data)
print(len(test_dataset))
context_length = 1024
batch_test = 2
file_name = "td_mt"

test_translation(file_name, test_dataset, 
                   data_collator, batch_test, 
                   context_length, model, 
                   tokenizer
                   )

