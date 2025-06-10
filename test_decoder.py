import torch
t = torch.rand(10, 10).cuda()
print(t.device) # should be CUDA
print("PyTorch:", torch.__version__)  
print("CUDA runtime trong wheel:", torch.version.cuda)  
print("Tên GPU:", torch.cuda.get_device_name(0))  
print("Compute Capability:", torch.cuda.get_device_capability(0))  
print("CUDA khả dụng:", torch.cuda.is_available())  

from datasets import load_metric
import torch
import numpy as np 
import json
from tqdm import tqdm
from torch.utils.data import DataLoader
from safetensors.torch import load_file

from decoder_trainer import tokenizer, data_collator, model, info_model
from tokenized_data.load_data import test_dataset


context_length = 1024
batch_test = 4
file_name = "d_6layers_wiki_wiki1024"


metric = load_metric("rouge") 
# model.load_state_dict(load_file( info_model["folder"] + "/model.safetensors"))
test_dataloader = DataLoader(test_dataset, collate_fn=data_collator, batch_size=batch_test, drop_last = True)

predictions = []
references = []
texts = []
labels_list = []
for i, batch in enumerate(tqdm(test_dataloader)):
  #model.clear_cache()
  outputs = model.generate(
      input_ids=batch['input_ids'].to('cuda'),
      context_length=context_length,
  )
  texts.append(outputs)
  labels_list.append(batch['labels'])

with open(file_name + ".txt", "w", encoding="utf-8") as f:
    for outputs in texts:
        decoded = [tokenizer.decode(out, clean_up_tokenization_spaces=False, skip_special_tokens=True) for out in outputs]
        for line in decoded:
            f.write(line.strip() + "\n")
print("Saved text")

for outputs, labels in zip(texts, labels_list):
  with tokenizer.as_target_tokenizer():
    outputs = [tokenizer.decode(out, clean_up_tokenization_spaces=False, skip_special_tokens=True) for out in outputs]

    labels = torch.where(labels != -100,  labels, tokenizer.pad_token_id)
    actuals = [tokenizer.decode(out, clean_up_tokenization_spaces=False, skip_special_tokens=True) for out in labels]
  predictions.extend(outputs)
  references.extend(actuals)


results = dict(metric.compute(predictions=predictions, references=references).items() )
with open( file_name + ".json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)
print("Saved json")

print(results)