from datasets import load_metric 
import torch
from tqdm import tqdm
import numpy as np
import json

def test_summarization(file_name, test_dataset, 
                       data_collator, batch_test, 
                       context_length, model, 
                       tokenizer, metric):
    predictions = []
    references = []
    test_dataloader = torch.utils.data.DataLoader(test_dataset, collate_fn=data_collator, batch_size=batch_test, drop_last = True)
    texts = []
    labels_list = []
    model.eval()
    for i, batch in enumerate(tqdm(test_dataloader)):
        outputs = model.generate(
            input_ids=batch['input_ids'].to('cuda'),
            #context_length=context_length,
        )
        texts.append(outputs)
        labels_list.append(batch['labels'])

    with open( file_name +  ".txt", "w", encoding="utf-8") as f:
        for outputs in texts:
            decoded = [tokenizer.decode(out, clean_up_tokenization_spaces=False, skip_special_tokens=True) for out in outputs]
            for line in decoded:
                f.write(line.strip() + "\n")
    print("Saved text")
    for outputs, labels in zip(texts, labels_list):
        with tokenizer.as_target_tokenizer():
            outputs = [tokenizer.decode(out, clean_up_tokenization_spaces=False, skip_special_tokens=True) for out in outputs]

            labels = np.where(labels != -100,  labels, tokenizer.pad_token_id)
            actuals = [tokenizer.decode(out, clean_up_tokenization_spaces=False, skip_special_tokens=True) for out in labels]
        predictions.extend(outputs)
        references.extend(actuals)

    results = dict(metric.compute(predictions=predictions, references=references).items() )

    with open(file_name + ".json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print("Saved json")

    print(results)
    return results