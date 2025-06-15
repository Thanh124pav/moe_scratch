import torch
from transformers import AutoTokenizer

class BuildSummarizationDataset:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def build_encoder_decoder_dataset(self, pairs, max_length=512):
        dataset = []
        for document, summary in pairs:
            doc_enc = self.tokenizer(document, max_length=max_length, truncation=True, padding='max_length', return_tensors='pt')
            sum_enc = self.tokenizer(summary, max_length=max_length, truncation=True, padding='max_length', return_tensors='pt')
            sample = {
                'input_ids': doc_enc['input_ids'].squeeze(0),
                'attention_mask': doc_enc['attention_mask'].squeeze(0),
                'labels': sum_enc['input_ids'].squeeze(0)
            }
            dataset.append(sample)
        return dataset

    def build_decoder_only_dataset(self, pairs, max_length=512, sep_token='<sep>'):
        dataset = []
        eos_token = self.tokenizer.eos_token
        if self.tokenizer.sep_token is None:
            self.tokenizer.sep_token = sep_token
            self.tokenizer.add_special_tokens({'additional_special_tokens': [sep_token]})
        old_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = 'left'
        for document, summary in pairs:
            input_text = f"{document} {sep_token} {summary} {eos_token}"
            enc = self.tokenizer(input_text, max_length=max_length, truncation=True, padding='max_length', return_tensors='pt')
            input_ids = enc['input_ids'].squeeze(0)
            attention_mask = enc['attention_mask'].squeeze(0)
            sep_id = self.tokenizer.convert_tokens_to_ids(sep_token)
            sep_idx = (input_ids == sep_id).nonzero(as_tuple=True)[0].item() if sep_id in input_ids else 0
            labels = input_ids.clone()
            labels[:sep_idx+1] = -100
            sample = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels
            }
            dataset.append(sample)
        self.tokenizer.padding_side = old_padding_side
        return dataset

    def test_inference(self):
        pairs = [
            ("Hà Nội là thủ đô của Việt Nam.", "Thủ đô Việt Nam là Hà Nội."),
            ("Trí tuệ nhân tạo đang phát triển rất nhanh.", "AI phát triển nhanh.")
        ]
        print('--- Encoder-Decoder Summarization Dataset ---')
        ed_dataset = self.build_encoder_decoder_dataset(pairs, max_length=16)
        for i, sample in enumerate(ed_dataset):
            print(f"Sample {i}:")
            print('input_ids:', sample['input_ids'])
            print('attention_mask:', sample['attention_mask'])
            print('labels:', sample['labels'])
            print('input text:', self.tokenizer.decode(sample['input_ids'], skip_special_tokens=True))
            print('label text:', self.tokenizer.decode(sample['labels'], skip_special_tokens=True))
            print()
        print('--- Decoder-Only Summarization Dataset ---')
        do_dataset = self.build_decoder_only_dataset(pairs, max_length=24)
        for i, sample in enumerate(do_dataset):
            print(f"Sample {i}:")
            print('input_ids:', sample['input_ids'])
            print('attention_mask:', sample['attention_mask'])
            print('labels:', sample['labels'])
            print('input text:', self.tokenizer.decode(sample['input_ids'], skip_special_tokens=True))
            label_ids = sample['labels'][sample['labels'] != -100]
            print('label text:', self.tokenizer.decode(label_ids, skip_special_tokens=True))
            print('input: ', end = " ")
            for i in range(len(sample['input_ids'])):
                print(self.tokenizer.decode(sample['input_ids'][i], skip_special_tokens=False), end = " ")
            print('output: ', end = " ")
            for i in range(len(sample['labels'])):
                if sample['labels'][i] != -100:
                    print(self.tokenizer.decode(sample['labels'][i], skip_special_tokens=False), end = " ")
                else:
                    print('<mask>', end = " ")
            print()

if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained('VietAI/vit5-base')
    builder = BuildSummarizationDataset(tokenizer)
    builder.test_inference()
