import torch
from transformers import AutoTokenizer

class BuildTranslationDataset:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def build_encoder_decoder_dataset(self, pairs, max_length=1024):
        dataset = []
        for src, tgt in pairs:
            src_enc = self.tokenizer(src, max_length=max_length, truncation=True, padding='max_length', return_tensors='pt')
            tgt_enc = self.tokenizer(tgt, max_length=max_length, truncation=True, padding='max_length', return_tensors='pt')
            labels = [idx if idx != self.tokenizer.pad_token_id else -100 for idx in tgt_enc['input_ids'].squeeze(0).tolist()]
            sample = {
                'input_ids': src_enc['input_ids'].squeeze(0),
                'attention_mask': src_enc['attention_mask'].squeeze(0),
                'labels': labels
            }
            dataset.append(sample)
        return dataset

    def build_decoder_only_dataset(self, pairs, max_length=1024, sep_token='<sep>', inference=False):
        dataset = []
        if self.tokenizer.sep_token is None:
            self.tokenizer.sep_token = sep_token
            self.tokenizer.add_special_tokens({'additional_special_tokens': [sep_token]})
        old_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = 'left'
        for src, tgt in pairs:
            if not inference:
                input_text = f"{src} {sep_token} {tgt}"
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
            else:
                input_text = f"{src} {sep_token}"
                enc = self.tokenizer(input_text, max_length=max_length, truncation=True, padding='max_length', return_tensors='pt')
                input_ids = enc['input_ids'].squeeze(0)[:-1]
                attention_mask = enc['attention_mask'].squeeze(0)[:-1]
                sample = {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask
                }
                labels = self.tokenizer(tgt, max_length=max_length, truncation=True, padding='max_length', return_tensors='pt')
                label_ids = labels['input_ids'].squeeze(0)
                # attention_mask = labels['attention_mask'].squeeze(0)
                labels = [idx if idx != self.tokenizer.pad_token_id else -100 for idx in label_ids]
                sample['labels'] = label_ids
                dataset.append(sample)
        self.tokenizer.padding_side = old_padding_side
        return dataset

    def test_inference(self, pairs):
        print('--- Encoder-Decoder Dataset ---')
        ed_dataset = self.build_encoder_decoder_dataset(pairs, max_length=10)
        for i, sample in enumerate(ed_dataset):
            print(f"Sample {i}:")
            print('input_ids:', sample['input_ids'])
            print('attention_mask:', sample['attention_mask'])
            print('labels:', sample['labels'])
            print('input text:', self.tokenizer.decode(sample['input_ids'], skip_special_tokens=False))
            label_ids = sample['labels'][sample['labels'] != -100]
            print('label text:', self.tokenizer.decode(label_ids, skip_special_tokens=False))
            print()
        print('--- Decoder-Only Dataset ---')
        do_dataset = self.build_decoder_only_dataset(pairs, max_length=16)
        for i, sample in enumerate(do_dataset):
            print(f"Sample {i}:")
            print('input_ids:', sample['input_ids'])
            print('attention_mask:', sample['attention_mask'])
            print('labels:', sample['labels'])
            print('input text:', self.tokenizer.decode(sample['input_ids'], skip_special_tokens=False))
            label_ids = sample['labels'][sample['labels'] != -100]
            print('label text:', self.tokenizer.decode(label_ids, skip_special_tokens=False))
        print('--- Decoder-Only Inference Input ---')
        do_dataset = self.build_decoder_only_dataset(pairs, max_length=16, inference=True)
        for i, sample in enumerate(do_dataset):
            print(f"Sample {i}:")
            print('input_ids:', sample['input_ids'])
            print('attention_mask:', sample['attention_mask'])
            print('labels:', sample['labels'])
            print('input text:', self.tokenizer.decode(sample['input_ids'], skip_special_tokens=False))
            label_ids = sample['labels'][sample['labels'] != -100]
            print('label text:', self.tokenizer.decode(label_ids, skip_special_tokens=False))

if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained('VietAI/vit5-base')
    builder = BuildTranslationDataset(tokenizer)
    pairs = [
        ("How are you?", "Bạn khỏe không?"),
        ("I love machine learning.", "Tôi thích học máy.")
    ]
    builder.test_inference(pairs)
