import os
import re
from datasets import Dataset
from custom_tokenizer import CustomSentencePieceTokenizer
import json
from tqdm import tqdm

def normalize_text(text):
    """Chuẩn hóa text"""
    # Loại bỏ ký tự tab, xuống dòng
    text = re.sub(r'[\t\n]', ' ', text)
    # Loại bỏ các cặp {}, [], ()
    text = re.sub(r'[\{\}\[\]\(\)]', ' ', text)
    # Loại bỏ teen code/emoticon
    text = re.sub(r'(:\)|:\(|:D|:P|:p|:v|:V|:3|:o|:O|:x|:X|:\||:>|:<|:\)+|:\(+)', ' ', text)
    # Loại bỏ 2 dấu câu đứng cạnh nhau
    text = re.sub(r'([.,!?;:\"\']){2,}', r'\1', text)
    # Loại bỏ các dấu câu mà giữa chúng là khoảng trắng
    text = re.sub(r'([.,!?;:\"\'])\s+([.,!?;:\"\'])', r'\2', text)
    # Loại bỏ ký tự không phải chữ cái tiếng Việt, số, dấu câu cơ bản
    text = re.sub(r"[^A-Za-zÀ-ỹà-ỹ0-9.,!?;:'\" \-]", '', text)
    # Chuẩn hóa khoảng trắng
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_function_custom(examples, tokenizer, max_length=1024):
    """Preprocess function sử dụng custom tokenizer"""
    
    # Normalize text
    inputs = [normalize_text(inp) for inp in examples["inputs"]]
    labels = [normalize_text(lbl) for lbl in examples["labels"]]
    
    # Tokenize inputs
    model_inputs = tokenizer(
        inputs, 
        max_length=max_length, 
        truncation=True, 
        padding="max_length",
        return_tensors=None  # Return lists, not tensors
    )
    
    # Tokenize labels
    label_encodings = tokenizer(
        labels, 
        max_length=max_length, 
        truncation=True, 
        padding="max_length",
        return_tensors=None
    )
    
    # Đổi tất cả pad_token_id trong labels thành -100
    labels_ids = []
    for seq in label_encodings["input_ids"]:
        label_seq = [token if token != tokenizer.pad_token_id else -100 for token in seq]
        labels_ids.append(label_seq)
    
    model_inputs['labels'] = labels_ids
    
    return model_inputs

def create_tokenized_dataset_custom(file_name, tokenizer, base_dir="data/raw", Wiki=True, Vnews=True, max_length=1024):
    """Tạo tokenized dataset sử dụng custom tokenizer"""
    
    input_lines = []
    label_lines = []
    
    # Wiki data
    if Wiki:
        wiki_file = os.path.join(base_dir, "Wiki", file_name)
        if os.path.exists(wiki_file):
            print(f"Processing Wiki {file_name}...")
            with open(wiki_file, encoding='utf-8') as file:
                for i, line in enumerate(file):
                    if i == 0:  # Skip header
                        continue
                    try:
                        parts = line.strip().split('\t')
                        if len(parts) >= 3:
                            input_lines.append(parts[1])  # Input text
                            label_lines.append(parts[2])  # Summary
                    except Exception as e:
                        print(f"Error processing line {i}: {e}")
                        continue
        else:
            print(f"Wiki file not found: {wiki_file}")
    
    # VNews data
    if Vnews:
        vnews_file = os.path.join(base_dir, "VNews", file_name)
        if os.path.exists(vnews_file):
            print(f"Processing VNews {file_name}...")
            with open(vnews_file, encoding='utf-8') as file:
                for i, line in enumerate(file):
                    if i == 0:  # Skip header
                        continue
                    try:
                        parts = line.strip().split('\t')
                        if len(parts) >= 4:
                            input_lines.append(parts[3])  # Input text
                            label_lines.append(parts[2])  # Summary
                    except Exception as e:
                        print(f"Error processing line {i}: {e}")
                        continue
        else:
            print(f"VNews file not found: {vnews_file}")
    
    print(f"Total samples collected: {len(input_lines)}")
    
    # Tạo dataset
    dict_obj = {'inputs': input_lines, 'labels': label_lines}
    dataset = Dataset.from_dict(dict_obj)
    
    # Tokenize dataset
    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        lambda examples: preprocess_function_custom(examples, tokenizer, max_length),
        batched=True,
        remove_columns=['inputs', 'labels'],
        desc="Tokenizing"
    )
    
    return dataset, tokenized_dataset

def create_tokenized_MT_dataset_custom(en_file, vi_file, tokenizer, base_dir="data/raw/iwslt15", max_length=1024):
    """Tạo tokenized machine translation dataset"""
    
    en_path = os.path.join(base_dir, en_file)
    vi_path = os.path.join(base_dir, vi_file)
    
    if not os.path.exists(en_path) or not os.path.exists(vi_path):
        print(f"MT files not found: {en_path} or {vi_path}")
        return None, None
    
    print(f"Processing MT dataset: {en_file} -> {vi_file}")
    
    # Read files
    with open(en_path, encoding='utf-8') as f:
        en_lines = [line.strip() for line in f if line.strip()]
    
    with open(vi_path, encoding='utf-8') as f:
        vi_lines = [line.strip() for line in f if line.strip()]
    
    print(f"EN lines: {len(en_lines)}, VI lines: {len(vi_lines)}")
    
    # Ensure same length
    min_len = min(len(en_lines), len(vi_lines))
    en_lines = en_lines[:min_len]
    vi_lines = vi_lines[:min_len]
    
    # Create dataset
    dict_obj = {'inputs': en_lines, 'labels': vi_lines}
    dataset = Dataset.from_dict(dict_obj)
    
    # Tokenize
    print("Tokenizing MT dataset...")
    tokenized_dataset = dataset.map(
        lambda examples: preprocess_function_custom(examples, tokenizer, max_length),
        batched=True,
        remove_columns=['inputs', 'labels'],
        desc="Tokenizing MT"
    )
    
    return dataset, tokenized_dataset

def main():
    """Main function để process tất cả data"""
    
    # Load custom tokenizer
    tokenizer_path = "build_tokenizer/vietnamese_tokenizer_32000.model"
    
    if not os.path.exists(tokenizer_path):
        print(f"❌ Tokenizer not found: {tokenizer_path}")
        print("Please train the tokenizer first!")
        return
    
    print("Loading custom tokenizer...")
    tokenizer = CustomSentencePieceTokenizer(tokenizer_path)
    
    # Tạo thư mục output
    output_dir = "data/custom_tokenized"
    os.makedirs(output_dir, exist_ok=True)
    
    max_length = 1024
    
    # === Summarization Task ===
    print("\n" + "="*50)
    print("PROCESSING SUMMARIZATION DATA")
    print("="*50)
    
    for split in ['train.tsv', 'valid.tsv', 'test.tsv']:
        print(f"\nProcessing {split}...")
        
        raw_dataset, tokenized_dataset = create_tokenized_dataset_custom(
            split, tokenizer, max_length=max_length
        )
        
        if tokenized_dataset is not None:
            # Save datasets
            split_name = split.replace('.tsv', '')
            
            # Save tokenized dataset
            save_path = os.path.join(output_dir, f"summarization_{split_name}")
            tokenized_dataset.save_to_disk(save_path)
            print(f"✓ Saved tokenized dataset: {save_path}")
            
            # Save raw dataset for reference
            raw_save_path = os.path.join(output_dir, f"raw_summarization_{split_name}")
            raw_dataset.save_to_disk(raw_save_path)
            print(f"✓ Saved raw dataset: {raw_save_path}")
            
            print(f"Dataset size: {len(tokenized_dataset)}")
    
    # === Machine Translation Task ===
    print("\n" + "="*50)
    print("PROCESSING MACHINE TRANSLATION DATA")
    print("="*50)
    
    mt_files = [
        ('train.en', 'train.vi', 'train'),
        ('tst2012.en', 'tst2012.vi', 'test'),
        ('tst2013.en', 'tst2013.vi', 'valid')
    ]
    
    for en_file, vi_file, split_name in mt_files:
        print(f"\nProcessing MT {split_name}: {en_file} -> {vi_file}")
        
        raw_dataset, tokenized_dataset = create_tokenized_MT_dataset_custom(
            en_file, vi_file, tokenizer, max_length=max_length
        )
        
        if tokenized_dataset is not None:
            # Save datasets
            save_path = os.path.join(output_dir, f"translation_{split_name}")
            tokenized_dataset.save_to_disk(save_path)
            print(f"✓ Saved tokenized MT dataset: {save_path}")
            
            # Save raw dataset
            raw_save_path = os.path.join(output_dir, f"raw_translation_{split_name}")
            raw_dataset.save_to_disk(raw_save_path)
            print(f"✓ Saved raw MT dataset: {raw_save_path}")
            
            print(f"Dataset size: {len(tokenized_dataset)}")
    
    # === Save tokenizer info ===
    print("\n" + "="*50)
    print("SAVING TOKENIZER INFO")
    print("="*50)
    
    # Save tokenizer to output directory
    tokenizer.save_pretrained(os.path.join(output_dir, "tokenizer"))
    
    # Save processing info
    info = {
        "tokenizer_type": "SentencePiece BPE",
        "vocab_size": tokenizer.vocab_size,
        "max_length": max_length,
        "special_tokens": {
            "pad_token": tokenizer.pad_token,
            "unk_token": tokenizer.unk_token,
            "bos_token": tokenizer.bos_token,
            "eos_token": tokenizer.eos_token,
        },
        "special_token_ids": {
            "pad_token_id": tokenizer.pad_token_id,
            "unk_token_id": tokenizer.unk_token_id,
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
    }
    
    with open(os.path.join(output_dir, "processing_info.json"), 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Processing completed!")
    print(f"✓ All data saved to: {output_dir}")
    print(f"✓ Tokenizer saved to: {os.path.join(output_dir, 'tokenizer')}")

if __name__ == "__main__":
    main() 