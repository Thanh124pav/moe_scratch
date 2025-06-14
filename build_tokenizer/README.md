# Vietnamese Custom SentencePiece Tokenizer

Bộ công cụ để xây dựng tokenizer SentencePiece tùy chỉnh cho tiếng Việt từ raw data của bạn.

## 🎯 Tính năng

- **Thu thập raw text** từ các dataset Wiki, VNews, IWSLT15
- **Train SentencePiece tokenizer** với nhiều kích thước vocab (16K, 32K, 50K)
- **Wrapper class** tương thích với Transformers library
- **Xử lý data** tự động với custom tokenizer
- **Pipeline hoàn chỉnh** từ raw data đến tokenized data

## 📋 Yêu cầu

```bash
pip install -r requirements.txt
```

## 🚀 Cách sử dụng

### 1. Chuẩn bị dữ liệu

Đảm bảo bạn có dữ liệu trong cấu trúc sau:

```
data/raw/
├── Wiki/
│   ├── train.tsv
│   ├── valid.tsv
│   └── test.tsv
├── VNews/
│   ├── train.tsv
│   ├── valid.tsv
│   └── test.tsv
└── iwslt15/
    ├── train.en
    ├── train.vi
    ├── tst2012.en
    ├── tst2012.vi
    ├── tst2013.en
    └── tst2013.vi
```

### 2. Chạy pipeline hoàn chỉnh

```bash
python build_tokenizer/build_complete_pipeline.py
```

Hoặc chạy từng bước:

#### Bước 1: Thu thập raw text
```bash
python build_tokenizer/collect_raw_text.py
```

#### Bước 2: Train tokenizer
```bash
python build_tokenizer/train_sentencepiece.py
```

#### Bước 3: Test tokenizer
```bash
python build_tokenizer/custom_tokenizer.py
```

#### Bước 4: Process data
```bash
python build_tokenizer/process_data_with_custom_tokenizer.py
```

## 📊 Kết quả

Sau khi chạy pipeline, bạn sẽ có:

```
build_tokenizer/
├── raw_text_for_tokenizer.txt      # Raw text để train tokenizer
├── vietnamese_tokenizer_16000.model # Tokenizer 16K vocab
├── vietnamese_tokenizer_32000.model # Tokenizer 32K vocab
├── vietnamese_tokenizer_50000.model # Tokenizer 50K vocab
└── saved_tokenizer/                 # Tokenizer đã save

data/custom_tokenized/
├── tokenizer/                       # Tokenizer để dùng trong training
├── summarization_train/             # Summarization data
├── summarization_valid/
├── summarization_test/
├── translation_train/               # Translation data
├── translation_valid/
├── translation_test/
└── processing_info.json             # Metadata
```

## 💻 Sử dụng trong code

### Load tokenizer

```python
from build_tokenizer.custom_tokenizer import CustomSentencePieceTokenizer

# Load tokenizer
tokenizer = CustomSentencePieceTokenizer.from_pretrained('data/custom_tokenized/tokenizer')

# Hoặc load từ model file trực tiếp
tokenizer = CustomSentencePieceTokenizer('build_tokenizer/vietnamese_tokenizer_32000.model')
```

### Tokenize text

```python
# Single text
text = "Xin chào, tôi là AI assistant."
result = tokenizer(text, max_length=50, padding=True, return_tensors="pt")
print(result['input_ids'])
print(result['attention_mask'])

# Batch texts
texts = ["Câu 1", "Câu 2", "Câu 3"]
batch_result = tokenizer(texts, max_length=50, padding=True, return_tensors="pt")
print(batch_result['input_ids'].shape)  # [3, 50]
```

### Decode tokens

```python
# Decode single sequence
token_ids = [2, 1234, 5678, 3]  # [BOS, tokens..., EOS]
decoded = tokenizer.decode(token_ids)
print(decoded)

# Batch decode
sequences = [[2, 1234, 3], [2, 5678, 9012, 3]]
decoded_texts = tokenizer.batch_decode(sequences)
print(decoded_texts)
```

### Load processed data

```python
from datasets import load_from_disk

# Load summarization data
train_dataset = load_from_disk('data/custom_tokenized/summarization_train')
valid_dataset = load_from_disk('data/custom_tokenized/summarization_valid')

# Load translation data
mt_train = load_from_disk('data/custom_tokenized/translation_train')

print(f"Train size: {len(train_dataset)}")
print(f"Sample: {train_dataset[0]}")
```

### Tích hợp với Transformers

```python
from transformers import DataCollatorForSeq2Seq

# Tạo data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    return_tensors="pt",
    padding=True
)

# Sử dụng trong DataLoader
from torch.utils.data import DataLoader

dataloader = DataLoader(
    train_dataset,
    batch_size=4,
    collate_fn=data_collator
)

for batch in dataloader:
    print(batch.keys())  # ['input_ids', 'attention_mask', 'labels']
    break
```

## ⚙️ Tùy chỉnh

### Thay đổi vocab size

Sửa file `train_sentencepiece.py`:

```python
vocab_sizes = [8000, 16000, 24000, 32000]  # Thêm kích thước khác
```

### Thay đổi model type

```python
model_type='unigram'  # Thay vì 'bpe'
```

### Thay đổi character coverage

```python
character_coverage=0.999  # Tăng để cover nhiều ký tự hơn
```

### Thêm special tokens

```python
user_defined_symbols="<mask>,<sep>,<cls>"  # Thêm tokens tùy chỉnh
```

## 🔧 Troubleshooting

### Lỗi "Model file not found"
```bash
# Kiểm tra file có tồn tại không
ls -la build_tokenizer/vietnamese_tokenizer_*.model
```

### Lỗi memory khi train tokenizer
```python
# Giảm max_sentence_length
max_sentence_length=2048  # Thay vì 4192
```

### Lỗi encoding
```python
# Đảm bảo file được mở với encoding UTF-8
with open(file_path, 'r', encoding='utf-8') as f:
```

## 📈 So sánh vocab sizes

| Vocab Size | Model Size | Speed | Coverage | Use Case |
|------------|------------|--------|----------|----------|
| 16K        | Nhỏ        | Nhanh  | Thấp     | Testing, small models |
| 32K        | Vừa        | Vừa    | Tốt      | **Recommended** |
| 50K        | Lớn        | Chậm   | Cao      | Large models, best quality |

## 🎯 Best Practices

1. **Chọn vocab size 32K** cho hầu hết các ứng dụng
2. **Sử dụng BPE** cho tiếng Việt
3. **Character coverage 0.9995** cho tiếng Việt
4. **Test tokenizer** với nhiều loại text khác nhau
5. **Save tokenizer config** để reproduce được kết quả

## 📞 Hỗ trợ

Nếu gặp vấn đề, hãy kiểm tra:

1. Cấu trúc thư mục data đúng chưa
2. Dependencies đã install đủ chưa
3. File permissions có OK không
4. Disk space đủ không (cần ~2GB cho pipeline hoàn chỉnh) 