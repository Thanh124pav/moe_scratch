import os
import re
from pathlib import Path

def normalize_text(text):
    """Chuẩn hóa text trước khi train tokenizer"""
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

def collect_wiki_data(wiki_dir, output_file):
    """Thu thập data từ Wiki dataset"""
    files = ['train.tsv', 'valid.tsv', 'test.tsv']
    
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for file_name in files:
            file_path = os.path.join(wiki_dir, file_name)
            if os.path.exists(file_path):
                print(f"Processing Wiki {file_name}...")
                with open(file_path, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f):
                        if i == 0:  # Skip header
                            continue
                        try:
                            parts = line.strip().split('\t')
                            if len(parts) >= 3:
                                # parts[1] = input text, parts[2] = summary
                                input_text = normalize_text(parts[1])
                                summary_text = normalize_text(parts[2])
                                
                                if input_text.strip():
                                    out_f.write(input_text + '\n')
                                if summary_text.strip():
                                    out_f.write(summary_text + '\n')
                        except Exception as e:
                            print(f"Error processing line {i} in {file_name}: {e}")
                            continue

def collect_vnews_data(vnews_dir, output_file):
    """Thu thập data từ VNews dataset"""
    files = ['train.tsv', 'valid.tsv', 'test.tsv']
    
    with open(output_file, 'a', encoding='utf-8') as out_f:  # Append mode
        for file_name in files:
            file_path = os.path.join(vnews_dir, file_name)
            if os.path.exists(file_path):
                print(f"Processing VNews {file_name}...")
                with open(file_path, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f):
                        if i == 0:  # Skip header
                            continue
                        try:
                            parts = line.strip().split('\t')
                            if len(parts) >= 4:
                                # parts[3] = input text, parts[2] = summary
                                input_text = normalize_text(parts[3])
                                summary_text = normalize_text(parts[2])
                                
                                if input_text.strip():
                                    out_f.write(input_text + '\n')
                                if summary_text.strip():
                                    out_f.write(summary_text + '\n')
                        except Exception as e:
                            print(f"Error processing line {i} in {file_name}: {e}")
                            continue

def collect_iwslt_data(iwslt_dir, output_file):
    """Thu thập data từ IWSLT15 dataset (machine translation)"""
    file_pairs = [
        ('train.en', 'train.vi'),
        ('tst2012.en', 'tst2012.vi'),
        ('tst2013.en', 'tst2013.vi')
    ]
    
    with open(output_file, 'a', encoding='utf-8') as out_f:  # Append mode
        for en_file, vi_file in file_pairs:
            en_path = os.path.join(iwslt_dir, en_file)
            vi_path = os.path.join(iwslt_dir, vi_file)
            
            if os.path.exists(en_path) and os.path.exists(vi_path):
                print(f"Processing IWSLT {en_file} & {vi_file}...")
                
                # Read English lines
                with open(en_path, 'r', encoding='utf-8') as f:
                    en_lines = [normalize_text(line.strip()) for line in f if line.strip()]
                
                # Read Vietnamese lines
                with open(vi_path, 'r', encoding='utf-8') as f:
                    vi_lines = [normalize_text(line.strip()) for line in f if line.strip()]
                
                # Write both languages
                for en_line in en_lines:
                    if en_line.strip():
                        out_f.write(en_line + '\n')
                
                for vi_line in vi_lines:
                    if vi_line.strip():
                        out_f.write(vi_line + '\n')

def main():
    # Tạo thư mục output
    os.makedirs('build_tokenizer', exist_ok=True)
    
    # Đường dẫn đến raw data
    base_data_dir = "data/raw"
    wiki_dir = os.path.join(base_data_dir, "Wiki")
    vnews_dir = os.path.join(base_data_dir, "VNews") 
    iwslt_dir = os.path.join(base_data_dir, "iwslt15")
    
    # File output để train tokenizer
    raw_text_file = "build_tokenizer/raw_text_for_tokenizer.txt"
    
    # Xóa file cũ nếu có
    if os.path.exists(raw_text_file):
        os.remove(raw_text_file)
    
    print("Collecting raw text for tokenizer training...")
    
    # Thu thập data từ các nguồn
    if os.path.exists(wiki_dir):
        collect_wiki_data(wiki_dir, raw_text_file)
        print("✓ Wiki data collected")
    else:
        print("⚠ Wiki directory not found")
    
    if os.path.exists(vnews_dir):
        collect_vnews_data(vnews_dir, raw_text_file)
        print("✓ VNews data collected")
    else:
        print("⚠ VNews directory not found")
    
    if os.path.exists(iwslt_dir):
        collect_iwslt_data(iwslt_dir, raw_text_file)
        print("✓ IWSLT data collected")
    else:
        print("⚠ IWSLT directory not found")
    
    # Thống kê
    if os.path.exists(raw_text_file):
        with open(raw_text_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        print(f"✓ Total lines collected: {len(lines):,}")
        print(f"✓ Raw text saved to: {raw_text_file}")
        
        # Tính tổng số ký tự
        total_chars = sum(len(line) for line in lines)
        print(f"✓ Total characters: {total_chars:,}")
    else:
        print("❌ No data collected!")

if __name__ == "__main__":
    main() 