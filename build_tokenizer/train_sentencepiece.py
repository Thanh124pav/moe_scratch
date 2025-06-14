import sentencepiece as spm
import os
from pathlib import Path

def train_sentencepiece_tokenizer(
    input_file,
    model_prefix,
    vocab_size=32000,
    character_coverage=0.9995,
    model_type='bpe'
):
    """
    Train SentencePiece tokenizer
    
    Args:
        input_file: File chứa raw text
        model_prefix: Tên prefix cho model (sẽ tạo ra .model và .vocab)
        vocab_size: Kích thước vocabulary
        character_coverage: Tỷ lệ ký tự được cover (cao hơn cho tiếng Việt)
        model_type: 'bpe', 'unigram', 'char', 'word'
    """
    
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file {input_file} not found!")
    
    # Tạo thư mục output
    output_dir = os.path.dirname(model_prefix)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    print(f"Training SentencePiece tokenizer...")
    print(f"Input file: {input_file}")
    print(f"Model prefix: {model_prefix}")
    print(f"Vocab size: {vocab_size}")
    print(f"Model type: {model_type}")
    print(f"Character coverage: {character_coverage}")
    
    # Các tham số training
    train_args = [
        f"--input={input_file}",
        f"--model_prefix={model_prefix}",
        f"--vocab_size={vocab_size}",
        f"--character_coverage={character_coverage}",
        f"--model_type={model_type}",
        "--pad_id=0",
        "--unk_id=1", 
        "--bos_id=2",
        "--eos_id=3",
        "--user_defined_symbols=<pad>,<s>,</s>",  # Thêm special tokens
        "--max_sentence_length=4192",  # Tăng độ dài max cho long sequences
        "--shuffle_input_sentence=true",  # Shuffle để training tốt hơn
        "--train_extremely_large_corpus=false",
        "--num_threads=16",  # Sử dụng nhiều threads
    ]
    
    try:
        spm.SentencePieceTrainer.train(" ".join(train_args))
        print(f"✓ Training completed!")
        print(f"✓ Model saved: {model_prefix}.model")
        print(f"✓ Vocab saved: {model_prefix}.vocab")
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False

def test_tokenizer(model_path, test_sentences):
    """Test tokenizer với một vài câu mẫu"""
    
    try:
        sp = spm.SentencePieceProcessor()
        sp.load(model_path)
        
        print(f"\n=== Testing Tokenizer ===")
        print(f"Vocab size: {sp.vocab_size()}")
        print(f"BOS ID: {sp.bos_id()}")
        print(f"EOS ID: {sp.eos_id()}")
        print(f"PAD ID: {sp.pad_id()}")
        print(f"UNK ID: {sp.unk_id()}")
        
        for i, sentence in enumerate(test_sentences):
            print(f"\n--- Test {i+1} ---")
            print(f"Original: {sentence}")
            
            # Encode to pieces
            pieces = sp.encode_as_pieces(sentence)
            print(f"Pieces: {pieces}")
            
            # Encode to IDs
            ids = sp.encode_as_ids(sentence)
            print(f"IDs: {ids}")
            
            # Decode back
            decoded = sp.decode_ids(ids)
            print(f"Decoded: {decoded}")
            
            # Check if original == decoded
            print(f"Match: {sentence.strip() == decoded.strip()}")
            
    except Exception as e:
        print(f"❌ Testing failed: {e}")

def main():
    # Paths
    raw_text_file = "build_tokenizer/raw_text_for_tokenizer.txt"
    model_prefix = "build_tokenizer/vietnamese_tokenizer"
    
    # Kiểm tra input file
    if not os.path.exists(raw_text_file):
        print(f"❌ Raw text file not found: {raw_text_file}")
        print("Please run collect_raw_text.py first!")
        return
    
    # Train tokenizer với các kích thước vocab khác nhau
    vocab_sizes = [16000, 32000, 50000]
    
    for vocab_size in vocab_sizes:
        print(f"\n{'='*50}")
        print(f"Training tokenizer with vocab_size={vocab_size}")
        print(f"{'='*50}")
        
        model_prefix_sized = f"{model_prefix}_{vocab_size}"
        
        success = train_sentencepiece_tokenizer(
            input_file=raw_text_file,
            model_prefix=model_prefix_sized,
            vocab_size=vocab_size,
            character_coverage=0.9995,  # Cao cho tiếng Việt
            model_type='bpe'  # Byte Pair Encoding
        )
        
        if success:
            # Test tokenizer
            test_sentences = [
                "Xin chào, tôi là một mô hình ngôn ngữ AI.",
                "Việt Nam là một đất nước xinh đẹp ở Đông Nam Á.",
                "Machine learning và deep learning đang phát triển rất nhanh.",
                "COVID-19 đã ảnh hưởng lớn đến kinh tế thế giới.",
                "Hello, how are you today? I'm fine, thank you!"
            ]
            
            test_tokenizer(f"{model_prefix_sized}.model", test_sentences)
    
    print(f"\n✓ All tokenizers trained successfully!")
    print(f"✓ Choose the best vocab_size based on your needs:")
    print(f"  - 16K: Faster, smaller model, might have more UNK tokens")
    print(f"  - 32K: Balanced choice for most applications") 
    print(f"  - 50K: Better coverage, larger model")

if __name__ == "__main__":
    main() 