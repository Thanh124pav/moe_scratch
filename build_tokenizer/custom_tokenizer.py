import sentencepiece as spm
import os
import json
from typing import List, Dict, Optional, Union
import torch


class CustomSentencePieceTokenizer:
    """
    Custom SentencePiece tokenizer wrapper tương thích với Transformers
    """
    
    def __init__(self, model_path: str):
        """
        Args:
            model_path: Đường dẫn đến file .model của SentencePiece
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)
        
        # Special tokens
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.bos_token = "<s>"
        self.eos_token = "</s>"
        
        # Token IDs
        self.pad_token_id = self.sp.pad_id()
        self.unk_token_id = self.sp.unk_id()
        self.bos_token_id = self.sp.bos_id()
        self.eos_token_id = self.sp.eos_id()
        
        # Vocab size
        self.vocab_size = self.sp.vocab_size()
        
        # Model name cho compatibility
        self.name_or_path = model_path
        
        print(f"✓ Loaded SentencePiece tokenizer:")
        print(f"  - Vocab size: {self.vocab_size}")
        print(f"  - PAD: {self.pad_token_id}")
        print(f"  - UNK: {self.unk_token_id}")
        print(f"  - BOS: {self.bos_token_id}")
        print(f"  - EOS: {self.eos_token_id}")
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text thành list of token IDs"""
        if isinstance(text, list):
            # Nếu input là list, encode từng element
            return [self.encode(t, add_special_tokens) for t in text]
        
        # Encode text
        token_ids = self.sp.encode_as_ids(text)
        
        if add_special_tokens:
            token_ids = [self.bos_token_id] + token_ids + [self.eos_token_id]
            
        return token_ids
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode list of token IDs thành text"""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
            
        if skip_special_tokens:
            # Loại bỏ special tokens
            token_ids = [
                tid for tid in token_ids 
                if tid not in [self.pad_token_id, self.bos_token_id, self.eos_token_id]
            ]
        
        return self.sp.decode_ids(token_ids)
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text thành list of pieces"""
        return self.sp.encode_as_pieces(text)
    
    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """Convert list of tokens thành list of IDs"""
        return [self.sp.piece_to_id(token) for token in tokens]
    
    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        """Convert list of IDs thành list of tokens"""
        return [self.sp.id_to_piece(id) for id in ids]
    
    def __call__(self, 
                 text: Union[str, List[str]], 
                 max_length: Optional[int] = None,
                 padding: Union[bool, str] = False,
                 truncation: bool = False,
                 return_tensors: Optional[str] = None,
                 add_special_tokens: bool = True,
                 **kwargs) -> Dict:
        """
        Main tokenization method tương thích với Transformers
        """
        
        # Handle single text or batch
        if isinstance(text, str):
            texts = [text]
            single_input = True
        else:
            texts = text
            single_input = False
        
        # Encode all texts
        all_token_ids = []
        for t in texts:
            token_ids = self.encode(t, add_special_tokens=add_special_tokens)
            
            # Truncation
            if truncation and max_length and len(token_ids) > max_length:
                if add_special_tokens:
                    # Giữ BOS, truncate middle, giữ EOS
                    token_ids = token_ids[:max_length-1] + [self.eos_token_id]
                else:
                    token_ids = token_ids[:max_length]
            
            all_token_ids.append(token_ids)
        
        # Padding
        if padding and max_length:
            max_len = max_length
        elif padding:
            max_len = max(len(ids) for ids in all_token_ids)
        else:
            max_len = None
            
        if max_len:
            for i in range(len(all_token_ids)):
                current_len = len(all_token_ids[i])
                if current_len < max_len:
                    # Pad with pad_token_id
                    all_token_ids[i] = all_token_ids[i] + [self.pad_token_id] * (max_len - current_len)
        
        # Create attention masks
        attention_masks = []
        for token_ids in all_token_ids:
            mask = [1 if tid != self.pad_token_id else 0 for tid in token_ids]
            attention_masks.append(mask)
        
        # Prepare output
        result = {
            'input_ids': all_token_ids,
            'attention_mask': attention_masks
        }
        
        # Convert to tensors if requested
        if return_tensors == "pt":
            result['input_ids'] = torch.tensor(result['input_ids'], dtype=torch.long)
            result['attention_mask'] = torch.tensor(result['attention_mask'], dtype=torch.long)
        
        # Return single example if single input
        if single_input:
            for key in result:
                if isinstance(result[key], list):
                    result[key] = result[key][0]
                elif isinstance(result[key], torch.Tensor):
                    result[key] = result[key][0]
        
        return result
    
    def batch_decode(self, sequences: List[List[int]], skip_special_tokens: bool = True) -> List[str]:
        """Decode batch of sequences"""
        return [self.decode(seq, skip_special_tokens) for seq in sequences]
    
    def save_pretrained(self, save_directory: str):
        """Save tokenizer để có thể load lại"""
        os.makedirs(save_directory, exist_ok=True)
        
        # Copy model file
        import shutil
        model_dest = os.path.join(save_directory, "tokenizer.model")
        shutil.copy2(self.name_or_path, model_dest)
        
        # Save config
        config = {
            "model_type": "sentencepiece",
            "vocab_size": self.vocab_size,
            "pad_token": self.pad_token,
            "unk_token": self.unk_token,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
            "pad_token_id": self.pad_token_id,
            "unk_token_id": self.unk_token_id,
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
        }
        
        with open(os.path.join(save_directory, "tokenizer_config.json"), 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Tokenizer saved to {save_directory}")
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str):
        """Load tokenizer từ thư mục đã save"""
        if os.path.isdir(pretrained_model_name_or_path):
            model_path = os.path.join(pretrained_model_name_or_path, "tokenizer.model")
        else:
            model_path = pretrained_model_name_or_path
            
        return cls(model_path)


def test_custom_tokenizer():
    """Test custom tokenizer"""
    
    # Test với tokenizer đã train (nếu có)
    model_path = "build_tokenizer/vietnamese_tokenizer_32000.model"
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("Please train tokenizer first!")
        return
    
    print("=== Testing Custom Tokenizer ===")
    
    tokenizer = CustomSentencePieceTokenizer(model_path)
    
    # Test sentences
    test_sentences = [
        "Xin chào, tôi là AI assistant.",
        "Việt Nam là đất nước xinh đẹp.",
        "Machine learning rất thú vị!"
    ]
    
    for sentence in test_sentences:
        print(f"\n--- Testing: {sentence} ---")
        
        # Test __call__ method
        result = tokenizer(sentence, max_length=50, padding=True, truncation=True, return_tensors="pt")
        print(f"Input IDs: {result['input_ids']}")
        print(f"Attention Mask: {result['attention_mask']}")
        
        # Test decode
        decoded = tokenizer.decode(result['input_ids'].tolist())
        print(f"Decoded: {decoded}")
    
    # Test batch processing
    print(f"\n--- Testing Batch Processing ---")
    batch_result = tokenizer(test_sentences, max_length=50, padding=True, truncation=True, return_tensors="pt")
    print(f"Batch Input IDs shape: {batch_result['input_ids'].shape}")
    print(f"Batch Attention Mask shape: {batch_result['attention_mask'].shape}")
    
    # Test save and load
    print(f"\n--- Testing Save/Load ---")
    save_dir = "build_tokenizer/saved_tokenizer"
    tokenizer.save_pretrained(save_dir)
    
    # Load again
    loaded_tokenizer = CustomSentencePieceTokenizer.from_pretrained(save_dir)
    
    # Test loaded tokenizer
    result_loaded = loaded_tokenizer("Test câu này!", return_tensors="pt")
    print(f"Loaded tokenizer result: {result_loaded}")


if __name__ == "__main__":
    test_custom_tokenizer() 