#!/usr/bin/env python3
"""
Complete pipeline để build tokenizer từ raw data
"""

import os
import sys
import subprocess
import time

def run_script(script_path, description):
    """Chạy một script và báo cáo kết quả"""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # Chạy script
        result = subprocess.run([sys.executable, script_path], 
                              capture_output=True, 
                              text=True,
                              cwd=os.getcwd())
        
        if result.returncode == 0:
            print(f"✅ {description} completed successfully!")
            if result.stdout:
                print("Output:")
                print(result.stdout)
        else:
            print(f"❌ {description} failed!")
            if result.stderr:
                print("Error:")
                print(result.stderr)
            if result.stdout:
                print("Output:")
                print(result.stdout)
            return False
            
    except Exception as e:
        print(f"❌ Error running {script_path}: {e}")
        return False
    
    end_time = time.time()
    print(f"⏱️ Time taken: {end_time - start_time:.2f} seconds")
    
    return True

def check_prerequisites():
    """Kiểm tra các requirements"""
    print("🔍 Checking prerequisites...")
    
    # Check if raw data exists
    data_dirs = [
        "data/raw/Wiki",
        "data/raw/VNews", 
        "data/raw/iwslt15"
    ]
    
    found_dirs = []
    for data_dir in data_dirs:
        if os.path.exists(data_dir):
            found_dirs.append(data_dir)
            print(f"✅ Found: {data_dir}")
        else:
            print(f"⚠️ Not found: {data_dir}")
    
    if not found_dirs:
        print("❌ No raw data directories found!")
        print("Please ensure you have data in one of these directories:")
        for data_dir in data_dirs:
            print(f"  - {data_dir}")
        return False
    
    # Check Python packages
    required_packages = ['sentencepiece', 'datasets', 'torch', 'tqdm']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ Package available: {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ Package missing: {package}")
    
    if missing_packages:
        print(f"\n❌ Missing packages: {missing_packages}")
        print("Please install them using:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("✅ All prerequisites satisfied!")
    return True

def main():
    """Main pipeline"""
    print("🌟 Vietnamese Custom Tokenizer Build Pipeline")
    print("=" * 60)
    
    # Check prerequisites
    if not check_prerequisites():
        print("\n❌ Prerequisites not satisfied. Exiting...")
        return
    
    # Create build directory
    os.makedirs("build_tokenizer", exist_ok=True)
    
    # Pipeline steps
    steps = [
        ("build_tokenizer/collect_raw_text.py", "Collecting raw text for tokenizer training"),
        ("build_tokenizer/train_sentencepiece.py", "Training SentencePiece tokenizer"),
        ("build_tokenizer/custom_tokenizer.py", "Testing custom tokenizer wrapper"),
        ("build_tokenizer/process_data_with_custom_tokenizer.py", "Processing data with custom tokenizer")
    ]
    
    total_start_time = time.time()
    
    for script_path, description in steps:
        if not os.path.exists(script_path):
            print(f"❌ Script not found: {script_path}")
            print("Please make sure all scripts are created properly.")
            return
        
        success = run_script(script_path, description)
        if not success:
            print(f"\n❌ Pipeline failed at step: {description}")
            return
    
    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    
    print("\n" + "🎉" * 20)
    print("🎉 PIPELINE COMPLETED SUCCESSFULLY! 🎉")
    print("🎉" * 20)
    print(f"⏱️ Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    
    # Summary
    print("\n📋 SUMMARY:")
    print("=" * 40)
    
    # Check outputs
    outputs = {
        "Raw text for tokenizer": "build_tokenizer/raw_text_for_tokenizer.txt",
        "SentencePiece models": "build_tokenizer/vietnamese_tokenizer_*.model",
        "Custom tokenizer": "build_tokenizer/saved_tokenizer/",
        "Processed data": "data/custom_tokenized/",
        "Processing info": "data/custom_tokenized/processing_info.json"
    }
    
    for name, path in outputs.items():
        if "*" in path:
            # Check wildcard pattern
            import glob
            files = glob.glob(path)
            if files:
                print(f"✅ {name}: {len(files)} files")
                for f in files:
                    print(f"   - {f}")
            else:
                print(f"❌ {name}: Not found")
        elif os.path.exists(path):
            if os.path.isfile(path):
                size = os.path.getsize(path) / (1024*1024)  # MB
                print(f"✅ {name}: {path} ({size:.2f} MB)")
            else:
                print(f"✅ {name}: {path} (directory)")
        else:
            print(f"❌ {name}: {path} (not found)")
    
    print("\n🚀 NEXT STEPS:")
    print("1. Choose your preferred tokenizer model (16K, 32K, or 50K vocab)")
    print("2. Update your training scripts to use the custom tokenizer")
    print("3. Use data from 'data/custom_tokenized/' for training")
    
    # Sample usage
    print("\n💡 SAMPLE USAGE:")
    print("```python")
    print("from build_tokenizer.custom_tokenizer import CustomSentencePieceTokenizer")
    print("")
    print("# Load tokenizer")
    print("tokenizer = CustomSentencePieceTokenizer.from_pretrained('data/custom_tokenized/tokenizer')")
    print("")
    print("# Load data")
    print("from datasets import load_from_disk")
    print("train_dataset = load_from_disk('data/custom_tokenized/summarization_train')")
    print("")
    print("# Use in your training script")
    print("```")

if __name__ == "__main__":
    main() 