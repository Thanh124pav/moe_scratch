from datasets import Dataset
from transformers import AutoTokenizer
import random
from build_translation_dataset import BuildTranslationDataset

def create_pairs(path, translation=False, Vnews=False):
    train_pairs = []
    eval_pairs = []
    test_pairs = []
    if not translation:
        train_path = path + '/train.tsv'
        eval_path = path + '/eval.tsv'
        test_path = path + '/test.tsv'
        paths = [train_path, eval_path, test_path]
        src_idx = 1
        tgt_idx = 2
        if Vnews:
            src_idx = 3
            tgt_idx = 2
        for data_path in paths:
            pairs = []
            with open(data_path, encoding='utf-8') as file:
                for i, line in enumerate(file):
                    if i == 0:
                        continue
                    line = line.strip().split('\t')
                    pairs.append((line[src_idx], line[tgt_idx]))
            if data_path == train_path:
                train_pairs.extend(pairs)
            elif data_path == eval_path:
                eval_pairs.extend(pairs)
            else:
                test_pairs.extend(pairs)
        return train_pairs, eval_pairs, test_pairs
    else:
        train_path = path + '/train'
        test_path = path + '/test'
        paths = [train_path, test_path]
        input_lines = []
        label_lines = []
        for data_path in paths:
            pairs = []
            with open(data_path + '.en', encoding='utf-8') as file:
                for i, line in enumerate(file):
                    line = line.strip()
                    input_lines.append(line)
            with open(data_path + '.vi', encoding='utf-8') as file:
                for i, line in enumerate(file):
                    line = line.strip()
                    label_lines.append(line)
            for i in range(len(input_lines)):
                pairs.append((input_lines[i], label_lines[i]))
            if data_path == train_path:
                train_pairs.extend(pairs)
            else:
                test_pairs.extend(pairs)
        eval_pairs = random.sample(train_pairs, 1000)
        for item in eval_pairs:
            train_pairs.remove(item)
        return train_pairs, eval_pairs, test_pairs

if __name__ == '__main__':
    root_path = "D:/Downloads/DS_AI/VDT/MoE/moe"
    train_pairs, eval_pairs, test_pairs = create_pairs(root_path + '/data/raw/iwslt15', translation=True)
    train_pairs = random.sample(train_pairs, 13000)
    tokenizer = AutoTokenizer.from_pretrained("VietAI/vit5-base")
    translation_builder = BuildTranslationDataset(tokenizer)

    # Build decoder-only dataset (train/eval/test)
    train_do_translation = translation_builder.build_decoder_only_dataset(train_pairs)
    eval_do_translation = translation_builder.build_decoder_only_dataset(eval_pairs)
    test_do_translation = translation_builder.build_decoder_only_dataset(test_pairs, inference = True)
    train_do_mt_dataset = Dataset.from_list(train_do_translation)
    eval_do_mt_dataset = Dataset.from_list(eval_do_translation)
    test_do_mt_dataset = Dataset.from_list(test_do_translation)
    train_do_mt_dataset.save_to_disk(root_path + '/data/machine_translation/do/train')
    eval_do_mt_dataset.save_to_disk(root_path + '/data/machine_translation/do/eval')
    test_do_mt_dataset.save_to_disk(root_path + '/data/machine_translation/do/test')

    # Test inference input
    test_pairs = train_pairs[0:2]
    translation_builder.test_inference(test_pairs)