from datasets import Dataset
from transformers import AutoTokenizer
import random
from build_translation_dataset import BuildTranslationDataset
from build_summarization_dataset import BuildSummarizationDataset

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

        for data_path in paths:
            pairs = []
            input_lines = []
            label_lines = []
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

def build_dataset(train_pairs, eval_pairs, test_pairs, tokenizer, save_path, translation=False, decode_only=False):
    if decode_only:
        if not translation:
            sum_builder = BuildSummarizationDataset(tokenizer)
            train_dataset = sum_builder.build_decoder_only_dataset(train_pairs)
            eval_dataset = sum_builder.build_decoder_only_dataset(eval_pairs)
            test_dataset = sum_builder.build_decoder_only_dataset(test_pairs, inference = True)
        else:
            translation_builder = BuildTranslationDataset(tokenizer)
            train_dataset = translation_builder.build_decoder_only_dataset(train_pairs)
            eval_dataset = translation_builder.build_decoder_only_dataset(eval_pairs)
            test_dataset = translation_builder.build_decoder_only_dataset(test_pairs, inference = True)
    else:
        if not translation:
            sum_builder = BuildSummarizationDataset(tokenizer)
            train_dataset = sum_builder.build_encoder_decoder_dataset(train_pairs)
            eval_dataset = sum_builder.build_encoder_decoder_dataset(eval_pairs)
            test_dataset = sum_builder.build_encoder_decoder_dataset(test_pairs)
        else:
            translation_builder = BuildTranslationDataset(tokenizer)
            train_dataset = translation_builder.build_encoder_decoder_dataset(train_pairs)
            eval_dataset = translation_builder.build_encoder_decoder_dataset(eval_pairs)
            test_dataset = translation_builder.build_encoder_decoder_dataset(test_pairs)

    train_dataset = Dataset.from_list(train_dataset)
    eval_dataset = Dataset.from_list(eval_dataset)
    test_dataset = Dataset.from_list(test_dataset)
    train_dataset.save_to_disk(save_path + '/train')
    eval_dataset.save_to_disk(save_path + '/eval')
    test_dataset.save_to_disk(save_path + '/test')
    print("Size of the datasets:")
    print(len(train_dataset))
    print(len(eval_dataset))
    print(len(test_dataset))



if __name__ == '__main__':
    root_path = "D:/Downloads/DS_AI/VDT/MoE/moe/data/raw/iwslt15"
    save_path = "D:/Downloads/DS_AI/VDT/MoE/moe/data/translation/ed"
    tokenizer = AutoTokenizer.from_pretrained("VietAI/vit5-base")
    tokenizer.padding_side = 'right'
    train_pairs, eval_pairs, test_pairs = create_pairs(root_path, translation=True)
    train_pairs = random.sample(train_pairs, 13000)
    print(len(train_pairs))
    print(len(eval_pairs))
    print(len(test_pairs))
    build_dataset(train_pairs, eval_pairs, test_pairs, tokenizer, save_path, translation = True, decode_only = False)


