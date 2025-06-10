from datasets import load_from_disk
import torch
torch.manual_seed(1337)
path = "D:/Downloads/DS_AI/VDT/MoE/moe/data/abstract_summarization/tokenized_right_wiki/"
train_dataset = load_from_disk(path + "train")
eval_dataset = load_from_disk(path + "eval")
test_dataset = load_from_disk(path + "test")