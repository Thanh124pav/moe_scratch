from datasets import load_from_disk
import torch
torch.manual_seed(1337)

def load_data(path):
    train_dataset = load_from_disk(path + "train")
    eval_dataset = load_from_disk(path + "eval")
    test_dataset = load_from_disk(path + "test")
    return train_dataset, eval_dataset, test_dataset