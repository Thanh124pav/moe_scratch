import numpy as np
import torch
from torch.utils.data import DataLoader
from collections import Counter
from gensim.models import Word2Vec
from transformers import AutoTokenizer
from datasets import load_from_disk

from collections import Counter

tokenizer = AutoTokenizer.from_pretrained(r'D:\Downloads\DS_AI\VDT\MoE\moe\tokenized_data\do_tokenizer')

train_mt = load_from_disk(r'D:\Downloads\DS_AI\VDT\MoE\moe\data\translation\do\train')
train_as = load_from_disk(r'D:\Downloads\DS_AI\VDT\MoE\moe\data\summarization\do\train')

decode_mt = tokenizer.batch_decode(train_mt['input_ids'])
print(decode_mt[0])

decode_as = tokenizer.batch_decode(train_as['input_ids'])

all_words = []
for sent in decode_mt: # decode_mt_list là list các câu
    all_words.extend(sent.split())
for sent in decode_as:
    all_words.extend(sent.split())


class Dataset:
    def __init__(self, decoded_texts, w2v_model=None, embedding_dim=256, batch_size=200, as_tensor=True, device='cpu', train_w2v=False):
        """
        decoded_texts: list các câu đã decode (list[str])
        w2v_model: đường dẫn file .model của gensim hoặc None để train mới
        embedding_dim: số chiều embedding
        train_w2v: True nếu muốn train Word2Vec mới từ dữ liệu
        """
        self.texts = decoded_texts
        self.bow = self.build_bow(self.texts)
        self.vocab = sorted(self.bow.keys())
        self.vocab_size = len(self.vocab)
        self.word2idx = {w: i for i, w in enumerate(self.vocab)}
        self.bow_matrix = self.texts_to_bow_matrix(self.texts)
        self.embedding_dim = embedding_dim

        # Train hoặc load Word2Vec
        if train_w2v or w2v_model is None:
            # Chuẩn bị dữ liệu cho Word2Vec (list các list từ)
            sentences = [sent.split() for sent in self.texts]
            self.w2v = Word2Vec(sentences, vector_size=embedding_dim, window=5, min_count=1, workers=4)
        else:
            self.w2v = Word2Vec.load(w2v_model)

        self.word_embeddings = self.build_w2v_embedding_matrix(self.w2v, self.vocab, embedding_dim)

        if as_tensor:
            self.bow_matrix = torch.from_numpy(self.bow_matrix).float().to(device)
            self.word_embeddings = torch.from_numpy(self.word_embeddings).float().to(device)
            self.dataloader = DataLoader(self.bow_matrix, batch_size=batch_size, shuffle=True)

    def build_bow(self, texts):
        all_words = []
        for sent in texts:
            all_words.extend(sent.split())
        return Counter(all_words)

    def texts_to_bow_matrix(self, texts):
        mat = np.zeros((len(texts), self.vocab_size), dtype=np.float32)
        for i, sent in enumerate(texts):
            for word in sent.split():
                if word in self.word2idx:
                    mat[i, self.word2idx[word]] += 1
        return mat

    def build_w2v_embedding_matrix(self, w2v, vocab, embedding_dim):
        embedding_matrix = np.random.normal(scale=0.6, size=(len(vocab), embedding_dim)).astype(np.float32)
        for i, word in enumerate(vocab):
            if word in w2v.wv:
                embedding_matrix[i] = w2v.wv[word]
        return embedding_matrix
