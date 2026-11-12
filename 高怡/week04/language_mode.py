import pandas as pd
import os
import csv
import math
import random
from collections import Counter
from torch.utils.data import DataLoader, TensorDataset
from torch import nn, optim
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VACAB_FILE = os.path.join(BASE_DIR, "vocab.csv")
TRAIN_FILE = f"{BASE_DIR}/corpus_general.txt"
DEV_FILE = f"{BASE_DIR}/dev.txt"
UNK_ID = 1
PAD_ID = 0
SEED = 42
BATCH_SIZE = 256
EPOCH_SIZE = 10
EMBEDDING_SIZE = 128
HIDDEN_SIZE = 128
NUM_OF_HEADS = 8
MAX_LENGTH = 80
LEARNING_RATE = 0.003
NUM_LAYERS = 2

def build_vocab_file():
    """
    用训练集数据构造字符表，按字符表中出现次数由高到低构造
    """
    lines = []
    with open(TRAIN_FILE, "r", encoding="utf-8") as f:
        for line in f:
            lines.append(line.strip())
    counter = Counter()
    for line in lines:
        counter.update(line)
    with open(VACAB_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["<PAD>", 0])
        writer.writerow(["<UNK>", 1])
        for char, count in counter.most_common():
            writer.writerow([char, count])

def load_vacab_file():
    """
    读取字符表文件并构造字符表
    """
    chars = pd.read_csv(VACAB_FILE, header=None, usecols=[0])[0]
    return {char: idx for idx, char in enumerate(chars)}

def get_vocab():
    """
    获取字符表
    """
    if not os.path.exists(VACAB_FILE):
        build_vocab_file()
    vocab = load_vacab_file()
    return vocab, len(vocab)

def load_data(file_path, vocab, max_len=MAX_LENGTH, shuffle=True):
    """
    读取数据文件生成torch.DataLoader
    """
    lines = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            lines.append(line.strip())
    seqs = [torch.LongTensor([vocab.get(c, UNK_ID) for c in t]) for t in lines]
    pad_sequence = []
    for seq in seqs:
        seq_len = len(seq)
        padding_count = max_len - seq_len
        if padding_count > 0:
            padding_array = torch.full((max_len,), PAD_ID, dtype=torch.long)
            padding_array[0:seq_len] = seq
            pad_sequence.append(padding_array)
        elif padding_count < 0:
            pad_sequence.append(seq[0:max_len])
        else:
            pad_sequence.append(seq)
    data = torch.stack(pad_sequence)
    x = data[:, :-1]
    y = data[:, 1:]
    return DataLoader(TensorDataset(x, y), batch_size=BATCH_SIZE, shuffle=shuffle)

class LanguageMode(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(input_size, embedding_size, padding_idx=0)
        self.pos_embedding = nn.Embedding(MAX_LENGTH, embedding_size)
        transformer_encode = nn.TransformerEncoderLayer(
            embedding_size, NUM_OF_HEADS, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(transformer_encode, NUM_LAYERS)
        self.liner = nn.Linear(embedding_size, input_size)
        self.loss = nn.functional.cross_entropy

    def input_embedding(self, x):
        y_em = self.embedding(x)
        pos = torch.arange(y_em.size(1), device=y_em.device)
        return y_em + self.pos_embedding(pos)

    def generate_causal_mask(self, seq_len, device):
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        # tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(device)
        return mask  # True 表示屏蔽

    def forward(self, x, y=None):
        y_em = self.input_embedding(x)
        pad_mask = (x == PAD_ID)
        causal_mask = self.generate_causal_mask(y_em.size(1), y_em.device)
        y_trans = self.transformer(y_em, causal_mask, pad_mask)
        y_pred = self.liner(y_trans)
        if y is None:
            return y_pred
        else:
            # print(f"y_pred shape: {y_pred.shape}")
            # print(f"y shape: {y.shape}")
            V = y_pred.size(-1)
            loss = self.loss(y_pred.reshape(-1, V), y.reshape(-1), ignore_index=PAD_ID)
            return loss

def eval(mode, validation_loader):
    mode.eval()
    total_loss = 0.0
    with torch.no_grad():
        for x_batch, y_batch in validation_loader:
            loss = mode(x_batch, y_batch)
            total_loss += loss.item()
    average_loss = total_loss / len(validation_loader)
    return average_loss, math.exp(average_loss)

def main():
    torch.manual_seed(SEED)
    random.seed(SEED)
    vocab, vocab_length = get_vocab()
    train_loader = load_data(TRAIN_FILE, vocab)
    validation_loader = load_data(DEV_FILE, vocab)
    mode = LanguageMode(vocab_length, EMBEDDING_SIZE, HIDDEN_SIZE)
    optimizer = optim.Adam(mode.parameters(), lr = LEARNING_RATE)
    for epoch in range(EPOCH_SIZE):
        mode.train()
        total_loss = 0.0
        for x_batch, y_batch in train_loader:
            loss = mode(x_batch, y_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        eval_loss, eval_perplexity = eval(mode, validation_loader)
        average_loss = total_loss/len(train_loader)
        perplexity = math.exp(average_loss)
        print(f"第{epoch + 1}轮训练集损失值：{average_loss:.5f}，困惑度：{perplexity}")
        print(f"第{epoch + 1}轮验证集损失值：{eval_loss:.5f}，困惑度：{eval_perplexity}")
    torch.save({
        "state_dict": mode.state_dict(),
        "vocab": vocab,
        "max_length": MAX_LENGTH,
        "embedding_size": EMBEDDING_SIZE,
        "hidden_size": HIDDEN_SIZE,
    }, "LanguageMode.bin")

if __name__ == "__main__":
    main()