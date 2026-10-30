import pandas as pd
import os
import csv
import random
from collections import Counter
from torch.utils.data import DataLoader, TensorDataset
from torch import nn, optim
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_SET_PATH = os.path.join(BASE_DIR, "data/langid/v12")
VACAB_FILE = os.path.join(BASE_DIR, "data/langid/vocab/vocab_v12.csv")
TRAIN_FILE = f"{DATA_SET_PATH}/train.csv"
DEV_FILE = f"{DATA_SET_PATH}/dev.csv"
TEST_FILE = f"{DATA_SET_PATH}/test.csv"
# LANGUAGE_DICT = {"zh":0, "ja":1, "en":2, "es":3, "pt":4, "ru":5, "ar":6, "hi":7, "ko":8, "th":9, "vi":10, "tr":11}
LANGUAGE_ARRAY = ["zh", "ja", "en", "es", "pt", "ru", "ar", "hi", "ko", "th", "vi", "tr"]
UNK_ID = 1
PAD_ID = 0
SEED = 42
BATCH_SIZE = 96
EPOCH_SIZE = 10
EMBEDDING_SIZE = 70
HIDDEN_SIZE = 128
MAX_LENGTH = 70
LEARNING_RATE = 0.003

def build_vocab_file():
    """
    用训练集数据构造字符表，按字符表中出现次数由高到低构造
    """
    df = pd.read_csv(TRAIN_FILE, header=0, names=["text", "label"])
    counter = Counter()
    for line in df["text"]:
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
    df = pd.read_csv(file_path, header=0, names=["text", "label"])
    seqs = [torch.LongTensor([vocab.get(c, UNK_ID) for c in t]) for t in df["text"]]
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
    x = torch.stack(pad_sequence)
    label_list = [LANGUAGE_ARRAY.index(label) for label in df["label"]]
    y = torch.LongTensor(label_list)
    return DataLoader(TensorDataset(x, y), batch_size=BATCH_SIZE, shuffle=shuffle)

class LanguageClassificationMode(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size):
        super().__init__()
        self.embedding = nn.Embedding(input_size, embedding_size, padding_idx=0)
        self.rnn = nn.LSTM(embedding_size, hidden_size, batch_first=True)
        self.norm = nn.LayerNorm(normalized_shape=hidden_size)
        self.dropout = nn.Dropout(p=0.5)
        self.liner = nn.Linear(hidden_size, output_size)
        # self.liner = nn.Linear(hidden_size * 2, output_size)
        self.loss = nn.functional.cross_entropy

    def forward(self, x, y=None):
        y_em = self.embedding(x)
        y_rnn, _ = self.rnn(y_em)
        y_norm = self.norm(y_rnn)
        mask = (x != PAD_ID).unsqueeze(-1)
        # y_pool = y_norm.masked_fill(~mask, float("-inf")).max(dim=1).values
        y_pool = (y_norm * mask).sum(1) / mask.sum(1).clamp(min=1)
        # y_max = y_norm.masked_fill(~mask, float("-inf")).max(dim=1).values
        # y_mean = (y_norm * mask).sum(1) / mask.sum(1).clamp(min=1)
        # y_pool = torch.cat([y_max, y_mean], dim=1)      # (B, 2H)
        y_drop = self.dropout(y_pool)
        y_pred = self.liner(y_drop)
        if y == None:
            return y_pred
        else:
            return self.loss(y_pred, y)

def eval(mode, validation_loader):
    mode.eval()
    correct = 0
    total = 0
    # accuracy = 0
    # summary = {}
    with torch.no_grad():
        # summary, accuracy = generate_summary_report(mode, validation_loader)
        for x_batch, y_batch in validation_loader:
            y_logist = mode(x_batch)
            y_pred = torch.argmax(y_logist, dim=1)
            correct += (y_pred == y_batch).sum().item()
            total += y_batch.size(0)
    return correct / total
    # return summary, accuracy

def main():
    torch.manual_seed(SEED)
    random.seed(SEED)
    vocab, vocab_length = get_vocab()
    train_loader = load_data(TRAIN_FILE, vocab)
    validation_loader = load_data(DEV_FILE, vocab)
    mode = LanguageClassificationMode(vocab_length, EMBEDDING_SIZE, HIDDEN_SIZE, len(LANGUAGE_ARRAY))
    optimizer = optim.Adam(mode.parameters(), lr = LEARNING_RATE)
    for epoch in range(EPOCH_SIZE):
        mode.train()
        epoch_loss = None
        for x_batch, y_batch in train_loader:
            loss = mode(x_batch, y_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss = loss.item()
        accuracy = eval(mode, validation_loader)
        print(f"第{epoch + 1}轮损失值：{epoch_loss:.5f}，准确度：{accuracy :.2f}")
    torch.save({
        "state_dict": mode.state_dict(),
        "vocab": vocab,
        "labels": LANGUAGE_ARRAY,
        "max_length": MAX_LENGTH,
        "embedding_size": EMBEDDING_SIZE,
        "hidden_size": HIDDEN_SIZE,
    }, "LanguageClassificationMode.bin")

def generate_summary_report(mode, loader, texts=None):
    correct = 0
    total = 0
    summary = {}
    confusion = {}
    idx = 0
    for x_batch, y_batch in loader:
        y_logist = mode(x_batch)
        y_pred_batch = torch.argmax(y_logist, dim=1)
        correct += (y_pred_batch == y_batch).sum().item()
        total += y_batch.size(0)
        for y_pred, y in zip(y_pred_batch, y_batch):
            y_key = LANGUAGE_ARRAY[y.item()]
            y_pred_key = LANGUAGE_ARRAY[y_pred.item()]
            if y_key not in summary:
                summary[y_key] = {}
                summary[y_key]["total"] = 0
                summary[y_key]["correct"] = 0
                summary[y_key]["TP"] = 0
                summary[y_key]["FP"] = 0
                summary[y_key]["FN"] = 0
            summary[y_key]["total"] += 1
            if y_pred == y:
                summary[y_key]["correct"] +=1
                summary[y_key]["TP"] += 1
            if y_pred != y:
                summary[y_key]["FN"] += 1
                summary[y_pred_key]["FP"] += 1
                pair = (y_key, y_pred_key)
                confusion[pair] = confusion.get(pair, 0) + 1
                if texts is not None:
                    print(f"{pair}: {texts[idx]}")
            idx += 1
    print(sorted(confusion.items(), key=lambda kv: -kv[1]))
    return summary, correct / total

def test(mode_path):
    checkpoint = torch.load(mode_path)
    vocab = checkpoint["vocab"]
    mode = LanguageClassificationMode(len(vocab), checkpoint["embedding_size"],
                                      checkpoint["hidden_size"], len(checkpoint["labels"]))
    mode.load_state_dict(checkpoint["state_dict"])
    mode.eval()
    test_loader = load_data(TEST_FILE, vocab, max_len=checkpoint["max_length"], shuffle=False)
    texts = pd.read_csv(TEST_FILE, header=0, names=["text", "label"])["text"].tolist()
    with torch.no_grad():
        summary, accuracy = generate_summary_report(mode, test_loader, texts)
    print("测试集各语言结果：")
    for key in summary.keys():
        class_accuracy = summary[key]["correct"] / summary[key]["total"]
        precision = summary[key]["TP"] / (summary[key]["TP"] + summary[key]["FP"])
        recall = summary[key]["TP"] / (summary[key]["TP"] + summary[key]["FN"])
        f1_score = 2 * precision * recall / (precision + recall )
        print(f"{key}: 准确度{class_accuracy}, 精确率{precision}, 召回率{recall}, F1分数{f1_score}")
    print(f"测试集准确度：{accuracy}")

if __name__ == "__main__":
    # main()
    test("LanguageClassificationMode.bin")