"""
作业：根据老师给的12种语言数据，训练一个多语种识别模型
模型：使用week3中学到的，Embedding -> CNN/RNN/LSTM -> max/mean pooling -> layernorm -> dropout -> linear
损失函数：交叉熵CrossEntropyLoss
优化器：Adam
数据来源：第三课 深度学习处理文本/week3-深度学习常用组件/语种识别实战/data/langid/v12
"""

from genericpath import isfile
import os
import csv
import time
import random
import argparse
import unicodedata
from collections import Counter, OrderedDict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ─── 超参数 ────────────────────────────────────────────────────────────
SEED = 42
MAX_LEN = 40          # 中文中位长度约 10 字符、德语约 39，统一截到 40
EMBED_DIM = 64
HIDDEN_DIM = 64
DROPOUT = 0.3
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 8
NUM_THREADS = 1       # 小模型单线程比 16 线程快 2.7 倍（实测），别改成 16

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data", "langid", "v12")
VOCAB_PATH = os.path.join(HERE, "data", "langid", "vocab", "vocab_v12.txt")

# 这里是12个语种
LABELS = ["ar", "en", "es", "hi", "ja", "pt", "ru", "zh","ko","th","vi","tr"]

# ─── 1. 文本归一化（必须和建数据时一致）────────────────────────────────
def normalize(text):
    t = unicodedata.normalize("NFKC",text).lower()
    return "".join(c for c in t if unicodedata.category(c)[0] in ("L","M"))


# ─── 2. 字符表构建 ─────────────────────────────────────────────────────
def build_char_vocab(sentences, min_freq=1):
    freq = Counter()
    for s in sentences:
        freq.update(normalize(s))
    chars = sorted(((c,n) for c,n in freq.items() if n >= min_freq),key=lambda kv:(-kv[1],kv[0]))
    vocab = OrderedDict([("<PAD>",0),("<UNK>",1)])
    for ch,_ in chars:
        vocab[ch] = len(vocab)
    return vocab,freq


def load_vocab_file(path):
    vocab = OrderedDict()
    with open(path,encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            ch, idx, _freq = line.rstrip("\n").split("\t")
            vocab[ch] = int(idx)
    return vocab

def save_vocab(vocab, freq, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path,"w",encoding="utf-8") as f:
        f.write("# 字符\tid\t出现次数\n")
        for ch,idx in vocab.items():
            f.write(f"{ch}\t{idx}\t{freq.get(ch,0)}\n")

def encode(text, vocab, max_len=MAX_LEN):
    text = normalize(text)
    text_ids = [vocab.get(c, 1) for c in text][:max_len]  # 1 是 <UNK>
    text_ids += [0] * (max_len - len(text_ids))  # 0 是 <PAD>
    return text_ids

# ─── 3. 读数据 ─────────────────────────────────────────────────────────
def load_split(data_dir, name):
    path = os.path.join(data_dir,f"{name}.csv")
    rows = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append((r["text"], r["label"]))
    return rows

class TextDataset(Dataset):
    def __init__(self, rows, vocab, label2id):
        self.X = [encode(t,vocab) for t,_ in rows]
        self.y = [label2id[l] for _,l in rows]
        self.label2id = label2id

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.long), torch.tensor(self.y[idx], dtype=torch.long)

# ─── 4. 模型 ───────────────────────────────────────────────────────────
class LangIDClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM,  dropout=DROPOUT):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.pooling = nn.AdaptiveMaxPool1d(1)
        self.layernorm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        x,_ = self.lstm(x)      # (batch_size, seq_len, hidden_dim)
        x = x.transpose(1,2)   # (batch_size, hidden_dim, seq_len)
        x = self.pooling(x).squeeze(-1)  # (batch_size, hidden_dim)
        x = self.layernorm(x)
        x = self.dropout(x)
        logits = self.fc(x)    # (batch_size, num_classes)
        return logits

# ─── 5. 评测 ───────────────────────────────────────────────────────────
def predict(model, loader):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for X, y in loader:
            logits = model(X)
            pred = logits.argmax(dim=1)
            y_true.extend(y.tolist())
            y_pred.extend(pred.tolist())
    return y_true, y_pred

def marco_f1(y_pred, y_true , num_classes):
    # 计算每个类别的精确率、召回率和F1分数
    f1_scores = []
    for i in range(num_classes):
        tp = sum((p == i and t == i) for p, t in zip(y_pred, y_true))
        fp = sum((p == i and t != i) for p, t in zip(y_pred, y_true))
        fn = sum((p != i and t == i) for p, t in zip(y_pred, y_true))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores.append(f1)
    macro_f1 = sum(f1_scores) / len(f1_scores)
    return macro_f1

def confusion(preds, golds, labels):
    n = len(labels)
    mat = [[0] * n for _ in range(n)]
    for p, g in zip(preds, golds):
        mat[g][p] += 1
    return mat


def print_confusion(mat, labels, top=3):
    """只打印错的格子，按错误数排序 —— 课堂上看 zh/ja 那一格就够了。"""
    pairs = [(mat[g][p], labels[g], labels[p])
             for g in range(len(labels)) for p in range(len(labels)) if g != p]
    pairs.sort(reverse=True)
    print("  最常混淆的组合（真实 → 预测）：")
    for cnt, g, p in pairs[:top]:
        if cnt == 0:
            break
        print(f"    {g} → {p}: {cnt} 条")


# ─── 6. 训练 ───────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--vocab", default=VOCAB_PATH,
                        help="字符表文件（默认用课堂那张 vocab_v8.txt）")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--per-class", type=int, default=0,
                        help="只用每类前 N 条训练，用来演示「数据少了会过拟合」")
    ap.add_argument("--threads", type=int, default=NUM_THREADS)
    ap.add_argument("--save", default="", help="保存模型到该路径，供 eval_langid.py 使用")
    args = ap.parse_args()

    # 设置系统进程和种子
    torch.set_num_threads(args.threads)
    random.seed(SEED)
    torch.manual_seed(SEED)

    # 加载数据集
    train_rows = load_split(args.data_dir,"train")
    dev_rows = load_split(args.data_dir, "dev")
    test_rows = load_split(args.data_dir, "test")

    # 判读标签是否在 LABELS 中
    seen = sorted(label for _,label in train_rows)
    extra = [l for l in seen if l not in LABELS]
    if extra:
        raise SystemExit(
            f"数据里出现了 LABELS 之外的标签：{' '.join(extra)}\n"
            f"  当前 LABELS（{len(LABELS)} 类）：{' '.join(LABELS)}\n"
            f"  模型输出维度 = len(LABELS)，多出来的语种没有对应的输出格子。"
        )
    label2id = {l: i for i, l in enumerate(LABELS)}
    num_classes = len(LABELS)
    # 下采样
    if args.per_class > 0:
        by_label = {}
        for t,l in train_rows:
            by_label.setdefault(l, []).append(t)
        train_rows = [(t, l) for l, ts in by_label.items()
                      for t in random.Random(SEED).sample(ts, min(args.per_class, len(ts)))]
        random.Random(SEED).shuffle(train_rows)

    vocab = load_vocab_file(args.vocab)
    print(f"数据目录：{args.data_dir}，训练集 {len(train_rows)} 条，验证集 {len(dev_rows)} 条，测试集 {len(test_rows)} 条")
    print(f"  类别 {num_classes} 个：{' '.join(LABELS)}")
    print(f"  字符表 {len(vocab)} 个字符（来自 {os.path.basename(args.vocab)}）")
    print(f"  随机猜的准确率基准：{1/num_classes:.1%}")

    train_loader = DataLoader(TextDataset(train_rows, vocab, label2id), batch_size=BATCH_SIZE, shuffle=True)
    dev_loader = DataLoader(TextDataset(dev_rows, vocab, label2id), batch_size=128)
    test_loader = DataLoader(TextDataset(test_rows, vocab, label2id), batch_size=128)

    model = LangIDClassifier(len(vocab), num_classes)
    print(f"模型参数量：{sum(p.numel() for p in model.parameters())}，约 {sum(p.numel() for p in model.parameters())/1e3:.1f}K")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for X, y in train_loader:
            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        pred_y, true_y = predict(model, dev_loader)
        acc = sum(t==p for t,p in zip(true_y,pred_y)) / len(true_y)
        print(f"epoch {epoch}/{args.epochs}，训练集 loss {epoch_loss/len(train_loader):.4f}，验证集准确率 {acc:.1%}")
    elapsed = time.time() - t0
    print(f"训练完成，耗时 {elapsed:.1f} 秒 ({args.epochs} 个 epoch，{elapsed/args.epochs:.1f} 秒/epoch)")

    # 测试集评测
    y_true, y_pred = predict(model, test_loader)
    acc = sum(t==p for t,p in zip(y_true,y_pred)) / len(y_true)
    macro_f1 = marco_f1(y_pred, y_true, num_classes)
    print(f"测试集准确率 {acc:.1%}，宏平均F1分数 {macro_f1:.1%}")
    print_confusion(confusion(y_pred,y_true,LABELS),LABELS)

    # 逐类打印准确率
    print("\n  各语种准确率：")
    for c, lab in enumerate(LABELS):
            idx = [i for i, g in enumerate(y_true) if g == c]
            hit = sum(y_pred[i] == c for i in idx)
            print(f"    {lab}: {hit}/{len(idx)} = {hit/max(1,len(idx)):.3f}")

    # 推理示例
    print("\n--- 推理示例 ---")
    model.eval()
    demos = [t for t, _ in test_rows[:4]]
    with torch.no_grad():
        for text in demos:
            logits = model(torch.tensor([encode(text, vocab)]))
            prob = torch.softmax(logits, dim=1)[0]
            pid = int(prob.argmax())
            print(f"  [{LABELS[pid]} {prob[pid]:.2f}]  {text[:50]}")

    if args.save:
        torch.save({"state_dict": model.state_dict(), "vocab": dict(vocab),
                    "labels": LABELS, "maxlen": MAX_LEN,
                    "embed_dim": EMBED_DIM, "hidden_dim": HIDDEN_DIM,
                    "dropout": DROPOUT}, args.save)
        print(f"\n模型已保存：{args.save}")

if __name__ == "__main__":
    # 判断v12字典表是否存在，如果不存在则创建
    if not os.path.isfile(VOCAB_PATH):
        with open(os.path.join(DATA_DIR, "train.csv"), encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        vocab, freq = build_char_vocab([r["text"] for r in rows])
        save_vocab(vocab,freq,VOCAB_PATH)
        print("首次使用，创建字符表完成")
    main()
