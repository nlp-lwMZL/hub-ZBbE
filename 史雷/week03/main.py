# -*- coding: utf-8 -*-
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
# 12 语种里有天城文(印地语)、泰文：一个"视觉字符"要 3~4 个码点，
# 40 会把泰语/印地语截得太狠，所以比课堂版放宽到 48（CPU 仍可 1 分钟内跑完）。
MAX_LEN = 48
EMBED_DIM = 64
HIDDEN_DIM = 64
DROPOUT = 0.3
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 8
NUM_THREADS = 1       # 小模型单线程比多线程快，保持和 demo 一致

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data", "langid", "v12")          # ← v8 → v12
VOCAB_PATH = os.path.join(HERE, "data", "langid", "vocab", "vocab_v12.txt")  # ← v8 → v12

# ─── 作业版 12 语种：模型输出就是这 12 个格子 ──────────────────────────
# 顺序与 build_langid_dataset.py 的 LANG_V12 完全一致，方便对照
LABELS = [
    "eng", "fra", "deu", "spa", "rus", "cmn", "jpn", "kor",  # v8 课堂 8 种
    "ara", "hin", "tha", "vie",                                # 作业新增 4 种
]

# 仅用于打印，方便课堂上看结果
LANG_NAME = {
    "eng": "英语", "fra": "法语", "deu": "德语", "spa": "西班牙语",
    "rus": "俄语", "cmn": "普通话", "jpn": "日语", "kor": "韩语",
    "ara": "阿拉伯语", "hin": "印地语", "tha": "泰语", "vie": "越南语",
}


# ─── 1. 文本归一化（必须和建数据时一致）────────────────────────────────
def normalize(text):
    """
    NFKC 归一化 + 转小写 + 只保留「字母(L)」与「组合符号(M)」。

    保留 M 是 12 语种版的关键：印地语(天城文)、泰语的元音/声调是组合标记，
    只留 L 会把 हिन्दी、ภาษาไทย 削成裸辅音骨架，两个语种立刻失去区分度。
    这也是 build_langid_dataset.py 清洗时坚持保留 M 的原因。
    """
    t = unicodedata.normalize("NFKC", text).lower()
    return "".join(c for c in t if unicodedata.category(c)[0] in ("L", "M"))


# ─── 2. 字符表构建 / 读取 ──────────────────────────────────────────────
def build_char_vocab(sentences, min_freq=1):
    """
    字符表 = 训练集里所有出现过的字符，每个字符发一个整数编号。

    第 0 号 <PAD>：补齐用（padding_idx=0，向量恒为 0、不参与训练）
    第 1 号 <UNK>：没见过的字符

    加语种后必须重新生成这张表，否则新语种的字符会全部落到 1 号 <UNK>，
    模型等于睁眼瞎。本脚本主流程读 vocab_v12.txt；如果你想现场重建：

        rows = load_split("data/langid/v12", "train")
        vocab = build_char_vocab([t for t, _ in rows])
    """
    freq = Counter()
    for s in sentences:
        freq.update(normalize(s))
    chars = sorted(((c, n) for c, n in freq.items() if n >= min_freq),
                   key=lambda kv: (-kv[1], kv[0]))
    vocab = OrderedDict([("<PAD>", 0), ("<UNK>", 1)])
    for ch, _ in chars:
        vocab[ch] = len(vocab)
    return vocab


def load_vocab_file(path):
    """读「字符 \\t id \\t 出现次数」格式的字符表（# 开头为注释行）。"""
    vocab = OrderedDict()
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            ch, idx, _freq = line.rstrip("\n").split("\t")
            vocab[ch] = int(idx)
    return vocab


def encode(text, vocab, maxlen=MAX_LEN):
    ids = [vocab.get(ch, 1) for ch in normalize(text)][:maxlen]
    ids += [0] * (maxlen - len(ids))
    return ids


# ─── 3. 读数据（关键修复：兼容 lang / label 两种字段）──────────────────
def load_split(data_dir, name):
    """
    读 csv 的 (text, 语种标签) 对。

    build_langid_dataset.py 写出的字段是 id, lang, text；
    课堂 demo 写的是 label。这里两种都兼容，避免作业时还要去改数据。
    """
    path = os.path.join(data_dir, f"{name}.csv")
    if not os.path.exists(path):
        raise SystemExit(
            f"找不到数据文件：{path}\n"
            f"请先运行 build_langid_dataset.py 生成 v12 数据集。"
        )
    rows = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            lang = r.get("lang") or r.get("label")
            if not lang:
                raise SystemExit(f"{path} 既没有 lang 也没有 label 列，字段为 {list(r)}")
            rows.append((r["text"], lang))
    return rows


class TextDataset(Dataset):
    def __init__(self, rows, vocab, label2id):
        self.X = [encode(t, vocab) for t, _ in rows]
        self.y = [label2id[l] for _, l in rows]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return torch.tensor(self.X[i], dtype=torch.long), self.y[i]


# ─── 4. 模型（结构与 demo 完全一致）─────────────────────────────────────
class LangIDClassifier(nn.Module):
    """
    Embedding → RNN → 掩码 MaxPool → LayerNorm → Dropout → Linear

    MaxPool 前的 masked_fill 是为了让补齐位不参与池化：
    补齐位是全 0 向量，RNN 照样会往前走，池化时会把"读过一串 0"的状态也算进来。
    mask 成 -inf 后取最大值就干净了。
    """

    def __init__(self, vocab_size, num_classes,
                 embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, dropout=DROPOUT):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        mask = (x != 0).unsqueeze(-1)
        h, _ = self.rnn(self.embedding(x))
        h = h.masked_fill(~mask, -1e9)
        pooled = h.max(dim=1)[0]
        pooled = self.dropout(self.norm(pooled))
        return self.fc(pooled)


# ─── 5. 评测（结构与 demo 一致）────────────────────────────────────────
def predict(model, loader):
    model.eval()
    preds, golds = [], []
    with torch.no_grad():
        for X, y in loader:
            preds += model(X).argmax(dim=1).tolist()
            golds += y.tolist()
    return preds, golds


def macro_f1(preds, golds, num_classes):
    f1s = []
    for c in range(num_classes):
        tp = sum(p == c and g == c for p, g in zip(preds, golds))
        fp = sum(p == c and g != c for p, g in zip(preds, golds))
        fn = sum(p != c and g == c for p, g in zip(preds, golds))
        if tp + fp + fn == 0:
            continue
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return sum(f1s) / len(f1s)


def confusion(preds, golds, labels):
    n = len(labels)
    mat = [[0] * n for _ in range(n)]
    for p, g in zip(preds, golds):
        mat[g][p] += 1
    return mat


def print_confusion(mat, labels, top=5):
    """12 类下 top 从 3 调到 5，把拉丁语系那堆互相混淆的都露出来。"""
    pairs = [(mat[g][p], labels[g], labels[p])
             for g in range(len(labels)) for p in range(len(labels)) if g != p]
    pairs.sort(reverse=True)
    print("  最常混淆的组合（真实 → 预测）：")
    for cnt, g, p in pairs[:top]:
        if cnt == 0:
            break
        print(f"    {g} → {p}: {cnt} 条   "
              f"({LANG_NAME.get(g, g)} → {LANG_NAME.get(p, p)})")


# ─── 6. 训练 ───────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--vocab", default=VOCAB_PATH)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--per-class", type=int, default=0,
                    help="只用每类前 N 条训练，用来演示「数据少了会过拟合」")
    ap.add_argument("--threads", type=int, default=NUM_THREADS)
    ap.add_argument("--save", default="", help="保存模型路径，供 eval 使用")
    args = ap.parse_args()

    if not os.path.exists(args.vocab):
        raise SystemExit(
            f"字符表不存在：{args.vocab}\n"
            f"请先运行 build_langid_dataset.py 生成 vocab_v12.txt。"
        )

    torch.set_num_threads(args.threads)
    random.seed(SEED)
    torch.manual_seed(SEED)

    train_rows = load_split(args.data_dir, "train")
    dev_rows = load_split(args.data_dir, "dev")
    test_rows = load_split(args.data_dir, "test")

    # 数据里的标签必须和 LABELS 对得上
    seen = sorted({l for _, l in train_rows})
    extra = [l for l in seen if l not in LABELS]
    if extra:
        raise SystemExit(
            f"数据里出现了 LABELS 之外的标签：{' '.join(extra)}\n"
            f"  当前 LABELS（{len(LABELS)} 类）：{' '.join(LABELS)}\n"
            f"  如果你用的是老师给的 ar/en/es/... 那套数据，"
            f"把顶部 LABELS 改成那套即可。"
        )
    missing = [l for l in LABELS if l not in seen]
    if missing:
        print(f"[警告] 这些 LABELS 在训练集里没出现：{missing}")

    label2id = {l: i for i, l in enumerate(LABELS)}
    num_classes = len(LABELS)

    if args.per_class:
        by_label = {}
        for t, l in train_rows:
            by_label.setdefault(l, []).append(t)
        rnd = random.Random(SEED)
        train_rows = [(t, l) for l, ts in by_label.items()
                      for t in rnd.sample(ts, min(args.per_class, len(ts)))]
        rnd.shuffle(train_rows)

    vocab = load_vocab_file(args.vocab)
    print(f"数据目录：{args.data_dir}")
    print(f"  训练 {len(train_rows)} 条 / 验证 {len(dev_rows)} 条 / 测试 {len(test_rows)} 条")
    print(f"  类别 {num_classes} 个：{' '.join(LABELS)}")
    print(f"  字符表 {len(vocab)} 个字符（来自 {os.path.basename(args.vocab)}）")
    print(f"  随机猜的准确率基准：{1/num_classes:.1%}")

    train_loader = DataLoader(TextDataset(train_rows, vocab, label2id),
                              batch_size=BATCH_SIZE, shuffle=True)
    dev_loader = DataLoader(TextDataset(dev_rows, vocab, label2id), batch_size=128)
    test_loader = DataLoader(TextDataset(test_rows, vocab, label2id), batch_size=128)

    model = LangIDClassifier(len(vocab), num_classes)
    print(f"  模型参数量：{sum(p.numel() for p in model.parameters()):,}\n")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for X, y in train_loader:
            logits = model(X)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        preds, golds = predict(model, dev_loader)
        acc = sum(p == g for p, g in zip(preds, golds)) / len(golds)
        print(f"Epoch {epoch:2d}/{args.epochs}  loss={total/len(train_loader):.4f}  "
              f"dev_acc={acc:.4f}")
    elapsed = time.time() - t0
    print(f"\n训练耗时：{elapsed:.0f} 秒（{args.epochs} 个 epoch，{args.threads} 线程）")

    # 测试集
    preds, golds = predict(model, test_loader)
    acc = sum(p == g for p, g in zip(preds, golds)) / len(golds)
    f1 = macro_f1(preds, golds, num_classes)
    print(f"\n测试集：accuracy = {acc:.4f}   macro-F1 = {f1:.4f}")
    print_confusion(confusion(preds, golds, LABELS), LABELS, top=5)

    # 逐类准确率（12 行）
    print("\n  各语种准确率：")
    for c, lab in enumerate(LABELS):
        idx = [i for i, g in enumerate(golds) if g == c]
        hit = sum(preds[i] == c for i in idx)
        name = LANG_NAME.get(lab, lab)
        print(f"    {lab:3s} ({name}): {hit}/{len(idx)} = {hit/max(1,len(idx)):.3f}")

    # 推理示例：每个语种来一条，方便课堂上逐语种看
    print("\n--- 推理示例（每语种 1 条）---")
    model.eval()
    seen_lang, demos = set(), []
    for t, l in test_rows:
        if l not in seen_lang:
            seen_lang.add(l)
            demos.append((t, l))
        if len(demos) == num_classes:
            break
    with torch.no_grad():
        for text, gold in demos:
            logits = model(torch.tensor([encode(text, vocab)]))
            prob = torch.softmax(logits, dim=1)[0]
            pid = int(prob.argmax())
            mark = "✓" if LABELS[pid] == gold else "✗"
            print(f"  {mark} [{LABELS[pid]} {prob[pid]:.2f}] "
                  f"gold={gold:3s}  {text[:45]}")

    if args.save:
        torch.save({"state_dict": model.state_dict(), "vocab": dict(vocab),
                    "labels": LABELS, "maxlen": MAX_LEN,
                    "embed_dim": EMBED_DIM, "hidden_dim": HIDDEN_DIM,
                    "dropout": DROPOUT}, args.save)
        print(f"\n模型已保存：{args.save}")


if __name__ == "__main__":
    main()