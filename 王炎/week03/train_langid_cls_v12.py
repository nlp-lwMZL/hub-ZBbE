# -*- coding: utf-8 -*-
"""
train_langid_cls_v12.py —— 多语种「语种识别」分类器（作业版 · 12 语种）

任务：一句话 → 它是什么语言（12 分类）
模型：Embedding → RNN → 掩码 MaxPool → LayerNorm → Dropout → Linear → Softmax
损失：CrossEntropyLoss      优化器：Adam
数据：data/langid/v12（作业 12 语种 = 课堂 8 个 + ko / th / vi / tr）
运行：CPU 即可，单线程实测约 10 秒（9600 条 × 8 epoch）。

【实测参考】8 个 epoch 后 test accuracy ≈ 0.958、macro-F1 ≈ 0.959；
  错误几乎全在 es ↔ pt（同一套拉丁字母），新加的 ko / th / vi / tr 都在 0.94 以上。
  12 类比 8 类难的地方不在"学不会"，而在拉丁语族内部越来越挤。

    python train_langid_cls_v12.py                          # 作业 12 语种
    python train_langid_cls_v12.py --save langid_rnn_v12.pt # 存档给 eval_langid.py 评测
    python train_langid_cls_v12.py --rebuild-vocab          # 强制重新统计字符表

【这一版相对 train_langid_cls.py（8 类版）改了什么】
  1. DATA_DIR  : data/langid/v8  → data/langid/v12
  2. VOCAB_PATH: vocab_v8.txt    → vocab_v12.txt
  3. LABELS    : 8 个 → 12 个（多了 ko / th / vi / tr），模型输出维度 = len(LABELS)
  4. 字符表    : v8 那张表是随课件发的；v12 这张表**不随数据发布**（见 DATA_CARD.md），
                 所以这里在第一次运行时用 build_char_vocab() 从 v12/train.csv 现算，
                 算完按同样格式写回 data/langid/vocab/vocab_v12.txt，之后就直接读文件。
  5. 其余超参、模型结构、训练/评测流程与 8 类版**完全一致**，方便和课堂结果对照。

【为什么字符表必须重建】
  vocab_v8.txt 里没有韩文（가）和泰文（ก）的任何一个字符，
  也没有土耳其语/越南语的特殊字母（ı ş ğ、ạ ả ấ）。
  如果沿用 8 类那张表，这些字全都会落到 1 号 <UNK>，
  韩语、泰语这类"整句都是新字符"的语种在模型眼里就成了一串 1，彼此无法区分
  —— 不是模型学不会，是输入信息在查表那一步就被抹掉了。
  一张表、12 个语种共用，这正是"字符级天然多语种"的意思：
  韩文按音节、泰文按字母+元音标记、越南文按带声调的拉丁字母，一视同仁。

【字符表是怎么来的】见 build_char_vocab()：
  把训练集每句话按字拆开 → 统计每个字的出现次数 → 按次数从高到低编号。
  已经生成好的表在 data/langid/vocab/ 下，每行「字符 \\t id \\t 出现次数」。

【和评测脚本对接】模型存档里带了 labels 和 vocab，
  所以 eval_langid.py / demo_mixed_text.py 不用改就能评 12 类模型：
    python eval_langid.py --model langid_rnn_v12.pt --data-dir data/langid/v12
"""

import os
import sys
import csv
import time
import random
import argparse
import unicodedata
from collections import Counter, OrderedDict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ─── 超参数（与 8 类版保持一致，便于对照）──────────────────────────────
SEED = 42
MAX_LEN = 40          # 数据卡规定句子归一化后 8~40 字符，统一截到 40
EMBED_DIM = 64
HIDDEN_DIM = 64
DROPOUT = 0.3
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 20
NUM_THREADS = 1       # 小模型单线程比 16 线程快 2.7 倍（实测），别改成 16

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data", "langid", "v12")
VOCAB_PATH = os.path.join(HERE, "data", "langid", "vocab", "vocab_v12.txt")

# 作业的 12 个语种：模型的输出就是这 12 个格子
LABELS = ["ar", "en", "es", "hi", "ja", "ko", "pt", "ru", "th", "tr", "vi", "zh"]


# ─── 1. 文本归一化（必须和建数据时一致）────────────────────────────────
def normalize(text):
    """
    NFKC 归一化 + 转小写 + 只保留「字母(L)」与「组合符号(M)」。

    为什么保留 M：印地语、泰语的元音是组合标记（कि 里的 ि、กิ 里的 ิ），
    只留字母会把印地语砍成裸辅音、把泰语砍成裸辅音，模型就没法区分。
    """
    t = unicodedata.normalize("NFKC", text).lower()
    return "".join(c for c in t if unicodedata.category(c)[0] in ("L", "M"))


# ─── 2. 字符表构建 ─────────────────────────────────────────────────────
def build_char_vocab(sentences, min_freq=1):
    """
    字符表 = 训练集里所有出现过的字符，每个字符发一个整数编号。

    第 0 号 <PAD>：补齐用（Embedding 里 padding_idx=0，向量恒为 0、不参与训练）
    第 1 号 <UNK>：没见过的字符（字符级下几乎用不到）
    其余按出现次数从高到低编号，次数相同按字符编码排序（保证可复现）

    多语种的关键点：这张表不需要按语种分开建。汉字、拉丁字母、西里尔字母、
    阿拉伯字母、天城文、韩文、泰文全都排在同一张表里 —— 这就是"字符级天然多语种"。

    v12 版：字符表第一次运行时由它现算（见 prepare_vocab），
    因为 vocab_v12.txt 不随数据发布。
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
    """
    读已经生成好的字符表（每行「字符 \\t id \\t 出现次数」，以 # 开头的是注释）。
    """
    vocab = OrderedDict()
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            ch, idx, _freq = line.rstrip("\n").split("\t")
            vocab[ch] = int(idx)
    return vocab


def save_vocab_file(vocab, sentences, path):
    """把现算的字符表按 load_vocab_file 认识的格式写出去（同 build_langid_dataset.py）。"""
    freq = Counter()
    for s in sentences:
        freq.update(normalize(s))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# 字符\tid\t出现次数\n")
        for ch, idx in vocab.items():
            f.write(f"{ch}\t{idx}\t{freq.get(ch, 0)}\n")


def prepare_vocab(train_rows, path, rebuild=False):
    """
    拿到 12 语种要用的字符表：

      · 文件已存在且没加 --rebuild-vocab → 直接读（编号固定，结果可复现）
      · 文件不存在（第一次跑）→ 用 build_char_vocab 从训练集现算，并落盘

    返回 (vocab, 是否是本次新生成的)。
    """
    texts = [t for t, _ in train_rows]
    if os.path.exists(path) and not rebuild:
        return load_vocab_file(path), False
    vocab = build_char_vocab(texts)
    save_vocab_file(vocab, texts, path)
    return vocab, True


def encode(text, vocab, maxlen=MAX_LEN):
    ids = [vocab.get(ch, 1) for ch in normalize(text)][:maxlen]
    ids += [0] * (maxlen - len(ids))
    return ids


# ─── 3. 读数据 ─────────────────────────────────────────────────────────
def console_safe(text):
    """
    Windows 控制台默认是 GBK，直接 print 印地语/泰语/阿拉伯语的原文会抛
    UnicodeEncodeError（脚本跑到最后"推理示例"这一栏直接崩掉）。

    这里把当前控制台编码不了的字换成 ?，只影响显示，不影响训练和预测结果。
    想让所有语种都正常显示：先 `chcp 65001` 再跑脚本。
    """
    enc = sys.stdout.encoding or "utf-8"
    try:
        return text.encode(enc, "replace").decode(enc)
    except (LookupError, UnicodeDecodeError):
        return text


def load_split(data_dir, name):
    path = os.path.join(data_dir, f"{name}.csv")
    rows = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append((r["text"], r["label"]))
    return rows


class TextDataset(Dataset):
    def __init__(self, rows, vocab, label2id):
        self.X = [encode(t, vocab) for t, _ in rows]
        self.y = [label2id[l] for _, l in rows]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return torch.tensor(self.X[i], dtype=torch.long), self.y[i]


# ─── 4. 模型（与 8 类版完全相同，只是输出维度变成 12）──────────────────
class LangIDClassifier(nn.Module):
    """
    Embedding → RNN → 掩码 MaxPool → LayerNorm → Dropout → Linear

    MaxPool 前面那两行 mask 是为了让补齐位不参与池化：
    补齐位喂进去的是全 0 向量，RNN 照样会往前走，池化时就会把
    "读过一串 0" 的状态也算进来。masked_fill 成 -inf 后取最大值就干净了。
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
        mask = (x != 0).unsqueeze(-1)              # (B, L, 1)  补齐位为 False
        h, _ = self.rnn(self.embedding(x))         # (B, L, H)
        h = h.masked_fill(~mask, -1e9)
        pooled = h.max(dim=1)[0]                   # (B, H)
        pooled = self.dropout(self.norm(pooled))
        return self.fc(pooled)                     # (B, C)  未过 softmax 的 logits


# ─── 5. 评测 ───────────────────────────────────────────────────────────
def predict(model, loader):
    model.eval()
    preds, golds = [], []
    with torch.no_grad():
        for X, y in loader:
            preds += model(X).argmax(dim=1).tolist()
            golds += y.tolist()
    return preds, golds


def macro_f1(preds, golds, num_classes):
    """每一类各自算 F1，再对类别取平均（少数类不会被大类淹掉）。"""
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
    """只打印错的格子，按错误数排序 —— 12 类时看得最多的是 es ↔ pt、zh ↔ ja。"""
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
                    help="字符表文件（默认 data/langid/vocab/vocab_v12.txt，不存在时会现算）")
    ap.add_argument("--rebuild-vocab", action="store_true",
                    help="强制从训练集重新统计字符表（改过训练集时用）")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--per-class", type=int, default=0,
                    help="只用每类前 N 条训练，用来演示「数据少了会过拟合」")
    ap.add_argument("--threads", type=int, default=NUM_THREADS)
    ap.add_argument("--save", default="", help="保存模型到该路径，供 eval_langid.py 使用")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    random.seed(SEED)
    torch.manual_seed(SEED)

    train_rows = load_split(args.data_dir, "train")
    dev_rows = load_split(args.data_dir, "dev")
    test_rows = load_split(args.data_dir, "test")

    # 数据里的标签必须和 LABELS 对得上，否则模型的输出格子根本不够用
    seen = sorted({l for _, l in train_rows})
    extra = [l for l in seen if l not in LABELS]
    if extra:
        raise SystemExit(
            f"数据里出现了 LABELS 之外的标签：{' '.join(extra)}\n"
            f"  当前 LABELS（{len(LABELS)} 类）：{' '.join(LABELS)}\n"
            f"  模型输出维度 = len(LABELS)，多出来的语种没有对应的输出格子。"
        )
    missing = [l for l in LABELS if l not in seen]
    if missing:
        print(f"[警告] LABELS 里这些语种在训练集里没有数据：{' '.join(missing)}")
    label2id = {l: i for i, l in enumerate(LABELS)}
    num_classes = len(LABELS)

    # 字符表：先按完整训练集建好（--per-class 只影响训练样本，不该影响字符编号）
    vocab, built_now = prepare_vocab(train_rows, args.vocab, args.rebuild_vocab)

    if args.per_class:
        by_label = {}
        for t, l in train_rows:
            by_label.setdefault(l, []).append(t)
        train_rows = [(t, l) for l, ts in by_label.items()
                      for t in random.Random(SEED).sample(ts, min(args.per_class, len(ts)))]
        random.Random(SEED).shuffle(train_rows)

    print(f"数据目录：{args.data_dir}")
    print(f"  训练 {len(train_rows)} 条 / 验证 {len(dev_rows)} 条 / 测试 {len(test_rows)} 条")
    print(f"  类别 {num_classes} 个：{' '.join(LABELS)}")
    print(f"  字符表 {len(vocab)} 个字符"
          f"（{'本次从训练集新生成' if built_now else '读自'} {os.path.basename(args.vocab)}）")
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
    print_confusion(confusion(preds, golds, LABELS), LABELS)

    # 逐类准确率（12 类，重点看新加的 ko / th / vi / tr 有没有掉队）
    print("\n  各语种准确率：")
    for c, lab in enumerate(LABELS):
        idx = [i for i, g in enumerate(golds) if g == c]
        hit = sum(preds[i] == c for i in idx)
        print(f"    {lab}: {hit}/{len(idx)} = {hit/max(1,len(idx)):.3f}")

    # 推理示例：按语种各取一条，12 个都看一眼
    print("\n--- 推理示例 ---")
    model.eval()
    demo_texts, used = [], set()
    for t, l in test_rows:
        if l in used or l not in label2id:
            continue
        used.add(l)
        demo_texts.append(t)
    with torch.no_grad():
        for text in demo_texts:
            logits = model(torch.tensor([encode(text, vocab)]))
            prob = torch.softmax(logits, dim=1)[0]
            pid = int(prob.argmax())
            print(f"  [{LABELS[pid]} {prob[pid]:.2f}]  {console_safe(text[:50])}")

    if args.save:
        torch.save({"state_dict": model.state_dict(), "vocab": dict(vocab),
                    "labels": LABELS, "maxlen": MAX_LEN,
                    "embed_dim": EMBED_DIM, "hidden_dim": HIDDEN_DIM,
                    "dropout": DROPOUT}, args.save)
        print(f"\n模型已保存：{args.save}")


if __name__ == "__main__":
    main()
