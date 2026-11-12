# -*- coding: utf-8 -*-
"""
train_langid_cls.py —— 多语种「语种识别」分类器（综合实战脚本 · 课堂 8 语种版）

任务：一句话 → 它是什么语言（12 分类）
模型：Embedding → RNN → 掩码 MaxPool → LayerNorm → Dropout → Linear → Softmax
损失：CrossEntropyLoss      优化器：Adam
数据：data/langid/v12（课堂 12 语种）
运行：CPU 即可，单线程实测约 35 秒。

    python train_langid_cls.py                       # 课堂 8 语种
    python train_langid_cls.py --save langid_rnn.pt  # 存档给 eval_langid.py 评测

【本课想让学生看到的四件事】
  1. 同一份代码、同一张字符表，换数据就能处理 8 种语言文字
  2. 字符级不需要分词器：中文按字、英文按字母、印地语按字母+元音标记，一视同仁
  3. 组件是一个个加进去的：去掉 LayerNorm / Dropout / RNN 都能现场对比
  4. 错得最有意思的是 es ↔ pt（同一套拉丁字母），不是"模型没学会"

【这份脚本是「8 类版」—— 语种列表和字符表都是写死的】
  · LABELS 直接列出了这 8 个语种，模型的输出维度 = len(LABELS)
  · 字符表不从训练集现算，而是读 data/langid/vocab/vocab_v8.txt（课堂用的那张表）
  要换数据集/加语种时，这两处都得跟着改 —— 下一页的作业就是这个。

【字符表是怎么来的】见 build_char_vocab()：
  把训练集每句话按字拆开 → 统计每个字的出现次数 → 按次数从高到低编号。
  已经生成好的表在 data/langid/vocab/vocab_v8.txt，每行「字符 \\t id \\t 出现次数」。
"""

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
MAX_LEN = 40  # 中文中位长度约 10 字符、德语约 39，统一截到 40
EMBED_DIM = 64  # 每个字符的"向量"长度是 64 维
HIDDEN_DIM = 64  # RNN 内部隐藏状态的维度
DROPOUT = 0.3  # 每轮随机"关掉"30%神经元，防止死记硬背(过拟合)
BATCH_SIZE = 64  # 每次喂 64 条样本给模型
LR = 1e-3  # 学习率（每次更新的步子大小），1e-3 = 0.001
EPOCHS = 12  # 把整个训练集反复看 12 遍
NUM_THREADS = 1  # 小模型单线程比 16 线程快 2.7 倍（实测），别改成 16

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data", "langid", "v12")  # 训练数据所在目录
VOCAB_PATH = os.path.join(
    HERE, "data", "langid", "vocab", "vocab_v12.txt"
)  # 字符表文件

# 12个语种
LABELS = ["ar", "en", "es", "hi", "ja", "pt", "ru", "zh", "th", "tr", "vi", "ko"]


# ─── 1. 文本归一化（必须和建数据时一致）────────────────────────────────
def normalize(text):
    """
    NFKC 归一化 + 转小写 + 只保留「字母(L)」与「组合符号(M)」。

    为什么保留 M：印地语、泰语的元音是组合标记（कि 里的 ि），
    只留字母会把印地语砍成裸辅音，模型就没法区分。
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
    阿拉伯字母、天城文、泰文全都排在同一张表里 —— 这就是"字符级天然多语种"。

    课堂主流程没有直接调用它（字符表是从 vocab_v8.txt 读的）；
    一旦换了训练集或加了语种，就要用它重新生成一张表，
    否则新语种的字符会全部落到 1 号 <UNK>，模型等于睁眼瞎。
    """
    freq = Counter()
    for s in sentences:
        freq.update(normalize(s))
    chars = sorted(
        ((c, n) for c, n in freq.items() if n >= min_freq),
        key=lambda kv: (-kv[1], kv[0]),
    )
    vocab = OrderedDict([("<PAD>", 0), ("<UNK>", 1)])
    for ch, _ in chars:
        vocab[ch] = len(vocab)
    return vocab


def load_vocab_file(path):
    """
    读已经生成好的字符表（每行「字符 \\t id \\t 出现次数」，以 # 开头的是注释）。

    课堂用的就是 data/langid/vocab/vocab_v12.txt，它是从 v12 训练集统计出来的。
    → 换了训练集、加了语种，这张表就得重新生成（用上面的 build_char_vocab）。
    """
    vocab = OrderedDict()
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            ch, idx, _freq = line.rstrip("\n").split("\t")
            vocab[ch] = int(idx)
    return vocab


def encode(text, vocab, maxlen=MAX_LEN):
    # 每个字符查表转编号，再截断到40位
    ids = [vocab.get(ch, 1) for ch in normalize(text)][:maxlen]
    ids += [0] * (maxlen - len(ids))  # 不够40位就用0补齐
    return ids


# ─── 3. 读数据 ─────────────────────────────────────────────────────────
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


# ─── 4. 模型 ───────────────────────────────────────────────────────────
class LangIDClassifier(nn.Module):
    """
    Embedding → RNN → 掩码 MaxPool → LayerNorm → Dropout → Linear

    MaxPool 前面那两行 mask 是为了让补齐位不参与池化：
    补齐位喂进去的是全 0 向量，RNN 照样会往前走，池化时就会把
    "读过一串 0" 的状态也算进来。masked_fill 成 -inf 后取最大值就干净了。
    """

    def __init__(
        self,
        vocab_size,
        num_classes,
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT,
    ):
        super().__init__()
        #  Embedding层  输入字符编号，输出对应的 64 维向量
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # RNN层 按顺序读字符
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        # 归一化，把数据"拉回正常范围"，让训练更稳定
        self.norm = nn.LayerNorm(hidden_dim)
        # dropout层，训练时随机关掉 30% 节点，防止过拟合
        self.dropout = nn.Dropout(dropout)
        # 全连接层，把 64 维压成 12 维（12 个分数）。
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        """
        前向传播
        查表 → RNN 读一遍 → 屏蔽补齐位 → 取最大 → 归一化 → 丢弃 → 线性输出 12 分
        """

        mask = (x != 0).unsqueeze(-1)  # (B, L, 1)  补齐位为 False
        # RNN 返回两个东西——`h`（每个时刻的输出）和最后一个隐藏状态（用 `_` 丢掉不用）
        h, _ = self.rnn(self.embedding(x))  # 先查表→再过RNN，得到每个位置的隐藏状态
        # 把 `mask` 为 False（即补齐位）的地方填成 `-1e9`（负十亿）。因为下一步要取最大值，负无穷大永远不可能是最大，等于"跳过这些位置"
        h = h.masked_fill(~mask, -1e9)  # 把补齐位的值改成负无穷大
        pooled = h.max(dim=1)[0]  # (B, H)  # 沿长度方向取最大值
        pooled = self.dropout(self.norm(pooled))  # 归一化→随机丢弃
        return self.fc(pooled)  # (B, C)  未过 softmax 的 logits


# ─── 5. 评测 ───────────────────────────────────────────────────────────
def predict(model, loader):
    model.eval()  # 切换到"评估模式"(关掉 Dropout 等训练专用行为)
    preds, golds = [], []
    with torch.no_grad():  # 关掉梯度计算，省内存、加快速度(预测不需要反传)
        for X, y in loader:
            preds += model(X).argmax(dim=1).tolist()  # 取12个分数里最大的一格作为预测
            golds += y.tolist()  # 真实标签也记下来
    return preds, golds


def macro_f1(preds, golds, num_classes):
    """每一类各自算 F1，再对类别取平均（少数类不会被大类淹掉）。"""
    f1s = []
    for c in range(num_classes):  # 每个类别单独算# 每个类别单独算
        # TP 真阳性：预测c且真是c# TP 真阳性：预测c且真是c
        tp = sum(p == c and g == c for p, g in zip(preds, golds))
        # FP 假阳性：预测c但实际不是c
        fp = sum(p == c and g != c for p, g in zip(preds, golds))
        # FN 假阴性：实际是c但没预测出来
        fn = sum(p != c and g == c for p, g in zip(preds, golds))
        if tp + fp + fn == 0:
            # 这个类别根本没出现，跳过
            continue
        # 精确率 = 预测对的里面有多少真对
        prec = tp / (tp + fp) if tp + fp else 0.0
        # 召回率=该类的样本里召回了多少
        rec = tp / (tp + fn) if tp + fn else 0.0

        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return sum(f1s) / len(f1s)


def confusion(preds, golds, labels):
    n = len(labels)
    mat = [[0] * n for _ in range(n)]
    for p, g in zip(preds, golds):
        # `g` 真实语种、`p` 预测语种。对角线是"猜对的"，非对角线是"猜错的"。
        mat[g][p] += 1
    return mat


def print_confusion(mat, labels, top=3):
    """只打印错的格子，按错误数排序 —— 课堂上看 zh/ja 那一格就够了。"""
    # 收集所有"错格"
    pairs = [
        (mat[g][p], labels[g], labels[p])
        for g in range(len(labels))
        for p in range(len(labels))
        if g != p
    ]
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
    ap.add_argument("--vocab", default=VOCAB_PATH, help="字符表文件")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument(
        "--per-class",
        type=int,
        default=0,
        help="只用每类前 N 条训练，用来演示「数据少了会过拟合」",
    )
    ap.add_argument("--threads", type=int, default=NUM_THREADS)
    print("保存模型到该路径保存模型到该路径")
    ap.add_argument(
        "--save", default="", help="保存模型到该路径，供 eval_langid.py 使用"
    )

    print("apap==", ap)
    args = ap.parse_args()
    # 设置 CPU 线程数
    torch.set_num_threads(args.threads)
    # 固定 Python 随机种子
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
    label2id = {l: i for i, l in enumerate(LABELS)}
    num_classes = len(LABELS)

    if args.per_class:
        by_label = {}
        for t, l in train_rows:
            by_label.setdefault(l, []).append(t)
        train_rows = [
            (t, l)
            for l, ts in by_label.items()
            for t in random.Random(SEED).sample(ts, min(args.per_class, len(ts)))
        ]
        random.Random(SEED).shuffle(train_rows)

    vocab = load_vocab_file(args.vocab)
    print(f"数据目录：{args.data_dir}")
    print(
        f"  训练 {len(train_rows)} 条 / 验证 {len(dev_rows)} 条 / 测试 {len(test_rows)} 条"
    )
    print(f"  类别 {num_classes} 个：{' '.join(LABELS)}")
    print(f"  字符表 {len(vocab)} 个字符（来自 {os.path.basename(args.vocab)}）")
    print(f"  随机猜的准确率基准：{1/num_classes:.1%}")

    train_loader = DataLoader(
        TextDataset(train_rows, vocab, label2id), batch_size=BATCH_SIZE, shuffle=True
    )
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
        print(
            f"Epoch {epoch:2d}/{args.epochs}  loss={total/len(train_loader):.4f}  "
            f"dev_acc={acc:.4f}"
        )
    elapsed = time.time() - t0
    print(
        f"\n训练耗时：{elapsed:.0f} 秒（{args.epochs} 个 epoch，{args.threads} 线程）"
    )

    # 测试集
    preds, golds = predict(model, test_loader)
    acc = sum(p == g for p, g in zip(preds, golds)) / len(golds)
    f1 = macro_f1(preds, golds, num_classes)
    print(f"\n测试集：accuracy = {acc:.4f}   macro-F1 = {f1:.4f}")
    print_confusion(confusion(preds, golds, LABELS), LABELS)

    # 逐类准确率
    print("\n  各语种准确率：")
    for c, lab in enumerate(LABELS):
        idx = [i for i, g in enumerate(golds) if g == c]
        hit = sum(preds[i] == c for i in idx)
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
        torch.save(
            {
                "state_dict": model.state_dict(),
                "vocab": dict(vocab),
                "labels": LABELS,
                "maxlen": MAX_LEN,
                "embed_dim": EMBED_DIM,
                "hidden_dim": HIDDEN_DIM,
                "dropout": DROPOUT,
            },
            args.save,
        )
        print(f"\n模型已保存：{args.save}")


if __name__ == "__main__":
    main()
