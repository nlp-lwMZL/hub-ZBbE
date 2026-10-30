# -*- coding: utf-8 -*-
"""
12语种分类 · （保证完整训练不中断）
=========================================

    ① 标签清洗     统一小写、去空格，消除"写法不一致"造成的假性未知标签
    ② 动态对齐     LABELS ∪ 数据里出现的标签 = 最终类别表
                   - 数据里有的、LABELS 里没有 → 自动追加为新类别（并在沙盒数据里预警）
                   - LABELS 里有的、数据里没有   → 保留输出格子（不影响训练，只是这一格没样本）
    ③ 未知标签     dev/test 里出现训练集没见过的标签 → 跳过该样本并计入日志，不中断

    默认策略是"自动扩展类别"（--on-unknown add）。
    如果你确实只想做严格的 12 分类，用 --on-unknown skip 把多出来的语种样本丢掉，
    训练同样不会中断。
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
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ============================================================
# 超参数
# ============================================================
SEED = 42
MAX_LEN = 40          # 中文中位长度约 10 字符、德语约 39，统一截到 40
EMBED_DIM = 64
HIDDEN_DIM = 64
DROPOUT = 0.3
BATCH_SIZE = 128
LR = 1e-3
EPOCHS = 10
NUM_THREADS = 1       # 小模型单线程比 16 线程快 2.7 倍（实测），别改成 16

HERE = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(HERE, "data", "langid", "v12")
VOCAB_PATH = os.path.join(HERE, "data", "langid", "vocab", "vocab_v12.txt")

# 基准语种表：模型至少要有这些输出格子
LABELS = ["ar", "en", "es", "hi", "ja", "pt", "ru", "zh", "ko", "th", "vi", "tr"]


# ============================================================
# 1. 文本归一化
# ============================================================
def normalize(text):
    """
    NFKC 归一化 + 转小写 + 只保留「字母(L)」与「组合符号(M)」。

    保留 M 的原因：印地语、泰语的元音是组合标记（कि 里的 ि），
    只留字母会把印地语砍成裸辅音，模型就没法区分。
    """
    t = unicodedata.normalize("NFKC", text).lower()
    return "".join(c for c in t if unicodedata.category(c)[0] in ("L", "M"))


def normalize_label(label):
    """
    【修复点 1】标签清洗。

    数据里常见的不一致：'EN'、'en '、' En'、'en\t'。
    严格比较时这些都会被当成"未知标签"，直接把训练打断。
    统一小写 + 去空白，能消掉绝大部分假性未知。
    """
    return str(label).strip().lower()


# ============================================================
# 2. 字符表
# ============================================================
def build_char_vocab(sentences, min_freq=1):
    """
    从训练集统计字符表（加语种/换数据时用它重新生成）。
    第 0 号 <PAD> 补齐、第 1 号 <UNK> 未见字符，其余按词频降序。
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


def save_vocab_file(vocab, path, freq=None):
    """把字符表写回文件（格式：字符 \\t id \\t 出现次数）。"""
    freq = freq or {}
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# 由 build_char_vocab 自动生成\n")
        f.write("# char\tid\tfreq\n")
        for ch, idx in vocab.items():
            f.write(f"{ch}\t{idx}\t{freq.get(ch, 0)}\n")


def load_vocab_file(path):
    """
    【修复点 2】原来这个函数只解析、从不写回 vocab，
    最后 return 的是一个空 OrderedDict —— 会导致词表大小=0，
    模型 Embedding 维度为 0，训练直接崩。

    这里补上 vocab[ch] = int(idx) 这一句。
    """
    vocab = OrderedDict()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            ch, idx = parts[0], parts[1].strip()
            # 跳过表头行（char\tid\tfreq），否则 int("id") 会崩
            if not idx.lstrip("-").isdigit():
                continue
            vocab[ch] = int(idx)          # ← 关键修复
    return vocab


def encode(text, vocab, max_len=MAX_LEN):
    ids = [vocab.get(c, 1) for c in normalize(text)][:max_len]
    ids += [0] * (max_len - len(ids))
    return ids


# ============================================================
# 3. 读取数据
# ============================================================
def load_split(data_dir, name):
    path = os.path.join(data_dir, f"{name}.csv")
    rows = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append((r["text"], normalize_label(r["label"])))
    return rows


class TextDataset(Dataset):
    def __init__(self, rows, vocab, label2id):
        self.X = [encode(t, vocab) for t, _ in rows]
        self.Y = [label2id[l] for _, l in rows]     # 保证 rows 已过滤过未知标签

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return torch.tensor(self.X[i], dtype=torch.long), self.Y[i]


# ============================================================
# 4. 【核心修复】类别表对齐
# ============================================================
def resolve_labels(base_labels, train_rows, dev_rows, test_rows, on_unknown="add"):
    """
    把"写死的 LABELS"和"数据里实际的标签"对齐，返回 (最终类别表, 清洗后的数据)。

    参数 on_unknown：
        add   —— 数据里多出来的语种自动追加为新类别（默认，保证训练跑完）
        skip  —— 多余语种的样本直接丢弃，只训练 base_labels（保证训练跑完）
        error —— 保持原脚本行为，直接报错退出（对照用）
    """
    seen_train = sorted({l for _, l in train_rows})
    extra = [l for l in seen_train if l not in base_labels]

    if extra:
        print("=" * 62)
        print(f"⚠️  训练集里发现 {len(extra)} 个 LABELS 之外的标签：{' '.join(extra)}")
        if on_unknown == "error":
            raise SystemExit(
                f"\n【严格模式】数据里出现了 LABELS 之外的标签：{' '.join(extra)}\n"
                f"  当前 LABELS（{len(base_labels)} 类）：{' '.join(base_labels)}\n"
                f"  模型输出维度 = len(LABELS)，多出来的语种没有对应的输出格子。\n"
                f"  改用 --on-unknown add 或 --on-unknown skip 可继续训练。"
            )
        elif on_unknown == "skip":
            print(f"→ 策略 skip：这些语种的样本将被丢弃，只训练 {len(base_labels)} 类")
            train_rows = [(t, l) for t, l in train_rows if l in base_labels]
            dev_rows = [(t, l) for t, l in dev_rows if l in base_labels]
            test_rows = [(t, l) for t, l in test_rows if l in base_labels]
        elif on_unknown == "add":
            print(f"→ 策略 add：自动扩展类别，输出维度 {len(base_labels)} → "
                  f"{len(base_labels) + len(extra)}")
        else:
            raise SystemExit(f"未知的 --on-unknown 取值：{on_unknown}")
        print("=" * 62)

    # 最终类别表 = 基准 LABELS + 数据里多出来的（保持稳定顺序）
    if on_unknown == "add":
        final_labels = list(base_labels) + [l for l in extra if l not in base_labels]
    else:
        final_labels = list(base_labels)

    # 还要考虑 dev/test 里可能存在、训练集却没有的标签
    # （这种样本无法训练也无法评估，只能跳过并计数）
    known = set(final_labels)
    orphan = 0
    for rows, tag in ((dev_rows, "dev"), (test_rows, "test")):
        bad = {l for _, l in rows if l not in known}
        if bad:
            n = sum(1 for _, l in rows if l not in known)
            print(f"⚠️  {tag} 中有 {n} 条样本的标签不在类别表里：{' '.join(sorted(bad))}"
                  f" → 已跳过")
            orphan += n
    dev_rows = [(t, l) for t, l in dev_rows if l in known]
    test_rows = [(t, l) for t, l in test_rows if l in known]

    return final_labels, train_rows, dev_rows, test_rows, orphan


# ============================================================
# 5. 模型
# ============================================================
class LangIDClassifier(nn.Module):
    """
    Embedding → RNN → 掩码 MaxPool → LayerNorm → Dropout → Linear

    MaxPool 前面的 mask 是为了让补齐位不参与池化：
    补齐位喂进去的是全 0 向量，RNN 照样会往前走，池化时会把
    "读过一串 0" 的状态也算进来。masked_fill 成 -1e9 后取最大值就干净了。
    """

    def __init__(self, vocab_size, num_class, embed_dim=EMBED_DIM,
                 hidden_dim=HIDDEN_DIM, dropout=DROPOUT):
        super(LangIDClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_class)

    def forward(self, x):
        mask = (x != 0).unsqueeze(-1)
        h, _ = self.rnn(self.embedding(x))
        h = h.masked_fill(~mask, -1e9)
        pooled = h.max(dim=1)[0]
        pooled = self.dropout(self.norm(pooled))
        return self.fc(pooled)


# ============================================================
# 6. 评测
# ============================================================
def predict(model, loader):
    model.eval()
    preds, golds = [], []
    with torch.no_grad():
        for X, Y in loader:
            preds += model(X).argmax(dim=1).tolist()
            golds += Y.tolist()
    return preds, golds


def macro_f1(preds, golds, num_class):
    """每一类各自算 F1，再对类别取平均（少数类不会被大类淹掉）。"""
    f1s = []
    for c in range(num_class):
        tp = sum(p == c and g == c for p, g in zip(preds, golds))
        fp = sum(p == c and g != c for p, g in zip(preds, golds))
        fn = sum(p != c and g == c for p, g in zip(preds, golds))
        if tp + fp + fn == 0:
            continue
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return sum(f1s) / len(f1s) if f1s else 0.0


def confusion(preds, golds, labels):
    """混淆矩阵：行=真实标签，列=预测标签。"""
    n = len(labels)
    mat = [[0] * n for _ in range(n)]
    for p, g in zip(preds, golds):
        mat[g][p] += 1
    return mat


def print_confusion(mat, labels, top=3):
    """只打印错的格子，按错误数排序。"""
    pairs = [(mat[g][p], labels[g], labels[p])
             for g in range(len(labels))
             for p in range(len(labels)) if g != p]
    pairs.sort(reverse=True)
    print("  最常混淆的组合（真实 → 预测）：")
    for cnt, g, p in pairs[:top]:
        if cnt == 0:
            break
        print(f"   {g} → {p}: {cnt} 条")


# ============================================================
# 7. 训练
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=DATA_DIR)
    ap.add_argument("--vocab", default=VOCAB_PATH,
                    help="字符表文件（默认 vocab_v12.txt）")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--per-class", type=int, default=0,
                    help="只用每类前 N 条训练，演示「数据少了会过拟合」")
    ap.add_argument("--threads", type=int, default=NUM_THREADS)
    ap.add_argument("--save", default="", help="保存模型路径，供 eval_langid.py 使用")
    ap.add_argument("--on-unknown", choices=["add", "skip", "error"], default="add",
                    help="数据里出现 LABELS 之外的标签时怎么办："
                         "add=自动加类别（默认，保证跑完）/ skip=丢掉这些样本 / error=报错退出")
    ap.add_argument("--rebuild-vocab", action="store_true",
                    help="用训练集重新生成字符表并覆盖 --vocab 指定的文件")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    random.seed(SEED)
    torch.manual_seed(SEED)

    # ---------- 读数据 ----------
    train_rows = load_split(args.data_dir, "train")
    dev_rows = load_split(args.data_dir, "dev")
    test_rows = load_split(args.data_dir, "test")

    # ---------- 【核心】类别对齐，保证不中断 ----------
    labels, train_rows, dev_rows, test_rows, orphan = resolve_labels(
        LABELS, train_rows, dev_rows, test_rows, on_unknown=args.on_unknown
    )
    label2id = {l: i for i, l in enumerate(labels)}
    num_classes = len(labels)

    if not train_rows:
        raise SystemExit("❌ 清洗后训练集为空，检查 --on-unknown 策略或数据内容")

    # ---------- per-class 采样 ----------
    if args.per_class:
        by_label = {}
        for t, l in train_rows:
            by_label.setdefault(l, []).append(t)
        rng = random.Random(SEED)
        train_rows = [(t, l) for l, ts in by_label.items()
                      for t in rng.sample(ts, min(args.per_class, len(ts)))]
        rng.shuffle(train_rows)

    # ---------- 字符表 ----------
    if args.rebuild_vocab:
        print("→ 正在用训练集重建字符表 ...")
        vocab = build_char_vocab([t for t, _ in train_rows])
        save_vocab_file(vocab, args.vocab)
        print(f"→ 新字符表已写入 {args.vocab}")
    else:
        vocab = load_vocab_file(args.vocab)

    if len(vocab) < 3:
        raise SystemExit(f"❌ 字符表异常（只有 {len(vocab)} 个），"
                         f"检查 {args.vocab} 是否存在或加 --rebuild-vocab")

    print(f"数据目录：{args.data_dir}")
    print(f"  训练 {len(train_rows)} 条 / 验证 {len(dev_rows)} 条 / 测试 {len(test_rows)} 条")
    if orphan:
        print(f"  跳过的未知标签样本：{orphan} 条")
    print(f"  类别 {num_classes} 个：{' '.join(labels)}")
    print(f"  字符表 {len(vocab)} 个字符（来自 {os.path.basename(args.vocab)}）")
    print(f"  随机猜的准确率基准：{1 / num_classes:.1%}")

    # ---------- DataLoader ----------
    train_loader = DataLoader(TextDataset(train_rows, vocab, label2id),
                              batch_size=BATCH_SIZE, shuffle=True)
    dev_loader = DataLoader(TextDataset(dev_rows, vocab, label2id), batch_size=128)
    test_loader = DataLoader(TextDataset(test_rows, vocab, label2id), batch_size=128)

    # ---------- 模型 ----------
    model = LangIDClassifier(len(vocab), num_classes)
    print(f"  模型参数：{sum(p.numel() for p in model.parameters()):,}\n")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    t0 = time.time()
    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        # ---- 训练 ----
        model.train()
        total = 0.0
        for X, y in train_loader:
            logits = model(X)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()

        # ---- 验证 ----
        preds, golds = predict(model, dev_loader)
        acc = sum(p == g for p, g in zip(preds, golds)) / max(1, len(golds))
        print(f"Epoch {epoch:2d}/{args.epochs}  loss={total / len(train_loader):.4f}  "
              f"dev_acc={acc:.4f}")

    elapsed = time.time() - t0
    print(f"\n训练耗时：{elapsed:.0f} 秒（{args.epochs} 个 epoch，{args.threads} 线程）")

    # ---------- 测试集（训练结束后评一次，避免每轮都刷屏）----------
    preds, golds = predict(model, test_loader)
    acc = sum(p == g for p, g in zip(preds, golds)) / max(1, len(golds))
    f1 = macro_f1(preds, golds, num_classes)
    print(f"\n测试集：accuracy = {acc:.4f}  macro_f1 = {f1:.4f}")
    print_confusion(confusion(preds, golds, labels), labels)

    # ---------- 逐类准确率 ----------
    print("\n  各语种准确率")
    for c, lab in enumerate(labels):
        idx = [i for i, g in enumerate(golds) if g == c]
        if not idx:
            print(f"  {lab}: 0/0 = 无样本")
            continue
        hit = sum(preds[i] == c for i in idx)
        print(f"  {lab}: {hit}/{len(idx)} = {hit / len(idx):.3f}")

    # ---------- 推理示例 ----------
    print("\n  --- 推理示例")
    model.eval()
    demos = [t for t, _ in test_rows[:4]]
    with torch.no_grad():
        for text in demos:
            logits = model(torch.tensor([encode(text, vocab)]))
            prob = torch.softmax(logits, dim=-1)[0]
            pid = int(prob.argmax())
            print(f"  [{labels[pid]}] {prob[pid]:.2f}  {text[:50]}")

    # ---------- 保存 ----------
    if args.save:
        torch.save({
            "state_dict": model.state_dict(),
            "vocab": dict(vocab),
            "labels": labels,          # ← 注意：存的是最终类别表，不是原始 LABELS
            "maxlen": MAX_LEN,
            "embed_dim": EMBED_DIM,
            "hidden_dim": HIDDEN_DIM,
            "dropout": DROPOUT,
        }, args.save)
        print(f"\n模型已保存：{args.save}")


if __name__ == "__main__":
    main()
