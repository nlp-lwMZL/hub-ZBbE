"""
train_langid_cls_v12.py —— 多语种「语种识别」分类器（作业 12 语种版）

和课堂版 train_langid_cls.py 相比，只改了三处：

  1. LABELS 从 8 个扩到 12 个：多了 ko / th / vi / tr
     → 模型的输出维度自动变成 len(LABELS) = 12
  2. 字符表不再读现成的 vocab_v8.txt，而是从 v12/train.csv 现算并落盘
     → 新语种的字符不会全部退化成 <UNK>（这是课堂版换数据时最容易踩的坑）
  3. 训练时按 dev_acc 保留最佳模型
     → 课堂版固定跑 8 轮取最后一个；这里避免「dev 早就见顶、测试还用差的权重」

模型、归一化、编码、指标全部从课堂版 import，保证和 eval_langid.py 同一口径：
学生改了 train_langid_cls.py 里的模型结构，这份脚本和评测脚本会一起跟着变。

    python train_langid_cls_v12.py                          # 12 语种，词表缺失时自动生成
    python train_langid_cls_v12.py --rebuild-vocab          # 强制重建 vocab_v12.txt
    python train_langid_cls_v12.py --save langid_rnn_v12.pt # 存档给 eval_langid.py

    python eval_langid.py --model langid_rnn_v12.pt --data-dir data/langid/v12

参考规模：v12/train.csv 9600 条（12 类 × 800），字符表约 3276 个，
前 805 个字符覆盖 95% 的字符出现次数，测试集字符级 UNK 占比约 0.6%。
"""

import argparse
import os
import random
import sys
import time
from collections import Counter, OrderedDict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# 复用课堂版：归一化 / 编码 / 数据集 / 模型 / 评测，避免两套口径漂移
from train_langid_cls import (
    DROPOUT,
    EMBED_DIM,
    HIDDEN_DIM,
    MAX_LEN,
    LangIDClassifier,
    TextDataset,
    confusion,
    encode,
    load_split,
    load_vocab_file,
    macro_f1,
    normalize,
    predict,
    print_confusion,
)

# ─── 超参数（与课堂版保持一致，方便对照实验）──────────────────────────
SEED = 42
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 8
NUM_THREADS = 1

DATA_DIR = os.path.join(HERE, "data", "langid", "v12")
VOCAB_PATH = os.path.join(HERE, "data", "langid", "vocab", "vocab_v12.txt")

# 12 个语种 = 课堂 8 个 + ko / th / vi / tr；模型输出就是这 12 个格子
LABELS = ["ar", "en", "es", "hi", "ja", "ko", "pt", "ru", "th", "tr", "vi", "zh"]

# 测试集 UNK 超过这个比例就告警：说明词表和当前数据对不上
UNK_WARN = 0.03


# ─── 1. 字符表：按训练集现算并写成「字符 \\t id \\t 频次」────────────────
def build_char_vocab(sentences, min_freq=1):
    """
    与课堂版 build_char_vocab 同一规则，只是额外返回 freq 用于写文件。

    第 0 号 <PAD>、第 1 号 <UNK>，其余按出现次数从高到低编号；
    次数相同时按字符编码排序，保证多次运行结果一致。
    """
    freq = Counter()
    for s in sentences:
        freq.update(normalize(s))
    chars = sorted(((c, n) for c, n in freq.items() if n >= min_freq), key=lambda kv: (-kv[1], kv[0]))
    vocab = OrderedDict([("<PAD>", 0), ("<UNK>", 1)])
    for ch, _ in chars:
        vocab[ch] = len(vocab)
    return vocab, freq


def save_vocab(vocab, freq, path):
    """写成和 vocab_v8.txt 完全相同的格式：字符 \t id \t 出现次数。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# 字符\tid\t出现次数\n")
        for ch, idx in vocab.items():
            f.write(f"{ch}\t{idx}\t{freq.get(ch, 0)}\n")


def unk_rate(texts, vocab):
    """按字符统计未登录比例（词表和数据的匹配程度）。"""
    total = hit = 0
    for t in texts:
        for ch in normalize(t):
            total += 1
            if ch not in vocab:
                hit += 1
    return hit / max(1, total)


# ─── 2. 训练 ───────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DATA_DIR, help="默认 v12 数据集")
    ap.add_argument("--vocab", default=VOCAB_PATH, help="字符表路径；不存在会自动生成")
    ap.add_argument("--rebuild-vocab", action="store_true", help="忽略已有文件，强制从训练集重新生成字符表")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--per-class", type=int, default=0, help="只用每类前 N 条训练，用来对照「数据少了会怎样」")
    ap.add_argument("--threads", type=int, default=NUM_THREADS)
    ap.add_argument("--save", default="", help="保存最佳模型到该路径，供 eval_langid.py 使用")
    args = ap.parse_args()

    if args.epochs < 1:
        raise SystemExit("--epochs 至少为 1")

    torch.set_num_threads(args.threads)
    random.seed(SEED)
    torch.manual_seed(SEED)

    train_rows = load_split(args.data_dir, "train")
    dev_rows = load_split(args.data_dir, "dev")
    test_rows = load_split(args.data_dir, "test")

    # 标签必须和 LABELS 对得上，否则模型的输出格子不够用
    seen = sorted({l for _, l in train_rows})
    extra = [l for l in seen if l not in LABELS]
    if extra:
        raise SystemExit(
            f"数据里出现了 LABELS 之外的标签：{' '.join(extra)}\n  当前 LABELS（{len(LABELS)} 类）：{' '.join(LABELS)}"
        )
    label2id = {l: i for i, l in enumerate(LABELS)}
    num_classes = len(LABELS)

    # 字符表：v12 不提供现成文件，默认从当前训练集现算一张，并写成课堂版同样的格式
    if args.rebuild_vocab or not os.path.exists(args.vocab):
        vocab, freq = build_char_vocab([t for t, _ in train_rows])
        save_vocab(vocab, freq, args.vocab)
        print(f"已生成字符表 {len(vocab)} 个字符 → {args.vocab}")
    else:
        vocab = load_vocab_file(args.vocab)

    print(f"\n数据目录：{args.data_dir}")
    print(f"  训练 {len(train_rows)} 条 / 验证 {len(dev_rows)} 条 / 测试 {len(test_rows)} 条")
    print(f"  类别 {num_classes} 个：{' '.join(LABELS)}")
    print(f"  字符表 {len(vocab)} 个字符（{os.path.basename(args.vocab)}）")
    print(f"  随机猜的准确率基准：{1 / num_classes:.1%}")

    # 词表覆盖检查：换数据后最隐蔽的退化成 <UNK> 问题，这里直接暴露
    train_unk = unk_rate([t for t, _ in train_rows], vocab)
    test_unk = unk_rate([t for t, _ in test_rows], vocab)
    print(f"  UNK 占比：训练集 {train_unk:.2%} / 测试集 {test_unk:.2%}")
    if test_unk > UNK_WARN:
        print(f"  [警告] 测试集 UNK 超过 {UNK_WARN:.0%}：词表和这份数据不匹配，请用 --rebuild-vocab 重新生成")

    # 每类采样（放在建表之后，保证词表不随 --per-class 变化）
    if args.per_class:
        by_label = {}
        for t, l in train_rows:
            by_label.setdefault(l, []).append(t)
        train_rows = [
            (t, l) for l, ts in by_label.items() for t in random.Random(SEED).sample(ts, min(args.per_class, len(ts)))
        ]
        random.Random(SEED).shuffle(train_rows)
        print(f"  --per-class 生效：训练集缩减为 {len(train_rows)} 条")

    train_loader = DataLoader(TextDataset(train_rows, vocab, label2id), batch_size=BATCH_SIZE, shuffle=True)
    dev_loader = DataLoader(TextDataset(dev_rows, vocab, label2id), batch_size=128)
    test_loader = DataLoader(TextDataset(test_rows, vocab, label2id), batch_size=128)

    model = LangIDClassifier(len(vocab), num_classes)
    print(f"  模型参数量：{sum(p.numel() for p in model.parameters()):,}\n")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # 按 dev_acc 选模型：课堂版是固定轮数取最后一个，见文档第 3 条差异
    best_acc, best_epoch, best_state = -1.0, 0, None
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for X, y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
            total += loss.item()

        preds, golds = predict(model, dev_loader)
        acc = sum(p == g for p, g in zip(preds, golds)) / len(golds)
        flag = ""
        if acc > best_acc:
            best_acc, best_epoch = acc, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            flag = "  ← best"
        print(f"Epoch {epoch:2d}/{args.epochs}  loss={total / len(train_loader):.4f}  dev_acc={acc:.4f}{flag}")

    elapsed = time.time() - t0
    print(f"\n训练耗时：{elapsed:.0f} 秒（{args.epochs} 个 epoch，{args.threads} 线程）")

    model.load_state_dict(best_state)
    print(f"按 dev_acc 选中 Epoch {best_epoch}（dev_acc={best_acc:.4f}），用它评测测试集")

    # 测试集
    preds, golds = predict(model, test_loader)
    acc = sum(p == g for p, g in zip(preds, golds)) / len(golds)
    f1 = macro_f1(preds, golds, num_classes)
    print(f"\n测试集：accuracy = {acc:.4f}   macro-F1 = {f1:.4f}")
    print_confusion(confusion(preds, golds, LABELS), LABELS, top=5)

    print("\n  各语种准确率：")
    for c, lab in enumerate(LABELS):
        idx = [i for i, g in enumerate(golds) if g == c]
        if not idx:
            continue
        hit = sum(preds[i] == c for i in idx)
        print(f"    {lab}: {hit}/{len(idx)} = {hit / len(idx):.3f}")

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
        print(f"  评测：python eval_langid.py --model {args.save} --data-dir {args.data_dir}")


if __name__ == "__main__":
    main()
