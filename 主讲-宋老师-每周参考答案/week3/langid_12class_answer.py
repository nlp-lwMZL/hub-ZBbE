# -*- coding: utf-8 -*-
"""
langid_12class_answer.py —— 第 3 周「语种识别」进阶选做作业 ① · 参考答案
                            （8 类 → 12 类改造）

【题目】
    课上发的 train_langid_cls.py 是 **8 语种版**，语种列表和字符表都是写死的。
    请把它改造成 **12 语种版**（加上 ko / th / tr / vi），用 data/langid/v12 训练并提交：
      · 改造后的脚本
      · 测试集 accuracy / macro-F1
      · 逐语种准确率 + 混淆矩阵里最常混的组合
      · 一句话：多出来的语种里，哪个最难？为什么？

【要改的三处】（本文件就是改完的样子，可以直接对照）

    ① 语种列表：LABELS 从 8 个补到 12 个
       LABELS = ["ar","en","es","hi","ja","pt","ru","zh"]
       → ["ar","en","es","hi","ja","ko","pt","ru","th","tr","vi","zh"]
       不改的话脚本会直接报错：数据里出现了 LABELS 之外的标签 ko th tr vi

    ② 词表：课件只给了 8 类的 data/langid/vocab/vocab_v8.txt，12 类那张表要自己生成
       课堂代码里的 build_char_vocab() 就是生成器，但课上只演示了"读表"（load_vocab_file）；
       这里补上"写表"的那 5 行 save_vocab_file()，产出 vocab_v12.txt，
       格式和 vocab_v8.txt 完全一样（每行「字符 \\t id \\t 出现次数」），能被课堂脚本直接读。
       不做的话不会报错，但会悄悄掉分：
       训练集里 18.3% 的字符不在 8 类字符表里，全部变成 <UNK>，
       韩文整句变成一串相同的 <UNK>，和泰文撞在一起 —— accuracy 从 0.953 掉到 0.883

    ③ 数据目录：指向 data/langid/v12

    提交内容：改造后的脚本 + vocab_v12.txt + 指标报告（见文件末尾的打印）

【实测结果】（8 个 epoch，CPU 单线程）

    做法                              accuracy   macro-F1   韩文准确率
    沿用 vocab_v8（没换词表）            0.8833     0.8555     0.000（149 条被认成泰语）
    生成 vocab_v12 并用它训练           0.9533     0.9537     0.987

【参考答案的话】
    - 12 类（0.9533）和 8 类（0.9558）基本持平，误差在 run-to-run 波动范围内；
      因为新加的 ko / th / vi 都有自己独特的文字，比 es / pt 好分得多
    - 真正难的一直是 pt → es（36 条），和语种类数没关系
    - 最容易忽略的就是词表：它不报错，只掉分 —— "数据决定模型能看见什么"最直观的一次演示
    - 词表文件是模型的一部分：换数据就必须重新生成，否则等于给模型戴着眼罩

【运行】
    python langid_12class_answer.py                 # 正确做法：生成 12 类词表并用它训练
    python langid_12class_answer.py --stale-vocab   # 反面教材：沿用 8 类词表
    python langid_12class_answer.py --save-vocab-only   # 只生成 vocab_v12.txt，不训练
"""

import os
import sys
import time
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
COURSE_DIR = os.path.join(HERE, "..", "..", "八斗课程", "week3-深度学习常用组件", "语种识别实战")
sys.path.insert(0, os.path.abspath(COURSE_DIR))

from train_langid_cls import (            # noqa: E402  复用课堂脚本，保证口径一致
    LangIDClassifier, build_char_vocab, load_vocab_file, encode, normalize,
    load_split, TextDataset, predict, macro_f1, confusion, print_confusion,
    EPOCHS, BATCH_SIZE, LR, SEED,
)

# ─── 改动 ①：8 类 → 12 类 ─────────────────────────────────────────────
LABELS = ["ar", "en", "es", "hi", "ja", "ko", "pt", "ru", "th", "tr", "vi", "zh"]

# ─── 改动 ③：数据目录指向 v12 ─────────────────────────────────────────
DATA_DIR = os.path.join(COURSE_DIR, "data", "langid", "v12")
STALE_VOCAB = os.path.join(COURSE_DIR, "data", "langid", "vocab", "vocab_v8.txt")
VOCAB_OUT = os.path.join(HERE, "vocab_v12.txt")      # 作业要交的词表就放这儿


def save_vocab_file(vocab, freq, path):
    """
    ← 这 5 行就是课堂没演示的"写表"部分。

    格式必须和课堂的 vocab_v8.txt 一致（每行「字符 \\t id \\t 出现次数」，
    第一行是注释），否则 train_langid_cls.load_vocab_file() 读不了。
    """
    with open(path, "w", encoding="utf-8") as f:
        f.write("# 字符\tid\t出现次数\n")
        for ch, idx in vocab.items():
            f.write(f"{ch}\t{idx}\t{freq.get(ch, 0)}\n")


def build_and_save_vocab(train_rows, path):
    """从 v12 训练集重新统计字符 → 写出 vocab_v12.txt → 再用课堂的函数读回来校验。"""
    from collections import Counter
    freq = Counter()
    for text, _ in train_rows:
        freq.update(normalize(text))
    vocab = build_char_vocab([t for t, _ in train_rows])
    save_vocab_file(vocab, freq, path)

    back = load_vocab_file(path)                       # 用课堂脚本的读法回读
    if list(back.items()) != list(vocab.items()):
        raise SystemExit("回读的字符表和内存里的不一致，格式写错了")
    return back


def unk_ratio(rows, vocab):
    """训练集里有多少字符会落到 <UNK>（1 号）—— 词表没换时这个数会很吓人。"""
    total = unk = 0
    for text, _ in rows:
        for ch in normalize(text):
            total += 1
            if vocab.get(ch, 1) == 1:
                unk += 1
    return unk / max(1, total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--stale-vocab", action="store_true",
                    help="反面教材：不生成新词表，直接沿用 vocab_v8.txt")
    ap.add_argument("--save-vocab-only", action="store_true",
                    help="只生成 vocab_v12.txt，不训练")
    args = ap.parse_args()

    torch.set_num_threads(1)
    torch.manual_seed(SEED)

    train_rows = load_split(DATA_DIR, "train")
    dev_rows = load_split(DATA_DIR, "dev")
    test_rows = load_split(DATA_DIR, "test")

    extra = [l for l in {l for _, l in train_rows} if l not in LABELS]
    if extra:
        raise SystemExit(f"LABELS 还没补全：{' '.join(sorted(extra))}")

    label2id = {l: i for i, l in enumerate(LABELS)}

    # ─── 改动 ②：生成 12 类词表，再用课堂脚本的读法读回来 ─────────────
    if args.stale_vocab:
        vocab = load_vocab_file(STALE_VOCAB)
        source = "沿用 vocab_v8.txt（❌ 反面教材）"
    else:
        vocab = build_and_save_vocab(train_rows, VOCAB_OUT)
        source = f"已生成 {os.path.basename(VOCAB_OUT)} 并回读校验（✅）"

    print(f"类别 {len(LABELS)} 个：{' '.join(LABELS)}")
    print(f"词表 {len(vocab)} 个字符，来源：{source}")
    print(f"训练集字符落到 <UNK> 的比例：{unk_ratio(train_rows, vocab):.1%}\n")

    if args.save_vocab_only:
        print(f"词表已写出：{VOCAB_OUT}")
        return

    train_loader = DataLoader(TextDataset(train_rows, vocab, label2id),
                              batch_size=BATCH_SIZE, shuffle=True)
    dev_loader = DataLoader(TextDataset(dev_rows, vocab, label2id), batch_size=128)
    test_loader = DataLoader(TextDataset(test_rows, vocab, label2id), batch_size=128)

    model = LangIDClassifier(len(vocab), len(LABELS))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for X, y in train_loader:
            loss = criterion(model(X), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        dp, dg = predict(model, dev_loader)
        print(f"Epoch {epoch:2d}/{args.epochs}  loss={total/len(train_loader):.4f}  "
              f"dev_acc={sum(p == g for p, g in zip(dp, dg))/len(dg):.4f}")
    print(f"\n训练耗时：{time.time() - t0:.0f} 秒")

    preds, golds = predict(model, test_loader)
    acc = sum(p == g for p, g in zip(preds, golds)) / len(golds)
    print(f"\n测试集：accuracy = {acc:.4f}   macro-F1 = {macro_f1(preds, golds, len(LABELS)):.4f}")
    print_confusion(confusion(preds, golds, LABELS), LABELS, top=5)

    print("\n  各语种准确率：")
    for c, lab in enumerate(LABELS):
        idx = [i for i, g in enumerate(golds) if g == c]
        hit = sum(preds[i] == c for i in idx)
        line = f"    {lab}: {hit}/{len(idx)} = {hit/max(1, len(idx)):.3f}"
        if hit < 0.8 * len(idx):                      # 掉得厉害的，标出它被认成了谁
            wrong = {}
            for i in idx:
                if preds[i] != c:
                    wrong[LABELS[preds[i]]] = wrong.get(LABELS[preds[i]], 0) + 1
            top = " ".join(f"{k}×{v}" for k, v in
                           sorted(wrong.items(), key=lambda kv: -kv[1])[:3])
            line += f"   ← 被认成：{top}"
        print(line)


if __name__ == "__main__":
    main()
