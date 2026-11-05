# -*- coding: utf-8 -*-
"""生成 v12 的字符表 vocab_v12.txt（文档要求：学生自己从 v12/train.csv 生成）"""
import csv
import os
import sys
from collections import Counter, OrderedDict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_langid_cls_v12 import normalize  # 复用 12 分类训练脚本的归一化，保证一致

HERE = os.path.dirname(os.path.abspath(__file__))
# 数据目录用绝对路径（脚本已移到 data/langid_v12_models/）
BASE_DATA = r"D:\AI\learning\work3\语种识别实战\data\langid"
TRAIN_CSV = os.path.join(BASE_DATA, "v12", "train.csv")
OUT = os.path.join(BASE_DATA, "vocab", "vocab_v12.txt")

# 1. 统计每个字符出现次数（和 build_langid_dataset.build_char_vocab 逻辑一致）
freq = Counter()
with open(TRAIN_CSV, encoding="utf-8") as f:
    for r in csv.DictReader(f):
        freq.update(normalize(r["text"]))

# 2. 按出现次数降序、次数相同按字符编码升序
chars = sorted(((c, n) for c, n in freq.items()), key=lambda kv: (-kv[1], kv[0]))

# 3. 0 号 <PAD>、1 号 <UNK>，其余按序编号
vocab = OrderedDict([("<PAD>", 0), ("<UNK>", 1)])
for ch, _ in chars:
    vocab[ch] = len(vocab)

# 4. 写文件（格式与 vocab_v8.txt 完全一致）
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    f.write("# 字符\tid\t出现次数\n")
    for ch, idx in vocab.items():
        f.write(f"{ch}\t{idx}\t{freq.get(ch, 0)}\n")

print(f"v12 字符表：{len(vocab)} 个字符 → {OUT}")
