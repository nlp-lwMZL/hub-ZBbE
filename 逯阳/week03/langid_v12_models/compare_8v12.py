# -*- coding: utf-8 -*-
"""8分类 vs 12分类：模型结构/参数量精确对比"""
import sys
import os
import torch

sys.path.insert(0, r"D:\AI\learning\work3\语种识别实战\langid_v12_models")
from train_langid_cls_v12 import LangIDClassifier

# 词表大小（真实文件）
def vocab_size(path):
    n = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.startswith("#") and line.strip():
                n += 1
    return n

v8 = vocab_size(r"D:\AI\learning\work3\语种识别实战\data\langid\vocab\vocab_v8.txt")
v12 = vocab_size(r"D:\AI\learning\work3\语种识别实战\data\langid\vocab\vocab_v12.txt")

print(f"词表大小：8分类={v8}  12分类={v12}  (+{v12-v8})")

for name, vocab, ncls in [("8分类", v8, 8), ("12分类", v12, 12)]:
    m = LangIDClassifier(vocab, ncls)
    total = sum(p.numel() for p in m.parameters())
    emb = m.embedding.weight.numel()
    rnn = sum(p.numel() for p in m.rnn.parameters())
    fc = m.fc.weight.numel() + m.fc.bias.numel()
    print(f"\n{name}：参数量 {total:,}")
    print(f"  Embedding: {emb:>8,}   (词表{vocab}×64)")
    print(f"  LSTM:      {rnn:>8,}")
    print(f"  FC:        {fc:>8,}   (64×{ncls})")
    print(f"  输出类别数: {ncls}")
