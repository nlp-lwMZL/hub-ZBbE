"""dataset.py — SFT 数据准备：模板编码、字表扩展、loss mask、padding

一条样本的模板（课件 slide 27）：
    <|user|> instruction <|assistant|> output <|end|>

x / y 的构造（错一位，同 week4 口径）：
    ids = [u] + instruction + [a] + output + [e]
    x   = ids[:-1]   模型输入
    y   = ids[1:]    每个位置的目标 = 下一个 token

loss mask（课件 slide 28，Q4 共识）：
    prompt 段（<|user|> + instruction + <|assistant|>）在 y 里对应的位置全部 -100；
    output + <|end|> 参与损失。x 的末位（output 最后一个字）预测 <|end|>——
    模型靠这个位置学会「答完就停」。

两个 mask 各管各的（不要混淆）：
    - causal mask（下三角，model.py）：注意力层面屏蔽「未来」
    - loss mask（-100，本文件）：标签层面屏蔽「问题」
"""
import json
import os
import re

import torch
from torch.utils.data import Dataset

HERE = os.path.dirname(os.path.abspath(__file__))
WEEK4_CKPT = os.path.join(HERE, "..", "week4", "tf_mask.pt")
DEFAULT_TRAIN = os.path.join(HERE, "data", "sft_train.jsonl")
DEFAULT_VAL = os.path.join(HERE, "data", "sft_val.jsonl")

SPECIAL_TOKENS = ["<|user|>", "<|assistant|>", "<|end|>", "<pad>"]
# 编码时把文本切成「特殊标记 | 普通文本」两类；<pad> 只用于填充，不出现在文本里
_SPLIT_RE = re.compile("(" + "|".join(re.escape(t) for t in SPECIAL_TOKENS[:3]) + ")")


def build_vocab(train_path=DEFAULT_TRAIN, val_path=DEFAULT_VAL,
                week4_ckpt=WEEK4_CKPT):
    """
    扩展字表。从 tf_mask.pt 的 char2idx（4000 字）出发——预训练 embedding 的行号
    与它对齐，所以扩展只能在末尾追加，不能改动前 4000 个 id：
        [0..3999]   week4 字表（tf_mask.pt 的 embedding 直接对应）
        [4000..]    SFT 数据里出现、但 week4 字表没有的字符（新行 std=0.02 初始化）
        [最后 4 个]  <|user|> <|assistant|> <|end|> <pad> 特殊 token

    从头训（--init_from none）也用同一份字表，保证两次实验可比。
    """
    ck = torch.load(week4_ckpt, map_location="cpu")
    char2idx = dict(ck["char2idx"])
    n_week4 = len(char2idx)

    texts = []
    for path in (train_path, val_path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    d = json.loads(line)
                    texts.append(d["instruction"])
                    texts.append(d["output"])
    for text in texts:
        for ch in text:
            if ch not in char2idx:
                char2idx[ch] = len(char2idx)
    n_added = len(char2idx) - n_week4

    special_ids = {}
    for tok in SPECIAL_TOKENS:
        special_ids[tok] = len(char2idx)
        char2idx[tok] = len(char2idx)

    idx2char = {i: c for c, i in char2idx.items()}
    print(f"字表: week4 {n_week4} 字 + 补缺 {n_added} 字 + 特殊 token {len(SPECIAL_TOKENS)} "
          f"= {len(char2idx)}")
    return char2idx, idx2char, special_ids


def encode_text(text, char2idx):
    """文本 → id 序列。特殊标记整串匹配成 1 个 id；普通字符逐字查表，字表外跳过。"""
    ids = []
    for seg in _SPLIT_RE.split(text):
        if not seg:
            continue
        if seg in char2idx and seg in SPECIAL_TOKENS:
            ids.append(char2idx[seg])
        else:
            ids.extend(char2idx[c] for c in seg if c in char2idx)
    return ids


class SFTDataset(Dataset):
    """每条样本返回 (x, y)，均已 pad 到 max_len（long 张量）。"""

    def __init__(self, path, char2idx, special_ids, max_len=128):
        self.char2idx = char2idx
        self.u = special_ids["<|user|>"]
        self.a = special_ids["<|assistant|>"]
        self.e = special_ids["<|end|>"]
        self.pad = special_ids["<pad>"]
        self.max_len = max_len

        self.samples = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                prompt = [self.u] + encode_text(d["instruction"], char2idx) + [self.a]
                answer = encode_text(d["output"], char2idx) + [self.e]
                # 截断：总长（x+y 共享的 ids）最多 max_len+1。
                # 只截 output 尾部，<|end|> 必须保留在末位（否则模型学不会停）。
                budget = self.max_len + 1 - len(prompt)
                if len(answer) > budget:
                    answer = answer[: budget - 1] + [self.e]
                ids = prompt + answer
                self.samples.append((prompt, ids))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        prompt, ids = self.samples[i]
        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor(ids[1:], dtype=torch.long)
        # loss mask：y 里预测 prompt 内容（instruction 内部 + <|assistant|>）的位置全部 -100
        y[: len(prompt) - 1] = -100
        # padding：x 补 <pad>，y 补 -100（pad 位置绝不参与 loss）
        if len(x) < self.max_len:
            x = torch.cat([x, torch.full((self.max_len - len(x),), self.pad, dtype=torch.long)])
            y = torch.cat([y, torch.full((self.max_len - len(y),), -100, dtype=torch.long)])
        return x, y


if __name__ == "__main__":
    # 自检：打印一条样本的 x/y/token 对齐情况
    char2idx, idx2char, special_ids = build_vocab()
    ds = SFTDataset(DEFAULT_VAL, char2idx, special_ids)
    x, y = ds[0]
    print(f"样本数 {len(ds)}  x 形状 {tuple(x.shape)}  y 形状 {tuple(y.shape)}")
    print(f"{'位置':>4} {'x(输入)':<14} {'y(目标)':<14} 说明")
    for t in range(10):
        yv = int(y[t])
        ytxt = "-100(不算loss)" if yv == -100 else idx2char[yv]
        print(f"{t:>4} {idx2char[int(x[t])]:<14} {ytxt:<14}")
    print("...（prompt 段 y 全为 -100，output 段开始计算 loss）")
    valid = [t for t in range(len(y)) if int(y[t]) != -100]
    first, last = valid[0], valid[-1]
    print(f"y 参与 loss 的位置: [{first}, {last}]，共 {len(valid)} 个")
    print(f"y[{first}] 起的目标序列: {''.join(idx2char[int(y[t])] for t in valid[:14])} ...")
    print(f"y 的最后目标: {idx2char[int(y[last])]}（<|end|>，模型学会停的关键）")
    print(f"x[{first}] = {idx2char[int(x[first])]}（<|assistant|>，它在位置 {first} 预测 y[{first}]={idx2char[int(y[first])]}，即回答的第一个字）")
    pad_pos = int((x == special_ids["<pad>"]).nonzero()[0])
    print(f"padding 从位置 {pad_pos} 开始，x 为 <pad>、y 为 -100，均不参与前向有效计算与 loss")
