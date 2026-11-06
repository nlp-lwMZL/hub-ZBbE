"""
字符级语言模型训练脚本，支持 RNN / LSTM 切换，含 PPL 计算。
用法:
    python language_model.py --model lstm --epochs 20
    python language_model.py --model rnn  --epochs 20
"""

import math
import argparse # argparse：命令行传参（--model lstm这种）
import glob # 批量读取*.txt语料文件
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn import TransformerEncoder, TransformerEncoderLayer


# ─────────────────────────── 数据 ───────────────────────────

def load_corpus(pattern="*.txt"): # 作用：批量读取匹配后缀的 txt 文件，全部拼接成一长串大文本字符串
    texts = []
    for path in glob.glob(pattern): # 找到当前目录所有符合*.txt的文件路径
        with open(path, encoding="utf-8", errors="ignore") as f:
            texts.append(f.read())
    return "".join(texts)


def build_vocab(text, max_vocab=None):
    """
    构建字表：按出现频率取前 max_vocab 个字（None = 全部字符）。
    语料里的生僻字超出字表后会被 CharDataset 过滤掉 —— 字表越小模型越小，
    但覆盖不足会丢信息，这是词表大小的取舍。
    """
    from collections import Counter
    counter = Counter(text)  # Counter(text)：统计每个字出现多少次
    chars = [c for c, _ in counter.most_common(max_vocab)] if max_vocab else sorted(set(text))
    char2idx = {"<PAD>": 0}
    for c in chars:
        if c not in char2idx:
            char2idx[c] = len(char2idx)
    idx2char = {i: c for c, i in char2idx.items()}
    return char2idx, idx2char


class CharDataset(Dataset):
    """
    字符级语言模型的训练样本：输入 x = 连续 seq_len 个字符，目标 y = 往后错一位。

    stride 控制切块间隔：
        stride=1  —— 每个字符位置都当一个样本（样本大量重叠，一期语料能切出上百万个样本）
        stride=seq_len —— 不重叠切块（标准做法，样本数 = 语料长度 / seq_len）
    语料一大，stride=1 会让一轮训练变得极慢，这不是模型慢，而是样本数被放大了 seq_len 倍。
    """

    def __init__(self, text, char2idx, seq_len, stride=1):
        self.seq_len = seq_len
        self.stride = max(1, stride)
        ids = [char2idx[c] for c in text if c in char2idx]
        self.data = torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        return max(0, (len(self.data) - self.seq_len - 1) // self.stride + 1)

    def __getitem__(self, idx):
        start = idx * self.stride
        x = self.data[start: start + self.seq_len]
        y = self.data[start + 1: start + self.seq_len + 1]
        return x, y


# ─────────────────────────── 模型 ───────────────────────────

class LM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, model_type, dropout):
        super().__init__()
        self.model_type = model_type  # 保存模型类型，forward用来分支
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.drop = nn.Dropout(dropout)

        if model_type in ("lstm", "rnn"):
            rnn_cls = nn.LSTM if model_type == "lstm" else nn.RNN
            self.rnn = rnn_cls(
                embed_dim, hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.out_dim = hidden_dim

        elif model_type == "transformer":
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=4,
                dim_feedforward=embed_dim * 2,
                dropout=dropout,
                batch_first=True
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.out_dim = embed_dim
            # 因果mask不在buffer初始化，forward动态生成
            self.causal_mask = None

        self.fc = nn.Linear(self.out_dim, vocab_size)

    def forward(self, x):
        e = self.drop(self.embed(x))
        B, L = x.shape
        pad_mask = (x == 0)  # padding mask

        if self.model_type == "transformer":
            # 生成因果mask
            if self.causal_mask is None or self.causal_mask.size(0) != L:
                self.causal_mask = torch.triu(
                    torch.ones(L, L, device=x.device, dtype=torch.bool),
                    diagonal=1
                )
            out = self.encoder(e, mask=self.causal_mask, src_key_padding_mask=pad_mask)
        else:
            # RNN / LSTM分支
            out, _ = self.rnn(e)

        logits = self.fc(self.drop(out))
        return logits


# ─────────────────────────── 训练 / 评估 ───────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, train=True,grad_clip=1.0):

    total_loss = 0.0
    total_tokens =0
    model.train() if train else model.eval()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        n_valid = (y != 0).sum().item()
        if n_valid == 0:  # ← 核心
            continue
        logits = model(x)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))  # 压扁：[B*L, V] 和 [B*L] 算交叉熵

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    return avg_loss, ppl


# ─────────────────────────── 主函数 ───────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      default="transformer", choices=["rnn", "lstm","transformer"])
    parser.add_argument("--epochs",     type=int,   default=60)
    parser.add_argument("--seq_len",    type=int,   default=64)
    parser.add_argument("--stride",     type=int,   default=64, help="切块间隔：1=密集重叠，seq_len=不重叠")
    parser.add_argument("--batch_size", type=int,   default=128)
    parser.add_argument("--embed_dim",  type=int,   default=128)
    parser.add_argument("--hidden_dim", type=int,   default=256)
    parser.add_argument("--num_layers", type=int,   default=2)
    parser.add_argument("--dropout",    type=float, default=0.3)
    parser.add_argument("--lr",         type=float, default=1e-2)
    parser.add_argument("--val_ratio",  type=float, default=0.1)
    parser.add_argument("--max_vocab",  type=int,   default=4000, help="字表最大字数，0=不限")
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--corpus",     default="*.txt")
    parser.add_argument("--save",       default="best_model_test.pt")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  model: {args.model.upper()}")

    # 数据准备
    text = load_corpus(args.corpus)
    if not text:
        raise FileNotFoundError("未找到任何 .txt 文件，请确认路径正确。")
    print(f"语料字符数: {len(text):,}")

    char2idx, idx2char = build_vocab(text, args.max_vocab or None)
    vocab_size = len(char2idx)
    pad_idx = char2idx.get("<PAD>", 0)
    print(f"词表大小: {vocab_size}（含 <PAD>，id={pad_idx}）")

    lines = [ln for ln in text.splitlines() if ln.strip()]   # ← 加 .strip() # text.splitlines() 作用：把一整个长字符串，按换行符 \n 切分成一行一行的列表。
    random.shuffle(lines)
    split = int(len(lines) * (1 - args.val_ratio)) # 按行划分训练集、验证集
    train_text = "\n".join(lines[:split]) # 训练文本
    val_text   = "\n".join(lines[split:])   # 验证文本

    train_ds = CharDataset(train_text, char2idx, args.seq_len, args.stride) # 训练集
    val_ds   = CharDataset(val_text,   char2idx, args.seq_len, args.stride) # 验证集
    print(f"验证集长度：{len(val_ds)}")
    # 取一条样本看看 x,y
    # xx, yy = val_ds[0]
    # print(xx, yy)
# ataLoader：把 Dataset 包装成可循环、按批次取数据的迭代器，给模型训练 / 验证喂数据。
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, drop_last=False)
# drop_last=True：如果最后一批样本数量不足一个 batch，直接丢掉这一批，避免维度报错
    # 模型
    model = LM(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        model_type=args.model,
        dropout=args.dropout, # 这里末尾逗号，允许
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters()) # 总计参数数量
    print(f"模型参数量: {total_params:,}")

    criterion = nn.CrossEntropyLoss(ignore_index=0) # 交叉商函数
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_ppl = float("inf") # 正无穷大
# 一开始把最优验证集困惑度设成无穷大，保证第一轮的验证 PPL 一定比它小，第一轮就能更新最优值。
    # 作用：记录目前为止最好（最小）的验证 PPL
    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Train PPL':>10}  {'Val Loss':>10}  {'Val PPL':>10}")
    print("-" * 56)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_ppl = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        with torch.no_grad():
            va_loss, va_ppl = run_epoch(model, val_loader, criterion, None, device, train=False)

        marker = "  *" if va_ppl < best_val_ppl else ""
        if va_ppl < best_val_ppl:
            best_val_ppl = va_ppl
            torch.save({
                "model_state": model.state_dict(),
                "char2idx": char2idx,
                "idx2char": idx2char,
                "args": vars(args),
            }, args.save)

        print(f"{epoch:>6}  {tr_loss:>10.4f}  {tr_ppl:>10.2f}  {va_loss:>10.4f}  {va_ppl:>10.2f}{marker}")

    print(f"\n训练完成。最佳验证 PPL: {best_val_ppl:.2f}  已保存至 {args.save}")


if __name__ == "__main__":
    main()
