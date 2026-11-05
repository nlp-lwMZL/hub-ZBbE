"""
字符级语言模型训练脚本，支持 RNN / LSTM 切换，含 PPL 计算。
用法:
    python language_model.py --model lstm --epochs 20
    python language_model.py --model rnn  --epochs 20
"""

import math
import argparse
import glob
import os
import random
import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ─────────────────────────── 数据 ───────────────────────────

# 脚本所在目录：语料/映射表等输入文件优先按这里解析，
# 避免「在别的目录执行 python d:\...\language_model.py」时扫不到同目录的 .txt
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def resolve_input(pattern):
    """
    相对路径的解析顺序：脚本所在目录 → 当前工作目录。
    绝对路径原样返回。glob 本身是相对 cwd 的，这里先兜住最常见的那种错。
    """
    if os.path.isabs(pattern):
        return pattern
    local = os.path.join(BASE_DIR, pattern)
    return local if glob.glob(local) else pattern


def load_corpus(pattern="*.txt"):
    pattern = resolve_input(pattern)
    paths = sorted(glob.glob(pattern))          # 排序保证每次拼出的语料顺序一致
    if not paths:
        raise FileNotFoundError(
            f"未找到匹配 {pattern!r} 的文件。\n"
            f"  脚本目录    : {BASE_DIR}\n"
            f"  当前工作目录: {os.getcwd()}\n"
            f"  用 --corpus 显式指定路径，例如:\n"
            f"    python language_model.py --corpus \"{os.path.join(BASE_DIR, '*.txt')}\""
        )
    texts = []
    for path in paths:
        with open(path, encoding="utf-8", errors="ignore") as f:
            texts.append(f.read())
    print(f"语料文件: {len(paths)} 个  ({', '.join(os.path.basename(p) for p in paths[:3])}"
          f"{' ...' if len(paths) > 3 else ''})")
    return "".join(texts)


def build_vocab(text, max_vocab=None):
    """
    构建字表：按出现频率取前 max_vocab 个字（None = 全部字符）。
    语料里的生僻字超出字表后会被 CharDataset 过滤掉 —— 字表越小模型越小，
    但覆盖不足会丢信息，这是词表大小的取舍。
    """
    from collections import Counter
    counter = Counter(text)
    chars = [c for c, _ in counter.most_common(max_vocab)] if max_vocab else sorted(set(text))
    char2idx = {c: i for i, c in enumerate(chars)}
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
        self.embed = nn.Embedding(vocab_size, embed_dim)
        rnn_cls = nn.LSTM if model_type == "lstm" else nn.RNN
        self.rnn = rnn_cls(
            embed_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        e = self.drop(self.embed(x))
        out, _ = self.rnn(e)
        logits = self.fc(self.drop(out))   # (B, T, V)
        return logits


# ─────────────────────────── 训练 / 评估 ───────────────────────────

def _fmt_sec(sec):
    """秒 → 1:23:45 / 3:20 这样的可读时长。"""
    sec = int(max(0, sec))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def run_epoch(model, loader, criterion, optimizer, device, train=True,
              epoch=0, total_epochs=0, log_every=0, tag=""):
    """
    log_every > 0 时每 N 个 batch 打一次进度（loss / 速度 / 剩余时间）。
    默认 stride=1 时一轮有上万个 batch，不打印进度会让人以为程序卡死了。
    """
    model.train(train)
    total_loss = 0.0
    total_tokens = 0
    n_batches = len(loader)
    t0 = time.time()
    tty = sys.stdout.isatty()

    for i, (x, y) in enumerate(loader, 1):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()

        if log_every and (i % log_every == 0 or i == n_batches):
            elapsed = time.time() - t0
            done = i / max(1, n_batches)
            avg = total_loss / total_tokens
            msg = (f"  {tag} {epoch}/{total_epochs}  {i}/{n_batches} "
                   f"({done * 100:5.1f}%)  loss {avg:.4f}  ppl {math.exp(min(avg, 700)):.1f}  "
                   f"{total_tokens / max(elapsed, 1e-9):,.0f} tok/s  "
                   f"已用 {_fmt_sec(elapsed)}  剩余 {_fmt_sec(elapsed / done - elapsed)}")
            if tty:
                # 单行原地刷新，避免刷屏
                print("\r" + msg.ljust(118), end="", flush=True)
            else:
                print(msg, flush=True)

    if log_every and tty:
        print("\r" + " " * 118 + "\r", end="", flush=True)   # 清掉进度行

    avg_loss = total_loss / max(1, total_tokens)
    ppl = math.exp(min(avg_loss, 700))
    return avg_loss, ppl


# ─────────────────────────── 主函数 ───────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      default="lstm", choices=["rnn", "lstm"])
    parser.add_argument("--epochs",     type=int,   default=20)
    parser.add_argument("--seq_len",    type=int,   default=64)
    parser.add_argument("--stride",     type=int,   default=1, help="切块间隔：1=密集重叠（样本数=语料长度），seq_len=不重叠（快 stride 倍）")
    parser.add_argument("--batch_size", type=int,   default=128)
    parser.add_argument("--log_every",  type=int,   default=200, help="每多少个 batch 打一次进度，0=关闭")
    parser.add_argument("--embed_dim",  type=int,   default=128)
    parser.add_argument("--hidden_dim", type=int,   default=256)
    parser.add_argument("--num_layers", type=int,   default=2)
    parser.add_argument("--dropout",    type=float, default=0.3)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--val_ratio",  type=float, default=0.05)
    parser.add_argument("--max_vocab",  type=int,   default=4000, help="字表最大字数，0=不限")
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--corpus",     default="*.txt")
    parser.add_argument("--save",       default="best_model.pt")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  model: {args.model.upper()}")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True   # 固定输入尺寸，让 cudnn 选最快卷积/RNN 实现

    # 数据准备
    text = load_corpus(args.corpus)
    if not text.strip():
        raise ValueError(f"语料内容为空（匹配 {args.corpus!r} 的文件都读不出字符）。")
    print(f"语料字符数: {len(text):,}")

    char2idx, idx2char = build_vocab(text, args.max_vocab or None)
    vocab_size = len(char2idx)
    print(f"词表大小: {vocab_size}")

    lines = text.splitlines()
    random.shuffle(lines)
    split = int(len(lines) * (1 - args.val_ratio))
    train_text = "\n".join(lines[:split])
    val_text   = "\n".join(lines[split:])

    train_ds = CharDataset(train_text, char2idx, args.seq_len, args.stride)
    val_ds   = CharDataset(val_text,   char2idx, args.seq_len, args.stride)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=True, drop_last=True)

    # 模型
    model = LM(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        model_type=args.model,
        dropout=args.dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    # 开工前先告诉用户这一轮要跑多少步 —— stride=1 时步数会被放大 seq_len 倍
    n_train_batches = len(train_loader)
    print(f"每轮 batch 数: 训练 {n_train_batches:,}  验证 {len(val_loader):,}")
    if args.stride == 1 and n_train_batches > 2000:
        print(f"提示: stride=1 是密集重叠切块，每轮 {n_train_batches:,} 步会跑很久。")
        print(f"      想快 {args.seq_len} 倍可以加 --stride {args.seq_len}"
              f"（不重叠切块，同时把 --epochs 乘 {args.seq_len} 保持总更新步数）。")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_ppl = float("inf")

    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Train PPL':>10}  {'Val Loss':>10}  {'Val PPL':>10}")
    print("-" * 56)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_ppl = run_epoch(model, train_loader, criterion, optimizer, device,
                                    train=True, epoch=epoch, total_epochs=args.epochs,
                                    log_every=args.log_every, tag="train")
        with torch.no_grad():
            va_loss, va_ppl = run_epoch(model, val_loader, criterion, optimizer, device,
                                        train=False, epoch=epoch, total_epochs=args.epochs,
                                        log_every=0, tag="val")

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

    print(f"\n训练完成。最佳验证 PPL: {best_val_ppl:.2f}  已保存至 {args.save}"
          if best_val_ppl < float("inf") else "\n训练完成。没有跑任何 epoch，未保存模型。")


if __name__ == "__main__":
    main()
