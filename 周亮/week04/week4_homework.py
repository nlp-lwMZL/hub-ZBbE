# 作业：用 PyTorch 的 Transformer 训练一个字符级生成式语言模型，用 PPL（困惑度）判断模型好坏

"""
字符级语言模型（decoder-only Transformer），含因果掩码、PPL 计算与完整训练日志。

用法:
    python week4_homework.py                          # 默认与 language_model.py 对齐
    python week4_homework.py --tag ts_exp1            # 指定运行目录名
    python week4_homework.py --epochs 30 --d_model 256
    python week4_homework.py --no_dedup               # 复现旧脚本的切分口径（含行级重复）

每次运行会在 logs/<tag>/ 下产出:
    train.log     人类可读的训练日志（参数、数据统计、每轮详情）
    metrics.csv   每轮一行的指标，可直接画曲线或拼对比表
    args.json     全部参数 + 环境信息 + 数据统计，供复现
    best.pt       验证集 PPL 最低那一轮的权重

前向形状:
    x       (B, T)             字符 id
    h       (B, T, d_model)    词向量 * sqrt(d) + 位置编码
    Block   (B, T, d_model)    维度全程不变，堆 N 层
    logits  (B, T, V)          每个位置对"下一个字"的打分
"""

import argparse
import csv
import glob
import json
import math
import random
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


DEFAULT_CORPUS = ("/Users/zhouliang/Downloads/"
                  "week4-语言模型/循环神经网络语言模型/corpus_general.txt")
MAX_POS = 1024          # 位置编码表的长度上限


# ─────────────────────────── 数据 ───────────────────────────

def load_corpus(pattern="*.txt"):
    texts = []
    for path in sorted(glob.glob(pattern)):
        with open(path, encoding="utf-8", errors="ignore") as f:
            texts.append(f.read())
    return "".join(texts)


def build_vocab(text, max_vocab=None):
    """
    构建字表：按出现频率取前 max_vocab 个字（None = 全部字符）。
    语料里的生僻字超出字表后会被 CharDataset 过滤掉 —— 字表越小模型越小，
    但覆盖不足会丢信息，这是词表大小的取舍。
    """
    counter = Counter(text)
    chars = [c for c, _ in counter.most_common(max_vocab)] if max_vocab else sorted(set(text))
    char2idx = {c: i for i, c in enumerate(chars)}
    idx2char = {i: c for c, i in char2idx.items()}
    return char2idx, idx2char


class CharDataset(Dataset):
    """
    字符级语言模型的训练样本：输入 x = 连续 seq_len 个字符，目标 y = 往后错一位。

    stride 控制切块间隔：
        stride=1        —— 每个字符位置都当一个样本（样本大量重叠，一期语料能切出上百万个样本）
        stride=seq_len  —— 不重叠切块（标准做法，样本数 = 语料长度 / seq_len）
    语料一大，stride=1 会让一轮训练变得极慢，这不是模型慢，而是样本数被放大了 seq_len 倍。
    Transformer 单步比 RNN 贵，建议就用 stride=seq_len。
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

class PosEncoding(nn.Module):
    """
    正弦位置编码：固定表、无参数，和词向量等维相加。

    注意力本身对位置是置换不变的（打乱输入只会打乱输出），顺序信息全靠这一步注入。
    PE[pos, 2i]   = sin(pos / 10000^(2i/d))
    PE[pos, 2i+1] = cos(pos / 10000^(2i/d))
    """

    def __init__(self, d_model, max_len=MAX_POS):
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError(f"正弦位置编码要求 d_model 为偶数，当前 {d_model}")
        pos = torch.arange(max_len).unsqueeze(1).float()          # (L, 1)
        i = torch.arange(0, d_model, 2).float()                   # (d/2,)
        div = torch.exp(-math.log(10000.0) * i / d_model)         # (d/2,)
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        # persistent=False: 跟着 .to(device) 迁移，但不写进 state_dict
        # (否则改 max_len 后 load_state_dict 会报 size mismatch)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):                                         # (B, T, d) -> (B, T, d)
        return x + self.pe[: x.size(1)]


class Block(nn.Module):
    """
    一个 Transformer block（Pre-Norm）：
        x = x + Attn(LayerNorm(x))      带因果掩码的多头自注意力 + 残差
        x = x + FFN (LayerNorm(x))      逐位置前馈 + 残差

    Pre-Norm 的好处是残差是一条干净的恒等通路，2 层的小模型也稳；
    代价是堆叠末尾必须再补一个 LayerNorm（见 TSLM.norm_f），否则最后一层输出没归一化。
    """

    def __init__(self, d_model, nhead, dim_ff, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        # batch_first=True: 输入输出都是 (B, T, d)，不写的话默认要 (T, B, d)
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask):
        h = self.norm1(x)
        # q = k = v：自注意力。attn_mask 传 (T, T)，-inf 的位置不参与打分
        a, _ = self.self_attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + self.drop(a)
        return x + self.drop(self.ffn(self.norm2(x)))


class TSLM(nn.Module):
    """字符级 decoder-only Transformer 语言模型。"""

    def __init__(self, vocab_size, d_model, nhead, dim_ff, num_layers, dropout, max_len=MAX_POS):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model({d_model}) 必须能被 nhead({nhead}) 整除")

        self.d_model = d_model
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PosEncoding(d_model, max_len)
        self.drop = nn.Dropout(dropout)
        # ModuleList + 列表推导：每次 Block(...) 都是新实例，各自持有独立权重
        self.blocks = nn.ModuleList(
            [Block(d_model, nhead, dim_ff, dropout) for _ in range(num_layers)]
        )
        self.norm_f = nn.LayerNorm(d_model)      # Pre-Norm 的收尾
        self.head = nn.Linear(d_model, vocab_size)

        self._mask = None                        # 因果掩码缓存，按 (T, device) 复用

    def _causal_mask(self, T, device):
        """上三角（不含对角线）为 -inf，第 i 行只能看 0..i，看不到未来。"""
        if self._mask is None or self._mask.size(0) != T or self._mask.device != device:
            self._mask = torch.triu(
                torch.full((T, T), float("-inf"), device=device), diagonal=1
            )
        return self._mask

    def forward(self, x):                        # x: (B, T)
        mask = self._causal_mask(x.size(1), x.device)
        h = self.tok_emb(x) * math.sqrt(self.d_model)   # 论文里 embedding 乘 sqrt(d)
        h = self.drop(self.pos_enc(h))                  # 位置编码只在这里加一次
        for blk in self.blocks:
            h = blk(h, mask)
        return self.head(self.norm_f(h))                # (B, T, V)


# ─────────────────────────── 日志 ───────────────────────────

class RunLogger:
    """同时写终端、train.log 和 metrics.csv，便于事后对比与画图。"""

    FIELDS = ["epoch", "lr", "train_loss", "train_ppl", "val_loss", "val_ppl",
              "grad_norm", "seconds", "is_best"]

    def __init__(self, run_dir):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.run_dir / "train.log"
        self.csv_path = self.run_dir / "metrics.csv"
        self._logf = open(self.log_path, "w", encoding="utf-8")
        self._csvf = open(self.csv_path, "w", newline="", encoding="utf-8")
        self._csv = csv.DictWriter(self._csvf, fieldnames=self.FIELDS)
        self._csv.writeheader()
        self._csvf.flush()

    def log(self, msg=""):
        print(msg)
        self._logf.write(msg + "\n")
        self._logf.flush()

    def metric(self, **row):
        self._csv.writerow({k: row.get(k, "") for k in self.FIELDS})
        self._csvf.flush()

    def dump_json(self, name, obj):
        path = self.run_dir / name
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        return path

    def close(self):
        self._logf.close()
        self._csvf.close()


# ─────────────────────────── 训练 / 评估 ───────────────────────────

def run_epoch(model, loader, criterion, device, optimizer=None, clip=0.0):
    """
    返回 (平均 loss, PPL, 平均梯度范数)。
    loss 用 token 数加权累计 —— CrossEntropyLoss 的 mean 是 batch 内平均，
    最后一个 batch 不满时直接平均各 batch 会偏。
    """
    train = optimizer is not None
    model.train(train)
    total_loss = 0.0
    total_tokens = 0
    gn_sum, gn_n = 0.0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            # clip<=0 时传 inf：只统计范数、不做裁剪
            gn = torch.nn.utils.clip_grad_norm_(
                model.parameters(), clip if clip > 0 else float("inf")
            )
            gn_sum += float(gn)
            gn_n += 1
            optimizer.step()

        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()

    if total_tokens == 0:
        return float("nan"), float("nan"), 0.0
    avg_loss = total_loss / total_tokens
    return avg_loss, math.exp(avg_loss), (gn_sum / gn_n if gn_n else 0.0)


# ─────────────────────────── 主函数 ───────────────────────────

def pick_device(choice):
    """auto: 有 CUDA 就用 GPU，否则回落 CPU。"""
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("指定了 --device cuda，但当前环境没有可用 GPU")
        return torch.device("cuda")
    # 注意: Intel Mac 上 torch.backends.mps.is_available() 会误报 True，
    # 实际 .to("mps") 直接抛 "only supported on devices with unified memory"，
    # 所以这里不提供 mps 选项。Apple Silicon 上可自行加一行。
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    p = argparse.ArgumentParser()
    # ── 与 language_model.py 对齐的默认值，便于横向比较 ──
    p.add_argument("--epochs",     type=int,   default=10)
    p.add_argument("--seq_len",    type=int,   default=64)
    p.add_argument("--stride",     type=int,   default=None,
                   help="切块间隔，默认等于 seq_len（不重叠切块）")
    p.add_argument("--batch_size", type=int,   default=128)
    p.add_argument("--d_model",    type=int,   default=128, help="模型维度，整个网络保持不变")
    p.add_argument("--nhead",      type=int,   default=8,   help="注意力头数")
    p.add_argument("--num_layers", type=int,   default=2,   help="block 层数")
    p.add_argument("--ff_dim",     type=int,   default=0,   help="FFN 中间维，0 = 4*d_model")
    p.add_argument("--dropout",    type=float, default=0.3)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--clip",       type=float, default=1.0, help="梯度裁剪阈值，0 = 关闭")
    p.add_argument("--val_ratio",  type=float, default=0.05)
    p.add_argument("--max_vocab",  type=int,   default=4000, help="字表最大字数，0=不限")
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--corpus",     default=DEFAULT_CORPUS)
    # ── 实验管理与数据口径 ──
    p.add_argument("--device",     default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--log_dir",    default="logs", help="日志根目录")
    p.add_argument("--tag",        default=None, help="运行目录名，默认用时间戳")
    p.add_argument("--save",       default=None, help="默认 logs/<tag>/best.pt")
    p.add_argument("--no_dedup",   action="store_true",
                   help="不做行级去重（复现旧脚本口径，验证集会与训练集重叠）")
    p.add_argument("--vocab_from", default=None,
                   help="从某个 checkpoint 读取 char2idx，保证和旧模型的词表逐字一致")
    p.add_argument("--num_workers", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()

    # ── 参数校验与派生 ──
    if args.ff_dim <= 0:
        args.ff_dim = 4 * args.d_model
    if args.stride is None:
        args.stride = args.seq_len
    if args.seq_len > MAX_POS:
        raise SystemExit(f"seq_len({args.seq_len}) 超过位置编码上限 {MAX_POS}")
    if args.d_model % args.nhead != 0:
        raise SystemExit(f"d_model({args.d_model}) 必须能被 nhead({args.nhead}) 整除")

    tag = args.tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.log_dir) / tag
    save_path = Path(args.save) if args.save else run_dir / "best.pt"
    lg = RunLogger(run_dir)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    dev_name = (f"cuda:{torch.cuda.current_device()} ({torch.cuda.get_device_name(0)})"
                if device.type == "cuda" else str(device))

    # ── 数据准备 ──
    text = load_corpus(args.corpus)
    if not text:
        raise FileNotFoundError(f"未读到任何文本，请检查 --corpus: {args.corpus}")

    vocab_src = "本语料现算"
    if args.vocab_from:
        ck = torch.load(args.vocab_from, map_location="cpu", weights_only=False)
        char2idx, idx2char = ck["char2idx"], ck["idx2char"]
        vocab_src = f"对齐 {args.vocab_from}"
    else:
        char2idx, idx2char = build_vocab(text, args.max_vocab or None)
    vocab_size = len(char2idx)

    lines = text.splitlines()
    n_lines_raw = len(lines)
    if not args.no_dedup:
        # 行级去重：这份语料有大量完全重复的问答对，不去重会让验证集 60% 的内容
        # 在训练集里原样出现过，PPL 变成"记忆能力"而不是"泛化能力"
        lines = list(dict.fromkeys(lines))
    n_lines_used = len(lines)

    random.seed(args.seed)
    random.shuffle(lines)
    split = int(n_lines_used * (1 - args.val_ratio))
    train_lines, val_lines = lines[:split], lines[split:]
    train_text, val_text = "\n".join(train_lines), "\n".join(val_lines)

    # 泄漏自检：验证集里有多少行在训练集里一字不差地出现过
    train_set = set(train_lines)
    leaked = sum(1 for l in val_lines if l in train_set)

    train_ds = CharDataset(train_text, char2idx, args.seq_len, args.stride)
    val_ds = CharDataset(val_text, char2idx, args.seq_len, args.stride)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              drop_last=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            drop_last=False, num_workers=args.num_workers)

    # ── 模型 ──
    model = TSLM(
        vocab_size=vocab_size, d_model=args.d_model, nhead=args.nhead,
        dim_ff=args.ff_dim, num_layers=args.num_layers, dropout=args.dropout,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())

    # ── 日志：参数与环境 ──
    lg.log("=" * 66)
    lg.log(f"运行目录  {run_dir.resolve()}")
    lg.log(f"开始时间  {datetime.now():%Y-%m-%d %H:%M:%S}")
    lg.log(f"设备      {dev_name}")
    lg.log(f"torch     {torch.__version__}")
    lg.log("-" * 66)
    lg.log("数据")
    lg.log(f"  语料          {args.corpus}")
    lg.log(f"  字符数        {len(text):,}")
    if args.no_dedup:
        lg.log(f"  行数          {n_lines_raw:,}（未去重）")
    else:
        lg.log(f"  行数          {n_lines_raw:,} -> 去重后 {n_lines_used:,}"
               f"（移除 {n_lines_raw - n_lines_used:,}）")
    lg.log(f"  词表          {vocab_size:,}（{vocab_src}）")
    lg.log(f"  切分          train {len(train_lines):,} 行 / val {len(val_lines):,} 行"
           f"（val_ratio={args.val_ratio}）")
    lg.log(f"  验证集泄漏    {leaked:,} 行与训练集完全重复"
           + ("  [干净]" if leaked == 0 else "  [警告：PPL 会偏乐观]"))
    lg.log(f"  切块          seq_len={args.seq_len}  stride={args.stride}")
    lg.log(f"  样本          train {len(train_ds):,} / val {len(val_ds):,}"
           f"（每轮 {len(train_loader):,} 步）")
    lg.log("模型")
    lg.log(f"  d_model={args.d_model}  nhead={args.nhead}  head_dim={args.d_model // args.nhead}"
           f"  dim_ff={args.ff_dim}  layers={args.num_layers}  dropout={args.dropout}")
    lg.log(f"  参数量        {total_params:,}")
    lg.log("训练")
    lg.log(f"  epochs={args.epochs}  batch={args.batch_size}  lr={args.lr}"
           f"  clip={args.clip}  optimizer=Adam  seed={args.seed}")
    lg.log("=" * 66)

    lg.dump_json("args.json", {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "device": dev_name,
        "torch": torch.__version__,
        "args": vars(args),
        "data": {
            "char_count": len(text),
            "lines_raw": n_lines_raw,
            "lines_used": n_lines_used,
            "dedup": not args.no_dedup,
            "vocab_size": vocab_size,
            "vocab_source": vocab_src,
            "train_lines": len(train_lines),
            "val_lines": len(val_lines),
            "val_leaked_lines": leaked,
            "train_samples": len(train_ds),
            "val_samples": len(val_ds),
            "steps_per_epoch": len(train_loader),
        },
        "model": {
            "params": total_params,
            "d_model": args.d_model,
            "nhead": args.nhead,
            "dim_ff": args.ff_dim,
            "num_layers": args.num_layers,
        },
    })

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_ppl = float("inf")
    best_epoch = 0

    lg.log(f"\n{'Epoch':>5}  {'TrainLoss':>10}  {'TrainPPL':>10}  {'ValLoss':>9}"
           f"  {'ValPPL':>9}  {'GradNorm':>9}  {'Sec':>6}  {'LR':>8}")
    lg.log("-" * 80)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_ppl, grad_norm = run_epoch(
            model, train_loader, criterion, device, optimizer=optimizer, clip=args.clip)
        with torch.no_grad():
            va_loss, va_ppl, _ = run_epoch(model, val_loader, criterion, device)
        secs = time.time() - t0

        is_best = va_ppl < best_val_ppl
        if is_best:
            best_val_ppl, best_epoch = va_ppl, epoch
            torch.save({
                "model_state": model.state_dict(),
                "char2idx": char2idx,
                "idx2char": idx2char,
                "args": vars(args),
                "epoch": epoch,
                "val_ppl": va_ppl,
            }, save_path)

        lg.log(f"{epoch:>5}  {tr_loss:>10.4f}  {tr_ppl:>10.2f}  {va_loss:>9.4f}"
               f"  {va_ppl:>9.2f}  {grad_norm:>9.3f}  {secs:>6.1f}  {args.lr:>8.1e}"
               + ("   *" if is_best else ""))
        lg.metric(epoch=epoch, lr=args.lr, train_loss=round(tr_loss, 6),
                  train_ppl=round(tr_ppl, 4), val_loss=round(va_loss, 6),
                  val_ppl=round(va_ppl, 4), grad_norm=round(grad_norm, 4),
                  seconds=round(secs, 2), is_best=int(is_best))

    lg.log("-" * 80)
    lg.log(f"最佳 epoch {best_epoch}   val_loss={math.log(best_val_ppl):.4f}"
           f"   val_PPL={best_val_ppl:.2f}")
    lg.log(f"权重   {save_path}")
    lg.log(f"日志   {lg.log_path}")
    lg.log(f"指标   {lg.csv_path}")
    lg.log(f"参数   {lg.run_dir / 'args.json'}")
    lg.close()


if __name__ == "__main__":
    main()
