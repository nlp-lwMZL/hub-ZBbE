import math
import argparse
import glob
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ─────────────────────────── 数据（与 language_model.py 相同） ───────────────────────────

def load_corpus(pattern="*.txt"):
    texts = []
    for path in glob.glob(pattern):
        with open(path, encoding="utf-8", errors="ignore") as f:
            texts.append(f.read())
    return "".join(texts)


def build_vocab(text, max_vocab=None):
    """按出现频率取前 max_vocab 个字（None = 全部字符，按码位排序）。"""
    from collections import Counter
    counter = Counter(text)
    chars = [c for c, _ in counter.most_common(max_vocab)] if max_vocab else sorted(set(text))
    char2idx = {c: i for i, c in enumerate(chars)}
    idx2char = {i: c for c, i in char2idx.items()}
    return char2idx, idx2char


class CharDataset(Dataset):
    """x = 连续 seq_len 个字，y = 往后错一位（每个位置都在预测下一个字）。"""

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

class TransformerLM(nn.Module):
    """
    完全对齐 language_model.py 的接口：输入 (B, T) 的字符 id，输出 (B, T, vocab_size) 的 logits。

    结构：字嵌入 + 位置嵌入 -> N 层 pre-LN Transformer（因果 mask）-> LayerNorm -> 线性头
    """

    def __init__(self, vocab_size, embed_dim, nhead, ff_dim, num_layers,
                 max_len=512, dropout=0.1, tie_weights=True):
        super().__init__()
        if embed_dim % nhead != 0:
            raise ValueError(
                f"embed_dim({embed_dim}) 必须能被 nhead({nhead}) 整除，"
                f"否则多头没法均分维度"
            )
        self.max_len = max_len
        self.tok_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(max_len, embed_dim)
        self.drop = nn.Dropout(dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,   # 输入按 (B, T, D) 排列，和原脚本一致
            norm_first=True,    # pre-LN：小模型 + 小 batch 时训练比 post-LN 稳得多
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)

        # 权重共享：输入嵌入矩阵直接当输出投影用，省 4000×128 的参数，也是 LM 的常规做法
        if tie_weights:
            self.head.weight = self.tok_emb.weight

        self.apply(self._init_weights)
        self._mask_cache = {}

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def _causal_mask(self, T, device, dtype):
        """上三角为 -inf 的 (T, T) mask：位置 i 只能看到 0..i。"""
        key = (T, str(device), str(dtype))
        mask = self._mask_cache.get(key)
        if mask is None:
            mask = torch.triu(
                torch.full((T, T), float("-inf"), device=device, dtype=dtype),
                diagonal=1,
            )
            self._mask_cache[key] = mask
        return mask

    def forward(self, x):
        B, T = x.shape
        if T > self.max_len:
            raise ValueError(
                f"输入长度 {T} 超过 max_len={self.max_len}，请调大 --max_len 或缩短 --seq_len"
            )
        pos = torch.arange(T, device=x.device).unsqueeze(0)          # (1, T)
        h = self.drop(self.tok_emb(x) + self.pos_emb(pos))          # (B, T, D)
        h = self.encoder(h, mask=self._causal_mask(T, x.device, h.dtype))
        h = self.norm(h)
        return self.head(self.drop(h))                               # (B, T, V)


# ─────────────────────────── 训练 / 评估 ───────────────────────────

def build_scheduler(optimizer, total_steps, warmup_steps):
    """线性 warmup + 余弦衰减；最后保留 5% 学习率，避免后期完全学不动。"""
    warmup_steps = max(1, min(warmup_steps, total_steps))

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.05, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def run_epoch(model, loader, criterion, optimizer, device,
              train=True, scheduler=None, clip=1.0):
    model.train(train)
    total_loss = 0.0
    total_tokens = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if clip and clip > 0:
                # Transformer 偶尔会炸梯度，裁剪是标配
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()

    if total_tokens == 0:                      # 空 loader（验证集太小）不要除零
        return float("inf"), float("inf")

    avg_loss = total_loss / total_tokens
    ppl = math.exp(min(avg_loss, 50.0))        # 防止 loss 异常大时 exp 溢出
    return avg_loss, ppl


# ─────────────────────────── 主函数 ───────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=20)
    parser.add_argument("--seq_len",    type=int,   default=64)
    parser.add_argument("--stride",     type=int,   default=1, help="切块间隔：1=密集重叠，seq_len=不重叠")
    parser.add_argument("--batch_size", type=int,   default=128)
    # Transformer 专属超参
    parser.add_argument("--embed_dim",  type=int,   default=128, help="d_model，必须能被 nhead 整除")
    parser.add_argument("--nhead",      type=int,   default=8)
    parser.add_argument("--ff_dim",     type=int,   default=0,   help="前馈层维度，0 = 4 * embed_dim")
    parser.add_argument("--num_layers", type=int,   default=2)
    parser.add_argument("--max_len",    type=int,   default=512, help="位置编码支持的最大长度")
    parser.add_argument("--no_tie",     action="store_true", help="关闭输入/输出嵌入权重共享")
    # 训练
    parser.add_argument("--dropout",    type=float, default=0.1)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--clip",       type=float, default=1.0)
    parser.add_argument("--warmup",     type=int,   default=200, help="warmup 步数")
    parser.add_argument("--val_ratio",  type=float, default=0.05)
    parser.add_argument("--max_vocab",  type=int,   default=4000, help="字表最大字数，0=不限")
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--corpus",     default="*.txt")
    parser.add_argument("--save",       default="best_model_transformer.pt")
    args = parser.parse_args()

    if args.seq_len > args.max_len:
        print(f"[提示] seq_len({args.seq_len}) > max_len({args.max_len})，自动把 max_len 提到 seq_len")
        args.max_len = args.seq_len
    if args.ff_dim <= 0:
        args.ff_dim = 4 * args.embed_dim

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  arch: TRANSFORMER")

    # 数据准备
    text = load_corpus(args.corpus)
    if not text:
        raise FileNotFoundError("未找到任何 .txt 文件，请确认路径正确。")
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
    print(f"训练样本数: {len(train_ds):,}  验证样本数: {len(val_ds):,}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    # 验证集不丢尾，避免"验证集比 batch 还小"时 0 个 batch
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, drop_last=False)

    # 模型
    model = TransformerLM(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        nhead=args.nhead,
        ff_dim=args.ff_dim,
        num_layers=args.num_layers,
        max_len=args.max_len,
        dropout=args.dropout,
        tie_weights=not args.no_tie,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}  (embed={args.embed_dim}, nhead={args.nhead}, "
          f"layers={args.num_layers}, ff={args.ff_dim}, max_len={args.max_len})")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = build_scheduler(optimizer, total_steps, args.warmup)
    print(f"每轮步数: {len(train_loader):,}  总步数: {total_steps:,}")

    best_val_ppl = float("inf")

    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Train PPL':>10}  {'Val Loss':>10}  {'Val PPL':>10}")
    print("-" * 56)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_ppl = run_epoch(
            model, train_loader, criterion, optimizer, device,
            train=True, scheduler=scheduler, clip=args.clip,
        )
        with torch.no_grad():
            va_loss, va_ppl = run_epoch(
                model, val_loader, criterion, optimizer, device,
                train=False, scheduler=None, clip=args.clip,
            )

        marker = "  *" if va_ppl < best_val_ppl else ""
        if va_ppl < best_val_ppl:
            best_val_ppl = va_ppl
            torch.save({
                "model_state": model.state_dict(),
                "char2idx": char2idx,
                "idx2char": idx2char,
                "args": {**vars(args), "arch": "transformer", "model": "transformer"},
            }, args.save)

        print(f"{epoch:>6}  {tr_loss:>10.4f}  {tr_ppl:>10.2f}  {va_loss:>10.4f}  {va_ppl:>10.2f}{marker}")

    print(f"\n训练完成。最佳验证 PPL: {best_val_ppl:.2f}  已保存至 {args.save}")


if __name__ == "__main__":
    main()
