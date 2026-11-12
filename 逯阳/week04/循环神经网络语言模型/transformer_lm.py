"""
字符级语言模型训练脚本（Transformer 版本），支持开关 causal 掩码，含 PPL 计算。

与 language_model.py 结构一一对应，只把 RNN/LSTM 换成 Transformer Encoder：
    Embedding + 位置编码 → N × Self-Attention 块 → LayerNorm → Linear(vocab)

causal 掩码开关（对应参数 --causal）：
    --causal 1  标准自回归语言模型：每个位置只能看见自己和左边的字符（默认）
    --causal 0  双向注意力（BERT 风格）：每个位置能看到整条序列

    注意 --causal 0 在「预测下一个字」的任务上存在信息泄漏 —— 位置 t 的输出
    能直接看到位置 t+1 的答案，训练 loss 会低到不真实（PPL 接近 1），
    且训练/推理分布不一致，生成质量会很差。保留该开关仅用于消融对比实验。

用法:
    python transformer_lm.py --causal 1 --epochs 20
    python transformer_lm.py --causal 0 --epochs 20          # 消融：双向注意力
    python transformer_lm.py --seq_len 128 --nhead 4 --num_layers 4 --save best_transformer.pt

路径约定:
    输入（--corpus）：当前工作目录 → 脚本目录，依次尝试
    输出（--save）  ：相对路径一律落到脚本所在目录，权重跟着代码走
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
# 以下函数与 language_model.py 完全一致，保证两个模型吃同样的数据。

# 脚本所在目录：语料等输入文件优先按这里解析，
# 避免「在别的目录执行 python d:\...\transformer_lm.py」时扫不到同目录的 .txt
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 特殊 token：<bos> 文档开头，<eos> 文档结尾。
# 用字符串而不是单个字符，所以永远不会和语料里的字撞车；id 固定占 0 / 1。
# 训练时每篇文档编码成 <bos> + 正文 + <eos>，模型因此能学会「这里该收尾了」，
# 生成时预测到 <eos> 就停 —— 这就是文本结尾的显式建模。
BOS, EOS = "<bos>", "<eos>"


def resolve_input(pattern):
    """
    相对路径的解析顺序：当前工作目录 → 脚本所在目录。
    绝对路径原样返回。
    先查 cwd 是为了符合命令行惯例（--corpus "*.txt" 应该匹配你当前所在的目录），
    只有当 cwd 里一个都没匹配上时，才回退到脚本目录兜底。
    """
    if os.path.isabs(pattern):
        return pattern
    if glob.glob(pattern):
        return pattern
    local = os.path.join(BASE_DIR, pattern)
    return local if glob.glob(local) else pattern


def resolve_output(path):
    """
    输出路径（模型权重）按脚本所在目录解析：相对路径一律落到代码同目录，
    这样无论在哪个目录执行，权重都跟着代码走，不会散落在 cwd 里。
    绝对路径原样返回。
    """
    return path if os.path.isabs(path) else os.path.join(BASE_DIR, path)


def load_corpus(pattern="*.txt"):
    pattern = resolve_input(pattern)
    paths = sorted(glob.glob(pattern))          # 排序保证每次拼出的语料顺序一致
    if not paths:
        raise FileNotFoundError(
            f"未找到匹配 {pattern!r} 的文件。\n"
            f"  脚本目录    : {BASE_DIR}\n"
            f"  当前工作目录: {os.getcwd()}\n"
            f"  用 --corpus 显式指定路径，例如:\n"
            f"    python transformer_lm.py --corpus \"{os.path.join(BASE_DIR, '*.txt')}\""
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

    <bos> / <eos> 固定占 id 0 / 1，真实字符从 2 开始。
    """
    from collections import Counter
    counter = Counter(text)
    chars = [c for c, _ in counter.most_common(max_vocab)] if max_vocab else sorted(set(text))
    chars = [c for c in chars if c not in (BOS, EOS)]   # 防御：语料里真有这串字符也不冲突
    chars = [BOS, EOS] + chars
    char2idx = {c: i for i, c in enumerate(chars)}
    idx2char = {i: c for c, i in char2idx.items()}
    return char2idx, idx2char


class CharDataset(Dataset):
    """
    字符级语言模型的训练样本：输入 x = 连续 seq_len 个字符，目标 y = 往后错一位。

    每篇文档编码成 <bos> + 正文 + <eos>，再首尾相接成一条流（packing）。
    这样窗口虽然还会跨文档，但边界是模型能看见、也能预测的 <eos>/<bos>，
    而不是把两段无关的文字悄悄粘在一起 —— 语料平均行长才 15 字，
    若强行要求窗口不跨文档，seq_len=128 的窗口根本装不下一篇。
    顺带的好处：模型学会在文末输出 <eos>，生成时就有了自然的停止信号。

    切块方式固定为「不重叠、首尾相接」，和大模型预训练的 packing 一致：
    样本数 = 语料长度 / seq_len，每个 token 只贡献一次 loss。
    重叠切块（每挪一格切一个样本）会让同一段文字被前向 seq_len 次，
    纯属重复计算，这里不提供该选项。
    """

    def __init__(self, docs, char2idx, seq_len):
        self.seq_len = seq_len
        bos_id = char2idx.get(BOS)
        eos_id = char2idx.get(EOS)
        ids = []
        for doc in docs:
            body = [char2idx[c] for c in doc if c in char2idx]
            if not body:
                continue
            if bos_id is not None:
                ids.append(bos_id)
            ids.extend(body)
            if eos_id is not None:
                ids.append(eos_id)
        self.data = torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        # 要求每个样本后面还有 1 个 token 当最后一位的标签，所以是 -seq_len-1
        return max(0, (len(self.data) - self.seq_len - 1) // self.seq_len + 1)

    def __getitem__(self, idx):
        start = idx * self.seq_len          # 窗口首尾相接，没有重叠
        x = self.data[start: start + self.seq_len]
        y = self.data[start + 1: start + self.seq_len + 1]
        return x, y


# ─────────────────────────── 模型 ───────────────────────────

class TransformerLM(nn.Module):
    """
    字符级 Transformer 语言模型。

    参数语义（和 RNN 版的对应关系）：
        embed_dim  → d_model，主干的宽度（相当于 RNN 的 hidden_dim 角色）
        hidden_dim → 前馈网络 FFN 的中间维度，惯例取 4 × embed_dim
        num_layers → Encoder 层数
        nhead      → 多头注意力的头数，必须能整除 embed_dim
        max_len    → 位置编码的上限，即模型能处理的最大序列长度
        causal     → True: 加下三角因果掩码（自回归）；False: 不加掩码（双向）
    """

    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers,
                 nhead, dropout, max_len, causal=True):
        super().__init__()
        if embed_dim % nhead != 0:
            raise ValueError(f"embed_dim({embed_dim}) 必须能被 nhead({nhead}) 整除")

        self.max_len = max_len
        self.causal = causal
        self.vocab_size = vocab_size

        self.tok_embed = nn.Embedding(vocab_size, embed_dim)
        # 可学习的位置编码：字符级模型序列不长（64~256），比 sin/cos 更好训也够用
        self.pos_embed = nn.Embedding(max_len, embed_dim)
        self.drop = nn.Dropout(dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,        # Pre-LN：深层 Transformer 更稳，不容易梯度爆炸
        )
        self.blocks = nn.TransformerEncoder(
            layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, vocab_size)

        self._mask_cache = {}       # (T, device) → 掩码张量，避免每个 batch 重建

    def _causal_mask(self, T, device):
        """上三角填 -inf 的 (T, T) 掩码：位置 i 只能 attend 到 j <= i。"""
        key = (T, str(device))
        mask = self._mask_cache.get(key)
        if mask is None:
            mask = torch.full((T, T), float("-inf"), device=device).triu(1)
            self._mask_cache[key] = mask
        return mask

    def forward(self, x):
        B, T = x.shape
        if T > self.max_len:
            raise ValueError(f"序列长度 {T} 超过 max_len={self.max_len}，请调大 --seq_len/--max_len")

        pos = torch.arange(T, device=x.device)
        h = self.drop(self.tok_embed(x) + self.pos_embed(pos))

        # 唯一的区别就在这里：加不加因果掩码
        mask = self._causal_mask(T, x.device) if self.causal else None
        h = self.blocks(h, mask=mask)

        logits = self.fc(self.norm(h))     # (B, T, V)
        return logits


# ─────────────────────────── 训练 / 评估 ───────────────────────────

def _fmt_sec(sec):
    """秒 → 1:23:45 / 3:20 这样的可读时长。"""
    sec = int(max(0, sec))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def run_epoch(model, loader, criterion, optimizer, device,
              train=True, clip=1.0, scheduler=None,
              epoch=0, total_epochs=0, log_every=0, tag=""):
    """
    log_every > 0 时每 N 个 batch 打一次进度（loss / 速度 / 剩余时间）。
    Transformer 单步比 RNN 慢，没有进度输出很容易被误认为卡死。
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
            # Transformer 对梯度尺度敏感，裁剪几乎是无成本的安全带
            if clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()        # 每步（而非每 epoch）更新学习率

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
                print("\r" + msg.ljust(118), end="", flush=True)
            else:
                print(msg, flush=True)

    if log_every and tty:
        print("\r" + " " * 118 + "\r", end="", flush=True)   # 清掉进度行

    avg_loss = total_loss / max(1, total_tokens)
    ppl = math.exp(min(avg_loss, 700))   # 防止 exp 溢出
    return avg_loss, ppl


def make_scheduler(optimizer, total_steps, warmup_ratio=0.1, min_lr_ratio=0.05):
    """
    Linear warmup + cosine 衰减。
    Transformer 一开始注意力很随机，直接上大学习率容易训崩，warmup 是必需品。
    """
    warmup_steps = max(1, int(total_steps * warmup_ratio))

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, progress)
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ─────────────────────────── 主函数 ───────────────────────────

def main():
    parser = argparse.ArgumentParser()
    # —— causal 掩码开关（本脚本的核心参数）——
    parser.add_argument("--causal",     type=int,   default=1, choices=[0, 1],
                        help="1=自回归（加因果掩码，正确做法）；0=双向注意力（仅用于消融对比，会泄漏答案）")
    # —— 结构超参 ——
    parser.add_argument("--epochs",     type=int,   default=20)
    parser.add_argument("--seq_len",    type=int,   default=128)
    parser.add_argument("--max_len",    type=int,   default=0, help="位置编码上限，0=自动(=seq_len)")
    parser.add_argument("--batch_size", type=int,   default=64)
    parser.add_argument("--embed_dim",  type=int,   default=256)
    parser.add_argument("--hidden_dim", type=int,   default=1024, help="FFN 中间维度，惯例 4×embed_dim")
    parser.add_argument("--num_layers", type=int,   default=4)
    parser.add_argument("--nhead",      type=int,   default=8)
    parser.add_argument("--dropout",    type=float, default=0.1)
    # —— 优化器 ——
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="warmup 占总步数的比例")
    parser.add_argument("--grad_clip",  type=float, default=1.0, help="梯度裁剪阈值，0=关闭")
    # —— 数据 / 杂项 ——
    parser.add_argument("--val_ratio",  type=float, default=0.05)
    parser.add_argument("--max_vocab",  type=int,   default=4000, help="字表最大字数，0=不限")
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--corpus",     default="*.txt")
    parser.add_argument("--num_workers", type=int,  default=0)
    parser.add_argument("--log_every",  type=int,   default=50, help="每多少个 batch 打一次进度，0=关闭")
    parser.add_argument("--save",       default="best_transformer.pt",
                        help="权重保存路径；相对路径按脚本所在目录解析，与代码同目录")
    args = parser.parse_args()

    # 写进 checkpoint，generate.py 靠它判断该复原哪种网络结构
    args.model = "transformer"

    # 权重固定落在代码同目录（--save 给相对路径时），并确保父目录存在
    save_path = resolve_output(args.save)
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    if args.max_len <= 0:
        args.max_len = args.seq_len
    if args.max_len < args.seq_len:
        raise ValueError("max_len 不能小于 seq_len")

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  model: TRANSFORMER")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True   # 固定输入尺寸，让 cudnn 选最快实现
    print(f"causal 掩码: {'开 (自回归)' if args.causal else '关 (双向，会泄漏答案)'}")
    if not args.causal:
        print("!! 警告：--causal 0 时每个位置都能看到右侧字符，next-token 任务上答案直接泄漏。")
        print("!!       训练 PPL 会异常低（接近 1），只能作为对照实验，不要用来生成文本。")

    # 数据准备
    text = load_corpus(args.corpus)
    if not text.strip():
        raise ValueError(f"语料内容为空（匹配 {args.corpus!r} 的文件都读不出字符）。")
    print(f"语料字符数: {len(text):,}")

    char2idx, idx2char = build_vocab(text, args.max_vocab or None)
    vocab_size = len(char2idx)
    print(f"词表大小: {vocab_size}（含 {BOS} / {EOS} 两个特殊 token，id 0/1）")

    # 每行当一篇独立文档（本语料是 39 万条独立短问句），空行丢掉。
    # 按文档切分训练/验证集，同一篇不会同时出现在两边。
    docs = [ln for ln in text.splitlines() if ln.strip()]
    if not docs:
        raise ValueError("语料里没有非空行。")
    random.shuffle(docs)
    split = int(len(docs) * (1 - args.val_ratio))
    train_docs, val_docs = docs[:split], docs[split:]
    print(f"文档数: 训练 {len(train_docs):,}  验证 {len(val_docs):,}")

    train_ds = CharDataset(train_docs, char2idx, args.seq_len)
    val_ds   = CharDataset(val_docs,   char2idx, args.seq_len)

    if len(train_ds) == 0 or len(val_ds) == 0:
        raise ValueError(
            f"样本数为 0（train={len(train_ds)}, val={len(val_ds)}）。"
            f"语料太短或 seq_len({args.seq_len}) 太大，请减小 seq_len 或增加语料。"
        )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              drop_last=True, num_workers=args.num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              drop_last=True, num_workers=args.num_workers)

    if len(train_loader) == 0:
        raise ValueError("train_loader 为空（样本数少于 batch_size），请减小 batch_size。")

    print(f"训练样本: {len(train_ds):,}  验证样本: {len(val_ds):,}  "
          f"每轮 batch 数: {len(train_loader)}")

    # 模型
    model = TransformerLM(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        nhead=args.nhead,
        dropout=args.dropout,
        max_len=args.max_len,
        causal=bool(args.causal),
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")
    # --save 的相对路径按脚本目录解析，这里打印绝对路径便于核对
    print(f"模型将保存至: {save_path}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay, betas=(0.9, 0.98))
    total_steps = len(train_loader) * args.epochs
    scheduler = make_scheduler(optimizer, total_steps, args.warmup_ratio)

    best_val_ppl = float("inf")

    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Train PPL':>10}  "
          f"{'Val Loss':>10}  {'Val PPL':>10}  {'LR':>9}")
    print("-" * 68)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_ppl = run_epoch(model, train_loader, criterion, optimizer, device,
                                    train=True, clip=args.grad_clip, scheduler=scheduler,
                                    epoch=epoch, total_epochs=args.epochs,
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
                "args": vars(args),          # 含 causal / nhead / max_len，供 generate.py 复原结构
            }, save_path)

        cur_lr = scheduler.get_last_lr()[0]
        print(f"{epoch:>6}  {tr_loss:>10.4f}  {tr_ppl:>10.2f}  "
              f"{va_loss:>10.4f}  {va_ppl:>10.2f}  {cur_lr:>9.2e}{marker}")

    print(f"\n训练完成。最佳验证 PPL: {best_val_ppl:.2f}  已保存至 {save_path}"
          if best_val_ppl < float("inf") else "\n训练完成。没有跑任何 epoch，未保存模型。")


if __name__ == "__main__":
    main()
