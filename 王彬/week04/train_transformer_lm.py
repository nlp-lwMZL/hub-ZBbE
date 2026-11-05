"""
从零训练一个小型 GPT 风格（decoder-only Transformer）中文语言模型。

资源来源：
  - 词表 / 分词器：bert-base-chinese 文件夹里的 tokenizer.json、vocab.txt（21128 个 token）
  - 语料：循环神经网络语言模型/corpus_general.txt（约 600 万字）

模型做的事：给定前文，预测下一个 token（自回归语言模型），可续写文本。
训练完成后保存权重，可直接用 --prompt 生成。

用法示例：
    python train_transformer_lm.py                          # 默认配置完整训练
    python train_transformer_lm.py --epochs 10 --seq_len 256
    python train_transformer_lm.py --max_chars 300000 --epochs 1   # 快速试跑
    python train_transformer_lm.py --load --prompt 今天天气        # 不训练，加载已有权重续写
"""

import argparse
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizerFast


# ─────────────────────────── 默认路径 ───────────────────────────

HERE = Path(__file__).resolve().parent
# 课件目录的上一级再上一级是工作区根目录，bert-base-chinese 就在那里
DEFAULT_BERT_DIR = HERE.parents[1] / "bert-base-chinese" / "bert-base-chinese"
DEFAULT_CORPUS = HERE / "循环神经网络语言模型" / "corpus_general.txt"


# ─────────────────────────── 数据 ───────────────────────────

def read_lines(path, max_chars=0):
    """按行读取语料，丢弃空行；max_chars > 0 时只取前若干字符用于快速试跑。"""
    lines = []
    total = 0
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            lines.append(line)
            total += len(line)
            if max_chars and total >= max_chars:
                break
    return lines


def encode_lines(lines, tokenizer):
    """
    把多行文本编码成 token id 序列。
    行与行之间插入 [SEP](102)，让模型知道"上一句结束了"，避免跨句直接粘连。
    """
    text = " [SEP] ".join(lines)
    ids = tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )["input_ids"]
    return ids


def decode_ids(tokenizer, ids):
    """WordPiece 解码后中文 token 之间会自动带空格，去掉空格便于阅读。"""
    return tokenizer.decode(ids, skip_special_tokens=True).replace(" ", "")


class TokenChunkDataset(Dataset):
    """
    语言模型样本：x = 连续 seq_len 个 token，y = 整体右移一位的下一 token。
    相邻样本不重叠（stride = seq_len），每个 token 每轮只被训练一次。
    """

    def __init__(self, ids, seq_len):
        self.seq_len = seq_len
        self.data = torch.from_numpy(np.asarray(ids, dtype=np.int64))
        self.n_chunks = max(0, (len(self.data) - 1) // seq_len)

    def __len__(self):
        return self.n_chunks

    def __getitem__(self, idx):
        start = idx * self.seq_len
        x = self.data[start:start + self.seq_len]
        y = self.data[start + 1:start + self.seq_len + 1]
        return x, y


# ─────────────────────────── 模型 ───────────────────────────

class CausalSelfAttention(nn.Module):
    """多头因果自注意力：每个位置只能看到自己和之前的 token。"""

    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x, mask):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, T, C) -> (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        att = att.masked_fill(mask[:, :, :T, :T] == 0, torch.finfo(att.dtype).min)
        att = self.attn_drop(F.softmax(att, dim=-1))
        out = att @ v                                    # (B, n_head, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(out))


class MLP(nn.Module):
    """Transformer 里的前馈网络：升维 4 倍，GELU，再降回原维度。"""

    def __init__(self, n_embd, dropout):
        super().__init__()
        self.fc = nn.Linear(n_embd, 4 * n_embd)
        self.proj = nn.Linear(4 * n_embd, n_embd)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    """Pre-LN 残差块（GPT-2 结构）：LN -> Attention -> 残差 -> LN -> MLP -> 残差。"""

    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, dropout)

    def forward(self, x, mask):
        x = x + self.attn(self.ln_1(x), mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class TransformerLM(nn.Module):
    def __init__(self, vocab_size, block_size, n_layer, n_head, n_embd, dropout):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [Block(n_embd, n_head, dropout) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        # 输入输出 embedding 共享权重，省一半 embedding 参数
        self.head.weight = self.tok_emb.weight

        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(block_size, block_size, dtype=torch.bool))
            .view(1, 1, block_size, block_size),
            persistent=False,
        )
        self.apply(self._init_weights)
        # 残差分支输出做缩放，深层训练更稳定
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x, self.causal_mask)
        x = self.ln_f(x)
        return self.head(x)                              # (B, T, vocab_size)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.9, top_k=50):
        """自回归采样，返回拼接后的 token 序列 (B, T + max_new_tokens)。"""
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits = self(idx_cond)[:, -1, :] / max(temperature, 1e-6)
            if top_k:
                k = min(top_k, logits.size(-1))
                values, _ = torch.topk(logits, k)
                logits[logits < values[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx


# ─────────────────────────── 训练 / 评估 ───────────────────────────

def evaluate(model, loader, device, vocab_size):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
            total_loss += loss.item() * y.numel()
            total_tokens += y.numel()
    avg_loss = total_loss / max(total_tokens, 1)
    return avg_loss, math.exp(avg_loss)


@torch.no_grad()
def generate_text(model, tokenizer, device, prompt, max_new_tokens, temperature, top_k):
    """用 prompt 续写，返回解码后的文本。"""
    ids = tokenizer(prompt, add_special_tokens=False, return_attention_mask=False)["input_ids"]
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens, temperature, top_k)
    return decode_ids(tokenizer, out[0].tolist())


def make_lr_lambda(total_steps, warmup_steps, min_lr_ratio):
    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        progress = min(progress, 1.0)
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    return lr_lambda


def main():
    parser = argparse.ArgumentParser(description="从零训练小型 Transformer 中文语言模型")
    # 路径
    parser.add_argument("--bert_dir", default=str(DEFAULT_BERT_DIR), help="bert-base-chinese 文件夹（提供分词器）")
    parser.add_argument("--corpus", default=str(DEFAULT_CORPUS), help="训练语料 txt")
    parser.add_argument("--save", default=str(HERE / "best_transformer_lm.pt"), help="最佳模型保存路径")
    parser.add_argument("--load", nargs="?", const=str(HERE / "best_transformer_lm.pt"), default=None,
                        help="加载已有权重，跳过训练直接续写；不跟路径时用默认权重")
    # 数据
    parser.add_argument("--seq_len", type=int, default=128, help="上下文长度")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--max_chars", type=int, default=0, help="只用前 N 个字符，0=全部（快速试跑用）")
    parser.add_argument("--num_workers", type=int, default=0, help="Windows 下建议 0")
    # 模型
    parser.add_argument("--n_layer", type=int, default=4)
    parser.add_argument("--n_head", type=int, default=8)
    parser.add_argument("--n_embd", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    # 优化
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=50)
    # 生成
    parser.add_argument("--prompt", default="今天天气", help="训练结束后用于续写的开头")
    parser.add_argument("--generate_tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device} | torch threads: {torch.get_num_threads()}")

    # ── 加载已有权重：跳过训练，直接续写 ──
    if args.load:
        ckpt = torch.load(args.load, map_location=device, weights_only=False)
        tokenizer = BertTokenizerFast.from_pretrained(ckpt["tokenizer_dir"])
        model = TransformerLM(**ckpt["model_config"]).to(device)
        model.load_state_dict(ckpt["model_state"])
        print(f"已加载权重: {args.load}")
        print(f"\nprompt: {args.prompt}")
        print(generate_text(model, tokenizer, device, args.prompt,
                            args.generate_tokens, args.temperature, args.top_k))
        return

    # ── 分词器与语料 ──
    if not Path(args.bert_dir).exists():
        raise FileNotFoundError(f"找不到 BERT 文件夹: {args.bert_dir}")
    tokenizer = BertTokenizerFast.from_pretrained(args.bert_dir)
    vocab_size = tokenizer.vocab_size
    print(f"分词器加载完成，词表大小: {vocab_size}")

    lines = read_lines(args.corpus, args.max_chars)
    if not lines:
        raise FileNotFoundError(f"语料为空或不存在: {args.corpus}")
    random.shuffle(lines)
    split = int(len(lines) * (1 - args.val_ratio))
    train_lines, val_lines = lines[:split], lines[split:]

    t0 = time.time()
    print(f"语料行数: {len(lines):,}（训练 {len(train_lines):,} / 验证 {len(val_lines):,}），开始分词 ...")
    train_ids = encode_lines(train_lines, tokenizer)
    val_ids = encode_lines(val_lines, tokenizer)
    print(f"token 数: 训练 {len(train_ids):,} / 验证 {len(val_ids):,}，分词耗时 {time.time() - t0:.1f}s")

    train_ds = TokenChunkDataset(train_ids, args.seq_len)
    val_ds = TokenChunkDataset(val_ids, args.seq_len)
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise ValueError("样本数不足，请增大语料或减小 --seq_len")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        drop_last=True, num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        drop_last=False, num_workers=args.num_workers,
    )
    print(f"训练样本: {len(train_ds):,} 块（{args.seq_len} token/块），每轮 {len(train_loader)} 步")

    # ── 模型 ──
    model = TransformerLM(
        vocab_size=vocab_size,
        block_size=args.seq_len,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {n_params:,}")

    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95),
    )

    total_steps = args.epochs * len(train_loader)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, make_lr_lambda(total_steps, warmup_steps, args.min_lr_ratio)
    )

    best_val_ppl = float("inf")
    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Train PPL':>10}  {'Val Loss':>10}  {'Val PPL':>10}  {'耗时':>8}")
    print("-" * 70)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_start = time.time()
        running_loss, seen_tokens = 0.0, 0

        for step, (x, y) in enumerate(train_loader, 1):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item() * y.numel()
            seen_tokens += y.numel()

            if step % args.log_every == 0:
                avg = running_loss / seen_tokens
                lr = scheduler.get_last_lr()[0]
                print(f"  epoch {epoch} step {step}/{len(train_loader)} "
                      f"loss {avg:.4f} ppl {math.exp(avg):.1f} lr {lr:.2e}")

        train_loss = running_loss / max(seen_tokens, 1)
        train_ppl = math.exp(train_loss)
        val_loss, val_ppl = evaluate(model, val_loader, device, vocab_size)

        marker = "  *" if val_ppl < best_val_ppl else ""
        if val_ppl < best_val_ppl:
            best_val_ppl = val_ppl
            torch.save({
                "model_state": model.state_dict(),
                "model_config": {
                    "vocab_size": vocab_size,
                    "block_size": args.seq_len,
                    "n_layer": args.n_layer,
                    "n_head": args.n_head,
                    "n_embd": args.n_embd,
                    "dropout": args.dropout,
                },
                "tokenizer_dir": str(Path(args.bert_dir).resolve()),
                "args": vars(args),
            }, args.save)

        print(f"{epoch:>6}  {train_loss:>10.4f}  {train_ppl:>10.2f}  {val_loss:>10.4f}  {val_ppl:>10.2f}  "
              f"{time.time() - epoch_start:>7.0f}s{marker}")

    print(f"\n训练完成。最佳验证 PPL: {best_val_ppl:.2f}，已保存至 {args.save}")

    # ── 用最佳权重续写 ──
    ckpt = torch.load(args.save, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    print(f"\n续写示例（prompt: {args.prompt}）:")
    print(generate_text(model, tokenizer, device, args.prompt,
                        args.generate_tokens, args.temperature, args.top_k))


if __name__ == "__main__":
    main()
