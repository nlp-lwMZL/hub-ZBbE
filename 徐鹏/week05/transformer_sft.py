"""
transformer_sft.py — 在字符级 Transformer 语言模型上做 SFT（监督微调）

和 transformer_lm.py 的唯一区别：
    SFT 只对「回答」部分计算损失，指令部分只作为上下文输入。
    做法：数据用 \x01 分隔指令与回答、\x02 结束样本；
          CharDataset 额外返回 mask（回答位置=1，指令/分隔符=0）；
          训练时把 mask=0 位置的标签设为 -100，交叉熵 ignore_index=-100 自动忽略。

用法:
    # 1) 把 Alpaca 格式数据（.json / .jsonl / .csv）转成字符级语料
    python transformer_sft.py --to_corpus alpaca_data/train.csv --out_corpus corpus_sft.txt
    # 2) 训练
    python transformer_sft.py --corpus corpus_sft.txt --epochs 2 --seq_len 128 --batch_size 64 --save sft_tf.pt
    # 3) 生成（输入里要带 \x01，模型才会进入“回答模式”；命令行里写 "\x01" 即可）
    python transformer_sft.py --generate "计算总费用。汽车3000元，书20元\x01" --ckpt sft_tf.pt
"""

import argparse
import glob
import json
import math
import os
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

HERE = os.path.dirname(os.path.abspath(__file__))

# 指令/回答分界符与样本结束符。选这两个字符是因为正常中文语料里几乎不会出现，
# 建字表时它们会被当成普通字符收进去，正好用来做掩码定位。
SEP = "\x01"
EOS = "\x02"


# ─────────────────────── 数据转换（Alpaca 数据 → 字符级语料） ───────────────────────


def _load_alpaca_records(path):
    """读取 Alpaca 记录，支持 .json / .jsonl / .csv。"""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        import csv

        with open(path, encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    if ext in (".jsonl", ".ndjson"):
        records = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        raise ValueError(
            f"{path} 不是 Alpaca 记录数组，而是字典（HuggingFace 的 "
            "dataset_infos.json 只是元信息）。请直接传入实际的 train.csv / train.json。"
        )
    if not isinstance(data, list):
        raise ValueError(f"{path} 里不是记录数组，无法转换。")
    return data


def to_corpus(json_path, out_path):
    """
    把 Alpaca 格式数据（instruction / input / output）转成字符级语料。
    每条样本形如：  指令内容 [输入内容] \x01 回答内容 \x02

    注意：回答内部可能本身含换行，所以一条样本不一定只占一行；
    训练时按 \x02 切分样本（见 split_samples），不能按行切分。
    """
    data = _load_alpaca_records(json_path)
    lines = []
    for item in data:
        if not isinstance(item, dict):
            raise ValueError(
                f"记录格式不对，期望字典，实际是 {type(item).__name__}: {item!r}"
            )
        ins = (item.get("instruction") or "").strip()
        inp = (item.get("input") or "").strip()
        out = (item.get("output") or "").strip()
        if not ins or not out:
            continue
        # 统一换行符，避免 \r 混进字表
        ins, inp, out = (
            s.replace("\r\n", "\n").replace("\r", "\n") for s in (ins, inp, out)
        )
        text = f"{ins} {inp}{SEP}{out}{EOS}" if inp else f"{ins}{SEP}{out}{EOS}"
        lines.append(text)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"写出 {len(lines)} 条 → {out_path}")


# ─────────────────────── 数据 ───────────────────────


def load_corpus(pattern):
    text = ""
    for p in sorted(glob.glob(pattern)):
        with open(p, encoding="utf-8", errors="ignore") as f:
            text += f.read()
    if not text:
        raise FileNotFoundError(f"没读到语料: {pattern}")
    return text


def split_samples(text):
    """
    按 EOS 切分样本。Alpaca 的输出里常含换行，如果按行切会把一条样本切成几段，
    导致训练/验证互相泄漏，也会让 CharSFTDataset 的 in_answer 状态错位。
    """
    if EOS in text:
        chunks = text.split(EOS)
    else:
        print(f"[警告] 语料里没有 EOS({EOS!r})，退化为按行切分")
        chunks = text.splitlines()  # type: ignore[assignment]
    samples = []
    for chunk in chunks:
        s = chunk.strip("\r\n")
        if s and SEP in s:
            samples.append(s)
    return samples


def build_vocab(text, max_vocab):
    from collections import Counter

    counted = Counter(text)
    if max_vocab:
        chars = [c for c, _ in counted.most_common(max_vocab)]
    else:
        chars = sorted(counted)
    # SEP/EOS 是 SFT 掩码的锚点，绝不能因为 max_vocab 截断而丢掉
    for s in (SEP, EOS):
        if s in counted and s not in chars:
            chars.append(s)
    char2idx = {c: i for i, c in enumerate(chars)}
    return char2idx, {i: c for c, i in char2idx.items()}


class CharSFTDataset(Dataset):
    """
    和原 CharDataset 的区别：多返回一个 mask，标记 y 的每个位置是否属于“回答”。

    x: (T,) 输入字符 id
    y: (T,) 目标字符 id（错一位）
    m: (T,) 1=回答位置（算损失），0=指令/分隔符位置（不算损失）
    """

    def __init__(self, text, char2idx, seq_len, stride):
        ids = []
        mask = []
        in_answer = False
        for c in text:
            if c not in char2idx:
                continue
            ids.append(char2idx[c])
            if c == SEP:
                in_answer = True
                mask.append(0)  # 分隔符本身不算回答
            elif c == EOS:
                in_answer = False
                mask.append(0)  # 结束符也不算
            else:
                mask.append(1 if in_answer else 0)
        self.data = torch.tensor(ids, dtype=torch.long)
        self.mask = torch.tensor(mask, dtype=torch.long)
        self.seq_len = seq_len
        self.stride = max(1, stride)

    def __len__(self):
        return max(0, (len(self.data) - self.seq_len - 1) // self.stride + 1)

    def __getitem__(self, i):
        s = i * self.stride
        x = self.data[s : s + self.seq_len]
        y = self.data[s + 1 : s + self.seq_len + 1]
        # mask 与 y 对齐：y 是“下一个字”，所以掩码也从 s+1 开始取
        m = self.mask[s + 1 : s + self.seq_len + 1]
        return x, y, m


# ─────────────────────── 模型（与原文件一致） ───────────────────────


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=192,
        n_head=4,
        n_layer=4,
        max_len=512,
        dropout=0.1,
        use_pos=True,
        causal=True,
    ):
        super().__init__()
        self.causal = causal
        self.use_pos = use_pos
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model) if use_pos else None
        self.drop = nn.Dropout(dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=n_layer,
            norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False,
        )
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok.weight

        nn.init.normal_(self.tok.weight, std=0.02)
        if self.pos is not None:
            nn.init.normal_(self.pos.weight, std=0.02)

    def forward(self, idx):
        x = self.tok(idx)
        if self.use_pos:
            x = x + self.pos(torch.arange(idx.size(1), device=idx.device))
        x = self.drop(x)
        mask = None
        if self.causal:
            mask = nn.Transformer.generate_square_subsequent_mask(
                idx.size(1), device=idx.device
            )
        x = self.encoder(x, mask=mask)
        return self.head(x)


# ─────────────────────── 训练 / 评估（核心改动在这里） ───────────────────────


@torch.no_grad()
def evaluate(model, loader, device, last_only=False):
    """返回 (loss, ppl)。SFT 口径：只统计回答位置；last_only 再额外只看每段最后一位。"""
    model.eval()
    total, tokens = 0.0, 0
    for x, y, m in loader:
        x, y, m = x.to(device), y.to(device), m.to(device)
        logits = model(x)
        if last_only:
            logits, y, m = logits[:, -1, :], y[:, -1], m[:, -1]
        # 指令位置标签设为 -100，交叉熵自动忽略
        y_masked = y.clone()
        y_masked[m == 0] = -100
        total += float(
            F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y_masked.reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )
        )
        # 只统计真正参与损失的 token 数
        tokens += int((y_masked != -100).sum())
    if tokens == 0:
        return float("nan"), float("nan")
    loss = total / tokens
    return loss, math.exp(loss)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    text = load_corpus(args.corpus)
    if args.max_chars and len(text) > args.max_chars:
        text = text[: args.max_chars]
    char2idx, idx2char = build_vocab(text, args.max_vocab)
    V = len(char2idx)
    print(f"device: {device}  语料 {len(text):,} 字  字表 {V}")

    if SEP not in char2idx or EOS not in char2idx:
        raise ValueError(
            f"语料里缺少分隔符 {SEP!r} 或结束符 {EOS!r}，无法做 SFT 掩码。"
            "请先用 --to_corpus 生成 SFT 语料。"
        )

    # 按「样本」打乱再切分，保证验证集里的样本训练时没见过。
    # 注意不能用 splitlines()：Alpaca 的回答里本身可能含换行。
    samples = split_samples(text)
    if len(samples) < 2:
        raise ValueError("可用 SFT 样本太少（至少需要 2 条），请检查语料格式。")
    random.shuffle(samples)
    split = min(max(1, int(len(samples) * (1 - args.val_ratio))), len(samples) - 1)
    tr = CharSFTDataset(
        EOS.join(samples[:split]) + EOS, char2idx, args.seq_len, args.stride
    )
    va = CharSFTDataset(
        EOS.join(samples[split:]) + EOS, char2idx, args.seq_len, args.stride
    )
    train_loader = DataLoader(
        tr, batch_size=args.batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        va, batch_size=args.batch_size, shuffle=False, drop_last=True
    )

    model = TransformerLM(
        V,
        args.d_model,
        args.heads,
        args.layers,
        args.seq_len,
        args.dropout,
        use_pos=not args.no_pos,
        causal=not args.no_mask,
    ).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"模型参数量 {n_par/1e6:.2f}M  （SFT 模式：仅回答位置计算损失）")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    best = float("inf")
    print(
        f"\n{'Epoch':>6} {'TrainLoss':>10} {'ValPPL(回答)':>14} {'ValPPL(末位)':>13} {'tok/s':>9}"
    )
    print("-" * 62)

    for ep in range(1, args.epochs + 1):
        model.train()
        st, seen, tot = time.time(), 0, 0.0
        for x, y, m in train_loader:
            x, y, m = x.to(device), y.to(device), m.to(device)
            y_masked = y.clone()
            y_masked[m == 0] = -100
            n_tok = int((y_masked != -100).sum())
            if n_tok == 0:
                continue  # 整批都没有回答 token，跳过，避免 loss 变 NaN
            logits = model(x)
            loss = F.cross_entropy(
                logits.reshape(-1, V),
                y_masked.reshape(-1),
                ignore_index=-100,
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss) * n_tok
            seen += n_tok
        dt = time.time() - st

        _, v_ppl = evaluate(model, val_loader, device, last_only=False)
        _, v_ppl_last = evaluate(model, val_loader, device, last_only=True)
        mark = ""
        if not math.isnan(v_ppl) and v_ppl < best:
            best = v_ppl
            mark = " *"
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "char2idx": char2idx,
                    "idx2char": idx2char,
                    "args": vars(args),
                },
                args.save,
            )
        print(
            f"{ep:>6} {tot/max(seen,1):>10.4f} {v_ppl:>14.2f} {v_ppl_last:>13.2f} {seen/dt:>9,.0f}{mark}"
        )

    if math.isinf(best):
        print("\n[警告] 没有保存模型：验证集上没有可计算的回答 token（请检查语料/掩码）")
    else:
        print(f"\n最佳验证 PPL（回答位置）: {best:.2f}  已保存 {args.save}")


@torch.no_grad()
def generate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location=device)
    c = ck["args"]
    char2idx, idx2char = ck["char2idx"], ck["idx2char"]
    # 位置编码只有 c["seq_len"] 个，输入绝不能超过它（不是命令行的 --seq_len）
    max_len = int(c["seq_len"])
    model = TransformerLM(
        len(char2idx),
        c["d_model"],
        c["heads"],
        c["layers"],
        max_len,
        0.0,
        use_pos=not c.get("no_pos", False),
        causal=not c.get("no_mask", False),
    ).to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()

    # 命令行里通常打不出真正的 \x01，允许用户写转义形式 "\x01"
    prompt = args.generate.replace("\\x01", SEP).replace("\\x02", EOS)
    ids = [char2idx[ch] for ch in prompt if ch in char2idx]
    if not ids:
        raise SystemExit("输入里没有字表内的字符，无法生成。请检查输入或字表。")
    if SEP not in prompt:
        print(f"[警告] 输入里没有分隔符 {SEP!r}（可写成 \\x01），模型可能不会进入回答模式。")

    eos_id = char2idx.get(EOS)
    out = list(ids)
    for _ in range(args.n_chars):
        # 只保留最近 max_len 个字符，否则会超出位置编码范围
        logits = model(torch.tensor([out[-max_len:]], device=device))[0, -1]
        if args.temperature <= 0.01:
            nxt = int(logits.argmax())
        else:
            nxt = int(
                torch.multinomial(F.softmax(logits / args.temperature, dim=-1), 1)
            )
        out.append(nxt)
        if eos_id is not None and nxt == eos_id:
            break
    print(f"输入: {args.generate}")
    print(f"续写: {''.join(idx2char[i] for i in out[len(ids):])}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=os.path.join(HERE, "corpus_sft.txt"))
    ap.add_argument("--max_chars", type=int, default=0)
    ap.add_argument("--max_vocab", type=int, default=4000)
    ap.add_argument("--seq_len", type=int, default=128)  # SFT 样本更长，默认调大
    ap.add_argument("--stride", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=64)  # 样本更长，batch 调小
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d_model", type=int, default=192)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--val_ratio", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-mask", dest="no_mask", action="store_true")
    ap.add_argument("--no-pos", dest="no_pos", action="store_true")
    ap.add_argument("--save", default="sft_tf.pt")
    ap.add_argument("--generate", default=None)
    ap.add_argument("--ckpt", default="sft_tf.pt")
    ap.add_argument("--n_chars", type=int, default=60)
    ap.add_argument("--temperature", type=float, default=0.8)
    # 数据转换开关：--to_corpus 给 Alpaca 文件路径（.json/.jsonl/.csv），只转换不训练
    ap.add_argument(
        "--to_corpus", default=None, help="Alpaca 数据路径（.json/.jsonl/.csv），转成字符级语料"
    )
    ap.add_argument("--out_corpus", default="corpus_sft.txt")
    args = ap.parse_args()

    if args.to_corpus:
        to_corpus(args.to_corpus, args.out_corpus)
    elif args.generate:
        generate(args)
    else:
        train(args)


if __name__ == "__main__":
    main()