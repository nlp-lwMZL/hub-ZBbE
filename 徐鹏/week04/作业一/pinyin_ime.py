"""
基于 Transformer 字符级语言模型的拼音输入法（模型自己训练）。

训练语料来源：
  1) 默认：pinyin_map.json（音节 -> 候选汉字）合成的音节-汉字序列
  2) 可选：--corpus 指定真实文本文件（更推荐，能学到真实语言上下文）

用法:
    python pinyin_ime.py
    python pinyin_ime.py --train
    python pinyin_ime.py --topk 8 --beam 10
    python pinyin_ime.py --train --corpus corpus.txt
"""

import argparse
import bisect
import json
import math
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────── 特殊符号与拼音字母 ───────────────────────

PAD, BOS, EOS, SEP = "<pad>", "<bos>", "<eos>", "<sep>"
LETTERS = "abcdefghijklmnopqrstuvwxyz"


def _load_pinyin_map(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到拼音映射表文件: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


PINYIN_MAP = {}


# ─────────────────────── Transformer 模型 ───────────────────────

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2,
                 dim_feedforward=256, max_len=128, dropout=0.1):
        super().__init__()
        self.max_len = max_len
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # 权重共享
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)

    def forward(self, x, last_only=False):
        B, T = x.shape
        if T > self.max_len:
            raise ValueError(f"序列长度 {T} 超过 max_len={self.max_len}")
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.drop(self.tok_emb(x) + self.pos_emb(pos))
        causal_mask = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1
        )
        h = self.encoder(h, mask=causal_mask)
        h = self.norm(h)
        if last_only:
            h = h[:, -1:, :]
        return self.head(h)


# ─────────────────────── 词表 ───────────────────────

def build_vocab(pinyin_map, corpus_chars=None):
    """特殊符号 + 26 字母 + 映射表汉字 (+ 语料中出现的额外汉字)。"""
    chars, seen = [], set()
    for syllable in sorted(pinyin_map):
        for ch in pinyin_map[syllable]:
            if ch not in seen:
                seen.add(ch)
                chars.append(ch)
    if corpus_chars:
        for ch in corpus_chars:
            if ch not in seen:
                seen.add(ch)
                chars.append(ch)

    idx2char = [PAD, BOS, EOS, SEP] + list(LETTERS) + chars
    char2idx = {c: i for i, c in enumerate(idx2char)}
    return char2idx, idx2char


def char_token_start(idx2char):
    """汉字 token 的起始 id：跳过特殊符号和字母。"""
    return len([PAD, BOS, EOS, SEP]) + len(LETTERS)


# ─────────────────────── 合成语料（默认） ───────────────────────

def _syllable_dists(pinyin_map):
    dists = {}
    for syllable, raw in pinyin_map.items():
        chars = list(dict.fromkeys(raw))
        if not chars:
            continue
        cum, acc = [], 0.0
        for i in range(len(chars)):
            acc += 1.0 / (i + 1)
            cum.append(acc)
        dists[syllable] = (chars, [x / acc for x in cum])
    return dists


def _sample_char(dist, rng):
    chars, cum = dist
    return chars[bisect.bisect_left(cum, rng.random())]


def build_corpus(pinyin_map, char2idx, num_lines=80000, seed=42):
    rng = random.Random(seed)
    dists = _syllable_dists(pinyin_map)
    syllables = sorted(dists)
    if not syllables:
        raise ValueError("pinyin_map 为空，无法生成训练语料")

    tokens = []
    for _ in range(num_lines):
        tokens.append(char2idx[BOS])
        for _ in range(rng.randint(1, 8)):
            syllable = rng.choice(syllables)
            char = _sample_char(dists[syllable], rng)
            tokens.extend(char2idx[c] for c in syllable)
            tokens.append(char2idx[SEP])
            tokens.append(char2idx[char])
        tokens.append(char2idx[EOS])
    return tokens


def build_corpus_from_text(text, pinyin_map, char2idx):
    """
    用真实中文文本构造训练序列。
    对每个汉字反查其候选拼音（取映射表中第一个包含该字的音节），
    拼成 <bos> pin yin <sep> 汉 ... <eos>。
    未收录的字直接跳过（无法提供拼音上下文）。
    """
    char2syl = {}
    for syl, chars in pinyin_map.items():
        for ch in chars:
            char2syl.setdefault(ch, syl)  # 取第一个出现的音节

    tokens = [char2idx[BOS]]
    for ch in text:
        syl = char2syl.get(ch)
        if syl is None or ch not in char2idx:
            continue
        tokens.extend(char2idx[c] for c in syl if c in char2idx)
        tokens.append(char2idx[SEP])
        tokens.append(char2idx[ch])
    tokens.append(char2idx[EOS])
    return tokens


# ─────────────────────── 训练 ───────────────────────

def _make_windows(tokens, block_size, char_start):
    n = (len(tokens) - 1) // block_size
    if n <= 0:
        raise ValueError("语料太短，无法切分训练窗口")
    x = tokens[: n * block_size].view(n, block_size)
    y = tokens[1: n * block_size + 1].view(n, block_size)
    y = torch.where(y >= char_start, y, torch.full_like(y, -100))
    return x, y


def _evaluate(model, x, y, device, batch_size=128):
    model.eval()
    total_nll, total_cnt = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            xb = x[i:i + batch_size].to(device)
            yb = y[i:i + batch_size].to(device)
            logits = model(xb)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), yb.reshape(-1),
                ignore_index=-100, reduction="sum",
            )
            total_nll += loss.item()
            total_cnt += (yb != -100).sum().item()
    return total_nll / max(total_cnt, 1)


def train_transformer(pinyin_map, args, device):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # 读入真实语料（可选）
    corpus_text = None
    corpus_chars = None
    if args.corpus:
        if not os.path.exists(args.corpus):
            raise FileNotFoundError(f"找不到语料文件: {args.corpus}")
        with open(args.corpus, encoding="utf-8", errors="ignore") as f:
            corpus_text = f.read()
        corpus_chars = set(corpus_text)
        print(f"使用真实语料: {args.corpus}  ({len(corpus_text):,} 字)")

    char2idx, idx2char = build_vocab(pinyin_map, corpus_chars)
    char_start = char_token_start(idx2char)  # 修复：显式计算汉字起始 id

    print("生成训练语料……")
    if corpus_text is not None:
        raw_tokens = build_corpus_from_text(corpus_text, pinyin_map, char2idx)
    else:
        raw_tokens = build_corpus(pinyin_map, char2idx,
                                  num_lines=args.num_lines, seed=args.seed)
        print("注意：当前为合成语料，音节/汉字独立采样，模型只能学到字频先验，"
              "上下文效果有限；建议用 --corpus 提供真实文本。")
    tokens = torch.tensor(raw_tokens, dtype=torch.long)
    print(f"语料 token 数: {len(tokens):,}  词表大小: {len(char2idx):,}")

    n_val = max(args.block_size * 2, int(len(tokens) * 0.02))
    x_train, y_train = _make_windows(tokens[:-n_val], args.block_size, char_start)
    x_val, y_val = _make_windows(tokens[-n_val:], args.block_size, char_start)

    config = dict(
        d_model=args.d_model, nhead=args.nhead, num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward, max_len=args.block_size,
        dropout=args.dropout,
    )
    model = TransformerLM(len(char2idx), **config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Transformer 参数量: {n_params / 1e6:.2f}M  "
          f"(d_model={args.d_model}, layers={args.num_layers}, heads={args.nhead})")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95)
    )
    steps_per_epoch = math.ceil(len(x_train) / args.batch_size)
    total_steps = max(1, steps_per_epoch * args.epochs)
    warmup = max(1, int(total_steps * 0.05))

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val = float("inf")
    print(f"开始训练: {args.epochs} epochs, {total_steps} steps, "
          f"batch={args.batch_size}, block={args.block_size}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(x_train))
        running, seen = 0.0, 0
        for i in range(0, len(perm), args.batch_size):
            idx = perm[i:i + args.batch_size]
            xb = x_train[idx].to(device)
            yb = y_train[idx].to(device)
            logits = model(xb)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), yb.reshape(-1),
                ignore_index=-100,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running += loss.item()
            seen += 1

        val_loss = _evaluate(model, x_val, y_val, device)
        print(f"epoch {epoch:02d}  train_loss={running / max(seen, 1):.4f}  "
              f"val_loss={val_loss:.4f}  lr={scheduler.get_last_lr()[0]:.2e}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model_type": "transformer",
                "char2idx": char2idx,
                "idx2char": idx2char,
                "config": config,
                "model_state": model.state_dict(),
                "val_loss": best_val,
            }, args.model_path)
            print(f"  -> 保存最佳模型: {args.model_path}")

    print(f"训练完成，最佳 val_loss={best_val:.4f}")


# ─────────────────────── 拼音分词 ───────────────────────

_SYLLABLES = []


def segment(pinyin_str):
    syllables = []
    for token in pinyin_str.strip().lower().split():
        i = 0
        while i < len(token):
            matched = next((s for s in _SYLLABLES if token[i:].startswith(s)), None)
            if matched:
                syllables.append(matched)
                i += len(matched)
            else:
                i += 1
    return syllables


# ─────────────────────── 束搜索 ───────────────────────

def _encode_prefix(prefix_syllables, prefix, char2idx):
    ids = [char2idx[BOS]]
    for syllable, ch in zip(prefix_syllables, prefix):
        if ch not in char2idx:
            continue
        ids.extend(char2idx[c] for c in syllable if c in char2idx)
        ids.append(char2idx[SEP])
        ids.append(char2idx[ch])
    return ids


def beam_search(syllables, prefix_syllables, prefix, model,
                char2idx, beam_size, device, max_len):
    beams = [(0.0, "", _encode_prefix(prefix_syllables, prefix, char2idx))]
    used_syllables = []

    for syllable in syllables:
        candidates = list(dict.fromkeys(
            c for c in PINYIN_MAP.get(syllable, []) if c in char2idx
        ))
        if not candidates:
            continue
        used_syllables.append(syllable)

        syl_ids = [char2idx[c] for c in syllable if c in char2idx] + [char2idx[SEP]]

        contexts = []
        for _, _, ids in beams:
            full = ids + syl_ids
            if len(full) > max_len:
                full = full[-max_len:]
            contexts.append(full)

        x = torch.tensor(contexts, dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(x)
        log_probs = F.log_softmax(logits[:, -1, :], dim=-1)

        cand_ids = torch.tensor(
            [char2idx[c] for c in candidates], dtype=torch.long, device=device
        )
        cand_logp = log_probs[:, cand_ids]

        new_beams = []
        for b, (score, partial, ids) in enumerate(beams):
            for j, char in enumerate(candidates):
                new_beams.append((
                    score + cand_logp[b, j].item(),
                    partial + char,
                    contexts[b] + [char2idx[char]],
                ))
        new_beams.sort(key=lambda item: item[0], reverse=True)
        beams = new_beams[:beam_size]

    return [(score, text) for score, text, _ in beams], used_syllables


# ─────────────────────── 交互主循环 ───────────────────────

def run(model, char2idx, topk, beam_size, device, max_len):
    print("=" * 52)
    print("  拼音输入法（Transformer 字符级语言模型）")
    print("  输入拼音回车 → 选候选编号追加到已输入文字")
    print("  r = 重置  q = 退出")
    print("=" * 52)

    confirmed = ""
    confirmed_syllables = []

    while True:
        print(f"\n已输入: 「{confirmed}」" if confirmed else "\n已输入: （空）")
        raw = input("拼音> ").strip()
        if not raw:
            continue
        if raw == "q":
            print("退出。")
            break
        if raw == "r":
            confirmed = ""
            confirmed_syllables = []
            continue

        syllables = segment(raw)
        if not syllables:
            print("无法识别任何音节，请检查拼音拼写。")
            continue

        print(f"音节: {' '.join(syllables)}")
        results, used = beam_search(
            syllables, confirmed_syllables, confirmed,
            model, char2idx, beam_size, device, max_len,
        )
        if not results:
            print("无候选结果。")
            continue

        print("候选:")
        for i, (score, text) in enumerate(results[:topk]):
            print(f"  [{i}] {text}  ({score:.2f})")

        choice = input("选择编号 (回车跳过): ").strip()
        if choice.isdigit():
            idx = int(choice)
            if 0 <= idx < min(len(results), topk):
                confirmed += results[idx][1]
                confirmed_syllables.extend(used)
            else:
                print("编号超出范围。")


# ─────────────────────── 入口 ───────────────────────

def _torch_load(path, device):
    """兼容新旧 PyTorch 的 torch.load。"""
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="best_model.pt")
    parser.add_argument("--pinyin_map", default="pinyin_map.json")
    parser.add_argument("--corpus", default=None,
                        help="真实中文文本文件（可选，强烈建议提供）")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--beam", type=int, default=10)

    parser.add_argument("--train", action="store_true")
    parser.add_argument("--num_lines", type=int, default=80000)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dim_feedforward", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    global PINYIN_MAP, _SYLLABLES
    PINYIN_MAP = _load_pinyin_map(args.pinyin_map)
    _SYLLABLES = sorted(PINYIN_MAP.keys(), key=len, reverse=True)
    print(f"拼音表: {args.pinyin_map}  ({len(PINYIN_MAP)} 个音节)  device={device}")

    if args.train or not os.path.exists(args.model_path):
        if not args.train:
            print(f"未找到 {args.model_path}，先训练一个 Transformer 模型……")
        train_transformer(PINYIN_MAP, args, device)

    ckpt = _torch_load(args.model_path, device)
    if ckpt.get("model_type") != "transformer":
        raise RuntimeError(
            f"{args.model_path} 不是 Transformer checkpoint，请加 --train 重新训练"
        )
    char2idx = ckpt["char2idx"]
    cfg = ckpt["config"]

    # 修复：dict 解包用 **，并以关键字参数传入
    model_cfg = {**cfg, "dropout": 0.0}
    model = TransformerLM(len(char2idx), **model_cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    max_len = cfg["max_len"]

    print(f"模型: {args.model_path}  (Transformer, 词表 {len(char2idx)} 个 token, "
          f"val_loss={ckpt.get('val_loss', float('nan')):.4f})")

    run(model, char2idx, args.topk, args.beam, device, max_len)


if __name__ == "__main__":
    main()
