import math
import json
import time
import random
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
# D:\...\第五周 大模型初探\sft_train\{sft_train,sft_val}.jsonl
DATA_DIR = SCRIPT_DIR.parents[1] / "sft_train"

# 特殊 token：加在字表最后，各占一个 id（不是被拆成 < | u s e r | > 这些字符）
PAD, USER, ASSISTANT, END = "<|pad|>", "<|user|>", "<|assistant|>", "<|end|>"
SPECIALS = [PAD, USER, ASSISTANT, END]


# ─────────────────────────── 数据 ───────────────────────────

def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_vocab(texts):
    """字表 = 数据里出现过的所有字符 + 4 个特殊 token（训练集和验证集一起建，避免 val 出现 OOV）。"""
    chars = sorted({c for t in texts for c in t})
    char2idx = {c: i for i, c in enumerate(chars)}
    for s in SPECIALS:
        char2idx[s] = len(char2idx)
    idx2char = {i: c for c, i in char2idx.items()}
    return char2idx, idx2char


def encode(text, char2idx):
    # 文本里不会出现 <|...|> 这类特殊符号，所以这里只按字映射；特殊 token 由下面的拼接显式插入
    return [char2idx[c] for c in text if c in char2idx]


class SFTDataset(Dataset):
    """
    一条样本 = prompt_ids + answer_ids：
        prompt = [USER] + 指令字符 + [ASSISTANT]      -> labels 全 -100
        answer = 回答字符 + [END]                     -> labels 保留真实 id
    """

    def __init__(self, rows, char2idx, max_len):
        self.items = []
        self.truncated = 0
        self.char2idx = char2idx
        self.max_len = max_len
        for r in rows:
            ids, n_prompt = self._build(r["instruction"], r["output"])
            self.items.append((ids, n_prompt))

    def _build(self, instruction, output):
        ins_ids = encode(instruction, self.char2idx)
        out_ids = encode(output, self.char2idx) + [self.char2idx[END]]
        ids = [self.char2idx[USER]] + ins_ids + [self.char2idx[ASSISTANT]] + out_ids
        n_prompt = 1 + len(ins_ids) + 1
        # 超长时截回答的尾巴，但一定保住 <|end|>，否则模型学不会「说完要停」
        if len(ids) > self.max_len:
            self.truncated += 1
            keep = self.max_len - n_prompt - 1
            out_ids = out_ids[:max(keep, 1)] + [self.char2idx[END]]
            ids = ids[:n_prompt] + out_ids
        return ids, n_prompt

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        ids, n_prompt = self.items[i]
        labels = [-100] * n_prompt + ids[n_prompt:]
        return torch.tensor(ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


def make_collate(pad_id):
    """把不同长度的样本右侧补齐到本 batch 最长，并给出 key_padding_mask（0=有效，-inf=pad）。"""

    def collate(batch):
        ids_list, labels_list = zip(*batch)
        T = max(len(x) for x in ids_list)
        input_ids = torch.full((len(batch), T), pad_id, dtype=torch.long)
        labels = torch.full((len(batch), T), -100, dtype=torch.long)
        pad_mask = torch.full((len(batch), T), float("-inf"), dtype=torch.float)
        for i, (ids, lab) in enumerate(zip(ids_list, labels_list)):
            n = len(ids)
            input_ids[i, :n] = ids
            labels[i, :n] = lab
            pad_mask[i, :n] = 0.0
        return input_ids, labels, pad_mask

    return collate


# ─────────────────────────── 模型（与第四周 language_model_transformer.py 同一份结构） ───────────────────────────

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, embed_dim, nhead, ff_dim, num_layers,
                 max_len=512, dropout=0.1, tie_weights=True):
        super().__init__()
        if embed_dim % nhead != 0:
            raise ValueError(f"embed_dim({embed_dim}) 必须能被 nhead({nhead}) 整除")
        self.max_len = max_len
        self.tok_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(max_len, embed_dim)
        self.drop = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, dim_feedforward=ff_dim,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)
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
        key = (T, str(device), str(dtype))
        mask = self._mask_cache.get(key)
        if mask is None:
            mask = torch.triu(
                torch.full((T, T), float("-inf"), device=device, dtype=dtype), diagonal=1
            )
            self._mask_cache[key] = mask
        return mask

    def forward(self, x, key_padding_mask=None):
        B, T = x.shape
        if T > self.max_len:
            raise ValueError(f"输入长度 {T} 超过 max_len={self.max_len}")
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.drop(self.tok_emb(x) + self.pos_emb(pos))
        h = self.encoder(h, mask=self._causal_mask(T, x.device, h.dtype),
                         src_key_padding_mask=key_padding_mask)
        h = self.norm(h)
        return self.head(self.drop(h))


# ─────────────────────────── 训练 / 评估 ───────────────────────────

def build_scheduler(optimizer, total_steps, warmup_steps):
    warmup_steps = max(1, min(warmup_steps, total_steps))

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.05, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def run_epoch(model, loader, optimizer, device, train=True, scheduler=None, clip=1.0):
    model.train(train)
    total_loss, total_tokens, correct = 0.0, 0, 0

    for input_ids, labels, pad_mask in loader:
        input_ids, labels = input_ids.to(device), labels.to(device)
        pad_mask = pad_mask.to(device)
        logits = model(input_ids, key_padding_mask=pad_mask)

        # 因果语言模型的错位对齐：位置 t 的 logits 预测的是 t+1 处的 token
        logits = logits[:, :-1, :].reshape(-1, logits.size(-1))
        targets = labels[:, 1:].reshape(-1)

        loss = F.cross_entropy(logits, targets, ignore_index=-100, reduction="sum")
        n = (targets != -100).sum().item()

        if train:
            optimizer.zero_grad(set_to_none=True)
            (loss / max(n, 1)).backward()
            if clip and clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
        else:
            with torch.no_grad():
                pred = logits.argmax(dim=-1)
                correct += ((pred == targets) & (targets != -100)).sum().item()

        total_loss += loss.item()
        total_tokens += n

    if total_tokens == 0:
        return float("inf"), float("inf"), 0.0
    avg_loss = total_loss / total_tokens
    acc = correct / total_tokens if not train else float("nan")
    return avg_loss, math.exp(min(avg_loss, 50.0)), acc


# ─────────────────────────── 生成 ───────────────────────────

@torch.no_grad()
def generate(model, char2idx, idx2char, instruction, device,
             max_new_tokens=80, temperature=0.8, top_k=20, rep_penalty=1.15):
    """按 chat template 拼 prompt，逐 token 自回归采样，遇到 <|end|> 就停。"""
    model.eval()
    prompt = [char2idx[USER]] + encode(instruction, char2idx) + [char2idx[ASSISTANT]]
    ids = list(prompt)
    end_id = char2idx[END]
    for _ in range(max_new_tokens):
        x = torch.tensor([ids[-model.max_len:]], dtype=torch.long, device=device)
        logits = model(x)[0, -1].float()

        # 重复惩罚：已经生成过的 token 压低概率，否则小模型很容易「人工智能人工智能…」死循环
        if rep_penalty and rep_penalty != 1.0:
            for t in set(ids[len(prompt):]):
                logits[t] = logits[t] / rep_penalty if logits[t] > 0 else logits[t] * rep_penalty

        if temperature and temperature > 0:
            logits = logits / temperature
            if top_k and 0 < top_k < logits.numel():
                kth = torch.topk(logits, top_k).values[-1]
                logits[logits < kth] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1).item()
        else:
            nxt = logits.argmax().item()
        if nxt == end_id:
            break
        ids.append(nxt)
    out_ids = ids[len(prompt):]
    return "".join(idx2char[i] for i in out_ids if i in idx2char and idx2char[i] not in SPECIALS)


def demo(model, char2idx, idx2char, val_rows, device, n=6):
    print("\n" + "=" * 70)
    print("  验证集样本：指令 → 参考答案 / 模型输出")
    print("=" * 70)
    for r in val_rows[:n]:
        ans = generate(model, char2idx, idx2char, r["instruction"], device,
                       max_new_tokens=64, temperature=0.0)
        print(f"\n指令: {r['instruction']}")
        print(f"参考: {r['output']}")
        print(f"模型: {ans}")


def chat(model, char2idx, idx2char, device, rep_penalty=1.15, temperature=0.8):
    print("=" * 60)
    print("  SFT 模型问答（q 退出）")
    print("=" * 60)
    while True:
        try:
            q = input("\n指令> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q in ("q", "quit", "exit", ""):
            break
        print("回答>", generate(model, char2idx, idx2char, q, device, max_new_tokens=120,
                              rep_penalty=rep_penalty, temperature=temperature))


# ─────────────────────────── 主函数 ───────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default=str(DATA_DIR / "sft_train.jsonl"))
    parser.add_argument("--val",   default=str(DATA_DIR / "sft_val.jsonl"))
    parser.add_argument("--epochs",     type=int,   default=20)
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--max_len",    type=int,   default=160)
    parser.add_argument("--embed_dim",  type=int,   default=256)
    parser.add_argument("--nhead",      type=int,   default=8)
    parser.add_argument("--ff_dim",     type=int,   default=1024)
    parser.add_argument("--num_layers", type=int,   default=4)
    parser.add_argument("--dropout",    type=float, default=0.1)
    parser.add_argument("--lr",         type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--clip",       type=float, default=1.0)
    parser.add_argument("--warmup",     type=int,   default=100)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--save",       default=str(SCRIPT_DIR / "sft_model.pt"))
    parser.add_argument("--model_path", default=None, help="--demo/--chat 时加载的权重")
    parser.add_argument("--demo",       action="store_true", help="训练后立刻在验证集上抽几条生成")
    parser.add_argument("--chat",       action="store_true", help="交互式问答，不训练")
    parser.add_argument("--rep_penalty", type=float, default=1.15, help="重复惩罚，1.0=关闭")
    parser.add_argument("--temperature", type=float, default=0.8, help="--chat 时的采样温度")
    parser.add_argument("--early_stop", type=int,   default=6, help="验证集连续多少轮不降就停，0=不早停")
    args = parser.parse_args()
    model_path = args.model_path or args.save

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── 1. 数据 ──
    train_rows = load_jsonl(args.train)
    val_rows = load_jsonl(args.val)
    char2idx, idx2char = build_vocab(
        [r["instruction"] + r["output"] for r in train_rows + val_rows]
    )
    vocab_size = len(char2idx)

    train_ds = SFTDataset(train_rows, char2idx, args.max_len)
    val_ds = SFTDataset(val_rows, char2idx, args.max_len)

    # ── 2. 只加载权重模式 ──
    if args.chat:
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        cfg = ckpt["args"]
        model = TransformerLM(
            vocab_size=len(ckpt["char2idx"]), embed_dim=cfg["embed_dim"], nhead=cfg["nhead"],
            ff_dim=cfg["ff_dim"], num_layers=cfg["num_layers"],
            max_len=cfg["max_len"], dropout=0.0, tie_weights=not cfg.get("no_tie", False),
        ).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        chat(model, ckpt["char2idx"], ckpt["idx2char"], device,
             rep_penalty=args.rep_penalty, temperature=args.temperature)
        return

    print(f"device: {device}")
    print(f"训练样本: {len(train_ds):,}   验证样本: {len(val_ds):,}   词表: {vocab_size}（含 4 个特殊 token）")
    if train_ds.truncated or val_ds.truncated:
        print(f"[提示] 有 {train_ds.truncated + val_ds.truncated} 条超长样本被截断到 max_len={args.max_len}")

    # 打印监督比例 —— 直观感受 Loss Mask 到底屏蔽了多少 token
    n_sup = sum(int((lab != -100).sum()) for _, lab in train_ds)
    n_all = sum(len(ids) for ids, _ in train_ds.items)
    print(f"监督 token 占比: {n_sup / n_all:.1%}（其余 {1 - n_sup / n_all:.1%} 是 prompt/pad，不参与 loss）")

    collate = make_collate(char2idx[PAD])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              drop_last=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            drop_last=False, collate_fn=collate)

    # ── 3. 模型 ──
    model = TransformerLM(
        vocab_size=vocab_size, embed_dim=args.embed_dim, nhead=args.nhead,
        ff_dim=args.ff_dim, num_layers=args.num_layers, max_len=args.max_len,
        dropout=args.dropout, tie_weights=True,
    ).to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    scheduler = build_scheduler(optimizer, total_steps, args.warmup)
    print(f"每轮步数: {len(train_loader)}   总步数: {total_steps}")

    best_val = float("inf")
    bad_epochs = 0

    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Train PPL':>10}  {'Val Loss':>10}  "
          f"{'Val PPL':>10}  {'Val Acc':>8}")
    print("-" * 68)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_ppl, _ = run_epoch(model, train_loader, optimizer, device,
                                       train=True, scheduler=scheduler, clip=args.clip)
        with torch.no_grad():
            va_loss, va_ppl, va_acc = run_epoch(model, val_loader, optimizer, device,
                                                train=False, clip=args.clip)

        marker = "  *" if va_loss < best_val else ""
        if va_loss < best_val:
            best_val = va_loss
            bad_epochs = 0
            torch.save({
                "model_state": model.state_dict(),
                "char2idx": char2idx,
                "idx2char": idx2char,
                "args": {**vars(args), "arch": "transformer", "model": "transformer"},
                "special_tokens": {"pad": PAD, "user": USER, "assistant": ASSISTANT, "end": END},
                "template": "<|user|>{instruction}<|assistant|>{output}<|end|>",
            }, model_path)
        else:
            bad_epochs += 1

        print(f"{epoch:>6}  {tr_loss:>10.4f}  {tr_ppl:>10.2f}  {va_loss:>10.4f}  "
              f"{va_ppl:>10.2f}  {va_acc:>7.1%}{marker}  ({time.time() - t0:.0f}s)")

        if args.early_stop and bad_epochs >= args.early_stop:
            print(f"验证集连续 {bad_epochs} 轮不降，提前停止（最好 Val Loss {best_val:.4f}）。")
            break

    print(f"\n训练完成。最佳验证 PPL: {math.exp(min(best_val, 50)):.2f}  已保存至 {model_path}")

    if args.demo:
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        demo(model, ckpt["char2idx"], ckpt["idx2char"], val_rows, device)


if __name__ == "__main__":
    main()
