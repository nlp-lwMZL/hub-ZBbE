"""
Transformer 语言模型 SFT 脚本。

数据格式：  指令 <sep> 回答 <eos>，只对「回答 + <eos>」计算 loss。
预训练权重：默认尝试加载 week04 的 best_model_transformer.pt（仅读取权重文件，不引用其代码）；
            文件不存在或加 --scratch 时从零训练。
用法：
  python3 sft_lm.py --epochs 20                训练
  python3 sft_lm.py --chat "指令内容"           加载 sft_lm.pt 单条生成
  python3 sft_lm.py --chat                      交互模式
"""

import os
import json
import math
import random
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PRETRAINED = os.path.join(HERE, "..", "week04", "循环神经网络语言模型", "best_model_transformer.pt")
IGNORE = -100
PAD, SEP, EOS = "<pad>", "<sep>", "<eos>"


# ─────────────────────────── 模型（改写自 transformer_lm.py） ───────────────────────────

class CausalSelfAttention(nn.Module):
    """多头因果自注意力：位置 t 只能看到 <= t 的字符。"""

    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"
        self.n_heads = n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = dropout
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = (t.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) for t in (q, k, v))
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(out))


class Block(nn.Module):
    """Pre-LN Transformer 块。"""

    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class TransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, num_layers, d_ff, max_len, dropout):
        super().__init__()
        self.max_len = max_len
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(Block(d_model, n_heads, d_ff, dropout) for _ in range(num_layers))
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok_embed.weight             # 输入输出嵌入共享
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device)
        h = self.drop(self.tok_embed(x) + self.pos_embed(pos))
        for block in self.blocks:
            h = block(h)
        return self.head(self.ln_f(h))

    @torch.no_grad()
    def generate(self, idx, eos_id, max_new_tokens=120, temperature=1.0, top_k=1):
        """自回归生成：top_k=1 为贪心；遇到 <eos> 停止。返回新生成的 id 列表。"""
        self.eval()
        out = []
        for _ in range(max_new_tokens):
            logits = self(idx[:, -self.max_len:])[:, -1, :] / max(temperature, 1e-6)
            if top_k > 1:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
                nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)
            else:
                nxt = logits.argmax(-1, keepdim=True)
            if nxt.item() == eos_id:
                break
            out.append(nxt.item())
            idx = torch.cat([idx, nxt], dim=1)
        return out


def make_scheduler(optimizer, total_steps, warmup_steps):
    """线性 warmup + 余弦衰减。"""
    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ─────────────────────────── SFT 数据 ───────────────────────────

def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def extend_vocab(char2idx, rows):
    """在已有字表后追加特殊符号和 SFT 数据里的新字，旧字 id 不变。"""
    c2i = dict(char2idx)
    for t in (PAD, SEP, EOS):
        c2i.setdefault(t, len(c2i))
    for r in rows:
        for ch in r["instruction"] + r["output"]:
            c2i.setdefault(ch, len(c2i))
    return c2i, {i: c for c, i in c2i.items()}


def encode(c2i, instruction, output=None):
    prompt = [c2i[c] for c in instruction if c in c2i] + [c2i[SEP]]
    if output is None:
        return prompt
    return prompt, [c2i[c] for c in output if c in c2i] + [c2i[EOS]]


class SFTDataset(Dataset):
    def __init__(self, rows, c2i, max_len):
        self.items = []
        for r in rows:
            p, a = encode(c2i, r["instruction"], r["output"])
            self.items.append(((p + a)[:max_len], ([IGNORE] * len(p) + a)[:max_len]))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def make_collate(pad_id):
    def collate(batch):
        n = max(len(x) for x, _ in batch)
        ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
        lab = torch.full((len(batch), n), IGNORE, dtype=torch.long)
        for i, (x, y) in enumerate(batch):
            ids[i, :len(x)] = torch.tensor(x)
            lab[i, :len(y)] = torch.tensor(y)
        return ids, lab
    return collate


# ─────────────────────────── 训练 / 评估 / 生成 ───────────────────────────

def build_model(ckpt, vocab_size, max_len, dropout, args):
    """有预训练 ckpt 时沿用其结构并拷贝权重（扩充字表/位置编码）；否则按命令行参数新建。"""
    if ckpt is None:
        return TransformerLM(vocab_size, args.d_model, args.n_heads, args.num_layers,
                             args.d_ff, max_len, dropout)
    a = ckpt["args"]
    model = TransformerLM(vocab_size, a["d_model"], a["n_heads"], a["num_layers"],
                          a["d_ff"], max_len, dropout)
    sd, new_sd = ckpt["model_state"], model.state_dict()
    for k, v in sd.items():
        if k == "tok_embed.weight":
            new_sd[k][:v.size(0)] = v          # 新增 token 保持随机初始化
        elif k == "pos_embed.weight":
            new_sd[k][:v.size(0)] = v          # 超出原长度的位置保持随机初始化
        elif k != "head.weight":               # head 与 tok_embed 共享
            new_sd[k] = v
    model.load_state_dict(new_sd)
    return model


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    tot, cnt = 0.0, 0
    for ids, lab in loader:
        ids, lab = ids.to(device), lab.to(device)
        logits, tgt = model(ids)[:, :-1], lab[:, 1:]
        tot += F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1),
                               ignore_index=IGNORE, reduction="sum").item()
        cnt += (tgt != IGNORE).sum().item()
    return tot / cnt


def answer(model, c2i, i2c, instruction, device, **kw):
    idx = torch.tensor([encode(c2i, instruction)], device=device)
    return "".join(i2c[i] for i in model.generate(idx, c2i[EOS], **kw))


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = get_device()

    train_rows = load_jsonl(os.path.join(HERE, "sft_train", "sft_train.jsonl"))
    val_rows = load_jsonl(os.path.join(HERE, "sft_train", "sft_val.jsonl"))

    ckpt = None
    if not args.scratch and os.path.exists(args.pretrained):
        ckpt = torch.load(args.pretrained, map_location="cpu")
    print(f"device: {device}  预训练权重: {'是' if ckpt else '否（从零训练）'}")

    base_vocab = ckpt["char2idx"] if ckpt else {}
    c2i, i2c = extend_vocab(base_vocab, train_rows + val_rows)
    model = build_model(ckpt, len(c2i), args.max_len, args.dropout, args).to(device)
    print(f"字表: {len(c2i)}  参数量: {sum(p.numel() for p in model.parameters()):,}")

    collate = make_collate(c2i[PAD])
    train_dl = DataLoader(SFTDataset(train_rows, c2i, args.max_len), batch_size=args.batch_size,
                          shuffle=True, collate_fn=collate)
    val_dl = DataLoader(SFTDataset(val_rows, c2i, args.max_len), batch_size=64, collate_fn=collate)

    samples = val_rows[:8]
    v0 = evaluate(model, val_dl, device)
    print(f"[SFT 前] val loss {v0:.4f}  PPL {math.exp(v0):.2f}")
    before = [answer(model, c2i, i2c, r["instruction"], device) for r in samples]

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = make_scheduler(opt, args.epochs * len(train_dl), args.warmup)

    best = float("inf")
    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Val Loss':>10}  {'Val PPL':>10}")
    for ep in range(1, args.epochs + 1):
        model.train()
        tot, n = 0.0, 0
        for ids, lab in train_dl:
            ids, lab = ids.to(device), lab.to(device)
            logits = model(ids)[:, :-1]
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), lab[:, 1:].reshape(-1),
                                   ignore_index=IGNORE)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item(); n += 1
        v = evaluate(model, val_dl, device)
        mark = ""
        if v < best:
            best, mark = v, "  *"
            torch.save({"model_state": model.state_dict(), "char2idx": c2i, "idx2char": i2c,
                        "model_args": {"d_model": model.tok_embed.embedding_dim,
                                       "n_heads": model.blocks[0].attn.n_heads,
                                       "num_layers": len(model.blocks),
                                       "d_ff": model.blocks[0].ffn[0].out_features,
                                       "max_len": model.max_len}}, args.save)
        print(f"{ep:>6}  {tot / n:>10.4f}  {v:>10.4f}  {math.exp(v):>10.2f}{mark}", flush=True)

    model.load_state_dict(torch.load(args.save, map_location=device)["model_state"])
    after = [answer(model, c2i, i2c, r["instruction"], device) for r in samples]
    lines = [f"SFT 前 val PPL {math.exp(v0):.2f}；SFT 后最佳 val PPL {math.exp(best):.2f}\n"]
    for r, b, a in zip(samples, before, after):
        lines.append(f"【指令】{r['instruction']}\n【参考】{r['output']}\n【SFT前】{b}\n【SFT后】{a}\n")
    report = "\n".join(lines)
    print("\n" + report)
    with open(os.path.join(HERE, "sft_lm_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)


def chat(args):
    device = get_device()
    ck = torch.load(args.save, map_location=device)
    m = ck["model_args"]
    model = TransformerLM(len(ck["char2idx"]), m["d_model"], m["n_heads"], m["num_layers"],
                          m["d_ff"], m["max_len"], 0.0).to(device)
    model.load_state_dict(ck["model_state"])
    c2i, i2c = ck["char2idx"], ck["idx2char"]
    kw = dict(max_new_tokens=args.max_new, top_k=args.top_k, temperature=args.temperature)
    if args.chat != "":
        print(answer(model, c2i, i2c, args.chat, device, **kw))
        return
    while True:
        ins = input("指令> ").strip()
        if ins in ("", "q", "quit"):
            break
        print(answer(model, c2i, i2c, ins, device, **kw), "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chat", nargs="?", const="", default=None, help="推理模式；可跟一条指令，不跟则交互")
    ap.add_argument("--pretrained", default=DEFAULT_PRETRAINED)
    ap.add_argument("--scratch", action="store_true", help="不加载预训练权重")
    ap.add_argument("--save", default=os.path.join(HERE, "sft_lm.pt"))
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--max_len", type=int, default=160)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    # 从零训练时的模型结构（加载预训练权重时沿用权重的结构）
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--num_layers", type=int, default=4)
    ap.add_argument("--d_ff", type=int, default=1024)
    # 生成参数
    ap.add_argument("--max_new", type=int, default=120)
    ap.add_argument("--top_k", type=int, default=1, help="1=贪心；>1 采样")
    ap.add_argument("--temperature", type=float, default=1.0)
    args = ap.parse_args()

    if args.chat is not None:
        chat(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
