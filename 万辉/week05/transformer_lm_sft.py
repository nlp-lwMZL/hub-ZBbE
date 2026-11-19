import argparse
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

# ================= 定义统一的 Prompt 模板 =================
PROMPT_TEMPLATE = "### 问：{}\n### 答：{}"
INFERENCE_TEMPLATE = "### 问：{}\n### 答："


# ─────────────────────── 数据 ───────────────────────

def load_sft_jsonl(file_path, char2idx):
    texts = []
    dropped_chars = 0
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            instruction = data.get("instruction", "")
            output = data.get("output", "")
            text = PROMPT_TEMPLATE.format(instruction, output) + "\n\n"

            clean_text = ""
            for c in text:
                if c in char2idx:
                    clean_text += c
                else:
                    dropped_chars += 1
            texts.append(clean_text)

    print(f"读取 {file_path} 完成。丢弃未登录字符数: {dropped_chars}")
    return "".join(texts)


class CharDataset(Dataset):
    def __init__(self, text, char2idx, seq_len, stride):
        ids = [char2idx[c] for c in text if c in char2idx]
        self.data = torch.tensor(ids, dtype=torch.long)
        self.seq_len = seq_len
        self.stride = max(1, stride)

    def __len__(self):
        return max(0, (len(self.data) - self.seq_len - 1) // self.stride + 1)

    def __getitem__(self, i):
        s = i * self.stride
        return self.data[s:s + self.seq_len], self.data[s + 1:s + self.seq_len + 1]


# ─────────────────────── 模型结构 ───────────────────────

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model=192, n_head=4, n_layer=4, max_len=512,
                 dropout=0.1, use_pos=True, causal=True):
        super().__init__()
        self.causal = causal
        self.use_pos = use_pos
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model) if use_pos else None
        self.drop = nn.Dropout(dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head, dim_feedforward=4 * d_model,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=n_layer, norm=nn.LayerNorm(d_model),
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
                idx.size(1), device=idx.device)
        x = self.encoder(x, mask=mask)
        return self.head(x)


# ─────────────────────── 评估与训练 ───────────────────────

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total, tokens = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total += float(F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                       y.reshape(-1), reduction="sum"))
        tokens += y.numel()
    loss = total / tokens
    return loss, math.exp(loss)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    print(f"正在加载预训练模型: {args.pretrained}")
    if not os.path.exists(args.pretrained):
        raise FileNotFoundError(f"找不到预训练模型 {args.pretrained}，请确保路径正确！")

    ck = torch.load(args.pretrained, map_location="cpu")
    char2idx, idx2char = ck["char2idx"], ck["idx2char"]
    old_args = ck["args"]
    V = len(char2idx)
    print(f"加载成功！词表大小: {V}")

    # 强制对齐 seq_len 和 stride，防止位置编码越界！
    args.seq_len = old_args["seq_len"]
    args.stride = args.seq_len

    text_tr = load_sft_jsonl(args.train_corpus, char2idx)
    text_va = load_sft_jsonl(args.val_corpus, char2idx)

    tr = CharDataset(text_tr, char2idx, args.seq_len, args.stride)
    va = CharDataset(text_va, char2idx, args.seq_len, args.stride)

    train_loader = DataLoader(tr, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(va, batch_size=args.batch_size, shuffle=False, drop_last=True)

    model = TransformerLM(
        vocab_size=V,
        d_model=old_args["d_model"],
        n_head=old_args["heads"],
        n_layer=old_args["layers"],
        max_len=old_args["seq_len"],
        dropout=args.dropout,
        use_pos=not old_args.get("no_pos", False),
        causal=not old_args.get("no_mask", False)
    ).to(device)

    model.load_state_dict(ck["model_state"], strict=False)
    print("预训练权重注入成功，开始 SFT 微调...")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    best = float("inf")

    print(f"\n{'Epoch':>6} {'TrainLoss':>10} {'ValPPL':>10} {'tok/s':>9}")
    print("-" * 45)

    for ep in range(1, args.epochs + 1):
        model.train()
        st, seen, tot = time.time(), 0, 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(model(x).reshape(-1, V), y.reshape(-1))

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            tot += float(loss) * y.numel()
            seen += y.numel()

        dt = time.time() - st
        _, v_ppl = evaluate(model, val_loader, device)

        mark = ""
        if v_ppl < best:
            best = v_ppl
            mark = " *"

            # 【修复保存逻辑】把预训练模型的参数补全进当前的 args，防止后续推理报错
            args.d_model = old_args.get("d_model", 192)
            args.heads = old_args.get("heads", 4)
            args.layers = old_args.get("layers", 4)
            args.no_pos = old_args.get("no_pos", False)
            args.no_mask = old_args.get("no_mask", False)

            torch.save({"model_state": model.state_dict(), "char2idx": char2idx,
                        "idx2char": idx2char, "args": vars(args)}, args.save)

        print(f"{ep:>6} {tot / seen:>10.4f} {v_ppl:>10.2f} {seen / dt:>9,.0f}{mark}")

    print(f"\n微调完成！最佳验证 PPL: {best:.2f}，已保存至 {args.save}")


@torch.no_grad()
def generate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"找不到模型 {args.ckpt}")

    ck = torch.load(args.ckpt, map_location=device)
    c = ck["args"]
    char2idx, idx2char = ck["char2idx"], ck["idx2char"]

    if "d_model" not in c:
        pre_ck_path = os.path.join(HERE, "tf_best.pt")
        if os.path.exists(pre_ck_path):
            pre_c = torch.load(pre_ck_path, map_location="cpu")["args"]
            c.update(pre_c)
        else:
            c.update({"d_model": 192, "heads": 4, "layers": 4, "seq_len": 64, "no_pos": False, "no_mask": False})

    model = TransformerLM(
        len(char2idx), c["d_model"], c["heads"], c["layers"], c["seq_len"], 0.0,
        use_pos=not c.get("no_pos", False),
        causal=not c.get("no_mask", False)
    ).to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()

    prompt_text = INFERENCE_TEMPLATE.format(args.generate)
    ids = [char2idx[ch] for ch in prompt_text if ch in char2idx]
    out = list(ids)

    # 记录生成的文本，方便实时截断
    generated_text = ""

    for _ in range(args.n_chars):
        logits = model(torch.tensor([out[-c["seq_len"]:]], device=device))[0, -1]

        if args.temperature <= 0.01:
            nxt = int(logits.argmax())
        else:
            nxt = int(torch.multinomial(F.softmax(logits / args.temperature, dim=-1), 1))

        out.append(nxt)
        char_out = idx2char[nxt]
        generated_text += char_out

        # ================= 强力截断逻辑 =================
        # 1. 如果生成了连续的两个换行符（代表一个样本结束）
        if "\n\n" in generated_text:
            generated_text = generated_text.split("\n\n")[0]
            break

        # 2. 如果模型试图开始生成下一道题（出现 "### 问："）
        if "### 问：" in generated_text:
            generated_text = generated_text.split("### 问：")[0]
            break

        # 3. 如果模型试图在回答里再次生成 "答："（说明它在复读格式）
        if generated_text.count("答：") > 1:
            # 截断到第二个 "答：" 之前
            generated_text = generated_text[:generated_text.rfind("答：")]
            break
        # =================================================

    print(f"\n[Instruction]: {args.generate}")
    print(f"[Answer]: {generated_text.strip()}")


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--train_corpus", default=os.path.join(HERE, "./sft_train/sft_train.jsonl"), help="SFT训练集")
    ap.add_argument("--val_corpus", default=os.path.join(HERE, "./sft_train/sft_val.jsonl"), help="SFT验证集")
    ap.add_argument("--pretrained", default=os.path.join(HERE, "tf_best.pt"), help="预训练模型路径")

    ap.add_argument("--seq_len", type=int, default=64)
    ap.add_argument("--stride", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save", default="sft_best.pt")

    ap.add_argument("--generate", default=None, help="给一段 instruction，进行推理")
    ap.add_argument("--ckpt", default="sft_best.pt", help="推理加载的模型")
    ap.add_argument("--n_chars", type=int, default=100)
    ap.add_argument("--temperature", type=float, default=0.7)

    args = ap.parse_args()

    if args.generate:
        generate(args)
    else:
        train(args)


if __name__ == "__main__":
    main()