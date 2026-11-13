"""
sft_finetune.py —— 基于 transformer_lm.py 的 SFT 微调脚本

目的：用 sft_train.jsonl / sft_val.jsonl 问答对，对 tf_full.pt 预训练模型做指令微调。
格式化：每条问答拼成  instruction + "\n" + output + "\n"，用 "\n"（id=0，已在原词表里）分隔角色。
Loss   ：指令部分 labels=-100（不参与 loss），回答部分正常算 loss（SFT 标准做法）。

用法:
    python sft_finetune.py                           # 默认训练
    python sft_finetune.py --epochs 5 --lr 5e-5      # 训 5 轮
    python sft_finetune.py --generate "计算3+5=" --ckpt sft_best.pt   # 推理
    python sft_finetune.py --seq_len 128             # 拉长序列（注意：原模型位置编码 max_len=64，
                                                      # 超过需要重建位置 Embedding，脚本会自动处理）
"""

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

# 复用 transformer_lm.py 里的模型类
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transformer_lm import TransformerLM   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


# ─────────────────────── 数据加载 ───────────────────────

def load_jsonl(path):
    """读 JSONL，每行一个 {"instruction": "...", "output": "..."}。"""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


class SFTDataset(Dataset):
    """
    SFT 数据集：每条问答拼成 [instruction, \\n, output, \\n] 的 id 序列，
    并构造 loss mask（指令位置 label=-100，回答位置是原字 id）。

    构造逻辑：
      1) 每条问答 → text = instruction + "\\n" + output + "\\n"
      2) 转 ids（词表外字直接丢，沿用原模型策略）
      3) 找第一个 "\\n" 的位置作为 instruction / output 的分界点
      4) 分界点之前的位置 label = -100（指令不参与 loss）
         分界点及之后的位置 label = 下一个字的 id（自回归目标）
      5) 截断 / 补齐到固定长度 seq_len
         - 太长：从开头截，尽量保住后面的 output
         - 太短：x 补 "\\n"(id=0)，y 补 -100（不贡献 loss）
    """

    def __init__(self, rows, char2idx, seq_len):
        self.seq_len = seq_len
        self.char2idx = char2idx
        self.pad_id = char2idx.get("\n", 0)     # "\\n" 是原词表第一个字，id=0

        data = []
        for row in rows:
            ins = row["instruction"]
            out = row["output"]
            ids, labels = self._build_pair(ins, out)
            if len(ids) >= 2:   # 至少能凑出一个预测对
                data.append((ids, labels))

        self.data = data
        print(f"  有效问答对 : {len(data):,}  （词表外字直接丢弃）")

    def _build_pair(self, instruction, output):
        """
        拼成 instruction + "\\n" + output + "\\n"，转 ids + 构造 labels。
        labels 对齐到自回归目标：y[t] = x[t+1]，即「当前输入预测下一个字」。
        """
        # 拼文本：用 "\\n" 分隔，它在原词表里，id=0
        full_text = instruction + "\n" + output + "\n"

        # 转 ids，词表外字跳过
        ids = []
        # 同时找 instruction 结束的位置（第一个 "\\n"）
        ins_end_pos = None
        for ch in full_text:
            if ch not in self.char2idx:
                continue
            ids.append(self.char2idx[ch])
            if ins_end_pos is None and ch == "\n":
                # 这个 "\\n" 是 instruction 和 output 的分界
                # 它的 label 应该是 output 的第一个字（自回归：当前输入预测下一个）
                ins_end_pos = len(ids) - 1   # 分界点的 id 在 ids 里的下标

        # 构造 labels：shift by 1（自回归目标）
        # ids[t] 的监督信号是 ids[t+1]（"给定 ids[0..t]，预测 ids[t+1]"）
        x = ids[:-1]                    # 最后一个字没有"下一个"可预测，丢掉
        y_true = ids[1:]                # 目标：每个位置预测下一个字

        # 构造 loss mask：指令部分 label=-100
        # 分界点 ins_end_pos 之前（含分界点自身）都属于 instruction 区
        # 但 x 比 ids 少一位，所以分界点在 x 里的位置是 ins_end_pos
        # （因为 ids[ins_end_pos] == "\\n"，它在 x 里的下标就是 ins_end_pos）
        labels = list(y_true)
        if ins_end_pos is not None:
            # x[0 : ins_end_pos] 这些位置对应的是 instruction 里的字（含 "\\n"）
            # 它们的 label 要设成 -100
            for i in range(min(ins_end_pos, len(labels))):
                labels[i] = -100
        # 如果没找到 "\\n"（极端情况：instruction 里全是词表外字被丢光了），
        # 就把整条的 label 都设成 -100（这条废了，__getitem__ 里会过滤）

        return x, labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        ids, labels = self.data[i]
        L = len(ids)

        if L >= self.seq_len:
            # 截断：从结尾截（尽量保住 output 部分）
            ids = ids[-self.seq_len:]
            labels = labels[-self.seq_len:]
        else:
            # 补齐：x 补 pad_id（"\\n"），labels 补 -100（不贡献 loss）
            pad_len = self.seq_len - L
            ids = ids + [self.pad_id] * pad_len
            labels = labels + [-100] * pad_len

        return (torch.tensor(ids, dtype=torch.long),
                torch.tensor(labels, dtype=torch.long))


# ─────────────────────── 训练 / 评估 / 推理 ───────────────────────

@torch.no_grad()
def evaluate(model, loader, device):
    """计算验证集 loss 和 PPL。只有 label != -100 的位置参与 loss。"""
    model.eval()
    total, tokens = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)                         # (B, T, V)
        total += float(F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1), reduction="sum", ignore_index=-100))
        tokens += int((y != -100).sum().item())   # 只算参与 loss 的 token 数
    loss = total / max(1, tokens)
    return loss, math.exp(loss)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    print("=" * 50)
    print("SFT 微调：指令微调问答模型")
    print("=" * 50)

    # 1) 加载预训练 checkpoint
    print(f"\n[1/5] 加载预训练模型  {args.ckpt}")
    ck = torch.load(args.ckpt, map_location=device)
    char2idx, idx2char = ck["char2idx"], ck["idx2char"]
    c = ck["args"]
    V = len(char2idx)
    print(f"  词表 {V} 字  预训练超参: d={c['d_model']} layers={c['layers']} heads={c['heads']}")

    # 2) 重建模型（seq_len 可能变了，位置编码 max_len 必须匹配）
    print(f"\n[2/5] 重建模型  seq_len={args.seq_len}")
    # 如果 seq_len 超过原模型位置编码的 max_len(=64)，需要扩大位置 Embedding
    # TransformerLM 里 pos = nn.Embedding(max_len, d_model)，直接把 max_len 改成 args.seq_len
    model = TransformerLM(
        vocab_size=V,
        d_model=c["d_model"],
        n_head=c["heads"],
        n_layer=c["layers"],
        max_len=args.seq_len,                     # 关键：用新 seq_len
        dropout=args.dropout,
        use_pos=not args.no_pos,
        causal=not args.no_mask,
    ).to(device)

    # 加载预训练权重：Embedding 行不变（V 相同），位置 Embedding 可能变长
    state = ck["model_state"]
    # 位置 Embedding 形状检查
    pos_key = "pos.weight" if model.use_pos else None
    model_state = model.state_dict()
    msg = model.load_state_dict(state, strict=False)

    # 处理位置 Embedding 变长：原 ckpt 里 pos.weight 可能只有 64 行，需要扩到 args.seq_len 行
    if model.use_pos and pos_key in state and state[pos_key].size(0) < args.seq_len:
        print(f"  位置 Embedding 扩容: {state[pos_key].size(0)} → {args.seq_len}")
        with torch.no_grad():
            new_pos = model.pos.weight.data       # 已经初始化好了（正态分布 std=0.02）
            new_pos[:state[pos_key].size(0)] = state[pos_key]   # 拷贝预训练的前 64 行
            # 后面新增的行保持随机初始化即可
    else:
        # 位置 Embedding 形状匹配，直接加载
        if pos_key in state:
            model_state[pos_key] = state[pos_key]

    # 其他权重正常加载（Embedding、Transformer Encoder、输出头）
    missing, unexpected = model.load_state_dict(ck["model_state"], strict=False)
    if missing:
        # missing 里可能有 pos.weight（刚手动处理过），只打印其他的
        real_missing = [k for k in missing if not k.startswith("pos")]
        if real_missing:
            print(f"  ⚠ missing keys: {real_missing}")
    if unexpected:
        print(f"  ⚠ unexpected keys: {unexpected}")

    n_par = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {n_par/1e6:.2f}M  设备: {next(model.parameters()).device}")

    # 3) 加载 SFT 数据
    print(f"\n[3/5] 加载 SFT 数据")
    train_rows = load_jsonl(args.train)
    val_rows = load_jsonl(args.val)
    print(f"  训练 {len(train_rows)}  验证 {len(val_rows)}")

    train_ds = SFTDataset(train_rows, char2idx, args.seq_len)
    val_ds = SFTDataset(val_rows, char2idx, args.seq_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    # 4) 优化器
    print(f"\n[4/5] 优化器  AdamW lr={args.lr}  weight_decay=0.01")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # 5) 训练循环
    print(f"\n[5/5] 开始 SFT 微调  epochs={args.epochs}\n")
    best = float("inf")
    print(f"{'Epoch':>6} {'TrainLoss':>10} {'ValLoss':>10} {'ValPPL':>10}")
    print("-" * 40)

    for ep in range(1, args.epochs + 1):
        model.train()
        tot, seen = 0.0, 0
        t0 = time.time()

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            # 训练：同样忽略 label=-100 的位置
            loss = F.cross_entropy(
                model(x).reshape(-1, V), y.reshape(-1),
                ignore_index=-100)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss) * int((y != -100).sum())
            seen += int((y != -100).sum())

        dt = time.time() - t0
        train_loss = tot / max(1, seen)
        val_loss, val_ppl = evaluate(model, val_loader, device)

        mark = ""
        if val_loss < best:
            best = val_loss
            mark = " *"
            torch.save({
                "model_state": model.state_dict(),
                "char2idx": char2idx, "idx2char": idx2char,
                "args": vars(args),
            }, args.save)

        print(f"{ep:>6} {train_loss:>10.4f} {val_loss:>10.4f} {val_ppl:>10.2f}{mark}  ({dt:.1f}s)")

    print(f"\n✅ 最佳验证 loss={best:.4f}  已保存 {args.save}")


@torch.no_grad()
def generate(args):
    """给一条新指令，让模型续写回答。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location=device)
    char2idx, idx2char = ck["char2idx"], ck["idx2char"]
    V = len(char2idx)

    # sft_best.pt 的 args 是本脚本的命令行参数；tf_full.pt 的 args 是 transformer_lm 的。
    # 两种字段名都兼容。
    c = ck["args"]
    d_model     = c.get("d_model", 192)
    heads       = c.get("heads", 4)
    layers      = c.get("layers", 4)
    seq_len     = c.get("seq_len", 64)
    no_pos_flag = c.get("no_pos", False) or c.get("no-pos", False)
    no_mask_flag = c.get("no_mask", False) or c.get("no-mask", False)

    model = TransformerLM(
        vocab_size=V, d_model=d_model, n_head=heads, n_layer=layers,
        max_len=seq_len, dropout=0.0,
        use_pos=not no_pos_flag, causal=not no_mask_flag,
    ).to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()

    # 构造输入：instruction + "\n"，让模型接着生成 output
    prompt = args.generate + "\n"
    ids = [char2idx[ch] for ch in prompt if ch in char2idx]
    if not ids:
        print("输入里没有任何字在词表里，无法生成")
        return

    print(f"问: {args.generate}")
    print(f"答: ", end="", flush=True)

    out = list(ids)
    for _ in range(args.n_chars):
        # 只喂最近 seq_len 个 token
        logits = model(torch.tensor([out[-c["seq_len"]:]], device=device))[0, -1]

        if args.temperature <= 0.01:
            nxt = int(logits.argmax())
        else:
            nxt = int(torch.multinomial(
                F.softmax(logits / args.temperature, dim=-1), 1))

        out.append(nxt)
        ch = idx2char.get(nxt, "?")
        print(ch, end="", flush=True)

        # 遇到第二个 "\n"（output 结束标记）就停
        # 第一个 "\n" 是 prompt 末尾的，output 里再出一个就认为回答完了
        if ch == "\n" and out.count(char2idx.get("\n", 0)) >= 2:
            break

    print()   # 换行


def main():
    ap = argparse.ArgumentParser()

    # 路径
    ap.add_argument("--ckpt", default=os.path.join(HERE, "tf_full.pt"),
                    help="预训练 checkpoint（tf_full.pt 或 sft 续训都可以）")
    # HERE = ...\week4，往上走 4 层才到 badou\course
    course_root = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
    ap.add_argument("--train", default=os.path.join(course_root, "sft_train", "sft_train.jsonl"))
    ap.add_argument("--val",   default=os.path.join(course_root, "sft_train", "sft_val.jsonl"))
    ap.add_argument("--save",  default=os.path.join(HERE, "sft_best.pt"))

    # 超参（SFT 用比预训练小的学习率）
    ap.add_argument("--seq_len",  type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--epochs",    type=int, default=5)
    ap.add_argument("--lr",        type=float, default=5e-5)
    ap.add_argument("--dropout",    type=float, default=0.1)
    ap.add_argument("--seed",      type=int, default=42)
    ap.add_argument("--no-mask",   action="store_true", help="关掉因果掩码（消融用）")
    ap.add_argument("--no-pos",     action="store_true", help="关掉位置编码（消融用）")

    # 生成
    ap.add_argument("--generate", default=None, help="给一条指令，让模型续写回答")
    ap.add_argument("--n_chars",   type=int, default=80)
    ap.add_argument("--temperature", type=float, default=0.8)

    args = ap.parse_args()

    if args.generate:
        generate(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
