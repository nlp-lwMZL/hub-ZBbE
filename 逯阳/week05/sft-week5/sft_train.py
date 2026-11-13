"""
SFT（监督微调）训练脚本 —— 在预训练好的字符级 Transformer 上，教模型「怎么回答」。

核心原则：**只改数据、loss 和 chat 模板，模型结构一个新层都不加。**

    预训练：<bos> 春天来了花就开了 <eos>          → 学会「中文长什么样」
    SFT   ：<bos> <|user|> 什么是AI <|assistant|> AI是智能 <|end|>
                                                  → 学会「该用什么姿势回答」

两者的训练目标完全一样，都是 next-token prediction；区别只在两处：
    1) 数据换成了 (指令, 回答) 对，并用 chat 模板拼接
    2) 只对「回答段」计算 loss，问句段全部打成 -100

Loss Mask 的直观解释（对应 -- 你贴的那张图）:
    ids   : <|user|>  什 么 是 AI  <|asst|>  AI  是  智 能  <|end|>
    labels:   -100    -100-100-100-100   -100   AI  是  智  能  <|end|>
              └────── 不计算 loss ──────┘   └──── 计算 loss ────┘

    为什么：我们只想让它学会「回答的动作」，不想让它把用户的提问背下来。
    实测这条数据上，有监督 token 约占 50%。

用法:
    python sft_train.py                                  # 数据默认在 ./sft_train/
    python sft_train.py --epochs 15 --lr 2e-4
    python sft_train.py --init_model best_transformer.pt --save sft_transformer.pt
    python sft_train.py --max_samples 200 --epochs 1     # 冒烟测试

路径约定:
    本文件所在的 sft-week5 目录 = SFT 工程根目录
        数据  : <SFT 目录>/sft_train/*.jsonl
        权重  : --save 的相对路径一律落到 <SFT 目录>
    模型定义（TransformerLM / build_model）和预训练权重仍在
    D:\\project\\循环神经网络语言模型\\，通过 --code_dir 挂进 sys.path 复用，
    不复制一份结构定义 —— 复制两份迟早会漂移。
    --init_model 的查找顺序：cwd → SFT 目录 → code_dir
"""

import argparse
import json
import math
import os
import random
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# SFT 工程根目录：数据、权重、日志都跟着它走
SFT_DIR = os.path.dirname(os.path.abspath(__file__))

# 预训练工程目录：TransformerLM 定义、generate.py、best_transformer.pt 都在那边。
# 代码分两个工程放，模型定义只保留一份，靠 sys.path 复用。
DEFAULT_CODE_DIR = r"D:\project\循环神经网络语言模型"


def _prescan(flag, default):
    """
    在 argparse 之前先把 --code_dir 读出来 —— 因为顶层 import 就需要它，
    等 argparse 解析完再 import 就太晚了。
    """
    argv = sys.argv[1:]
    for i, a in enumerate(argv):
        if a == flag:
            return argv[i + 1] if i + 1 < len(argv) else default
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
    return default


CODE_DIR = _prescan("--code_dir", DEFAULT_CODE_DIR)
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

# 复用预训练脚本里的模型定义与工具，保证两边结构逐字段一致（SFT 绝不改结构）
from transformer_lm import (          # noqa: E402
    TransformerLM, BOS, make_scheduler, _fmt_sec,
)


def resolve_input(path):
    """
    输入文件查找顺序：绝对路径 → cwd → SFT 目录 → 预训练工程目录。
    比预训练脚本多一层 code_dir，这样 --init_model best_transformer.pt
    不用写全路径也能找到那边的权重。
    """
    if os.path.isabs(path):
        return path
    for d in (os.getcwd(), SFT_DIR, CODE_DIR):
        p = os.path.join(d, path)
        if os.path.exists(p):
            return p
    return path


def resolve_output(path):
    """输出（权重）固定落到 SFT 目录，相对路径不跟着 cwd 跑。"""
    return path if os.path.isabs(path) else os.path.join(SFT_DIR, path)


# ─────────────────────── chat 模板 ───────────────────────
# 角色标记让模型知道「谁在说话、说到哪算完」。
# 这里用多字符字符串当 token，和 <bos>/<eos> 一样，永远不会和正文里的字撞车。
U_TAG, A_TAG, END_TAG, PAD = "<|user|>", "<|assistant|>", "<|end|>", "<pad>"
CHAT_SPECIALS = [U_TAG, A_TAG, END_TAG, PAD]

# 默认数据目录：用户指定的下载目录
DEFAULT_DATA_DIR = os.path.join(SFT_DIR, "sft_train")


# ─────────────────────── 数据 ───────────────────────

def load_jsonl(path):
    """
    读 jsonl，兼容几种常见字段名。
    返回 [(instruction, output), ...]，空行/残缺样本直接丢掉。
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到数据文件: {path}")
    records = []
    bad = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            q = obj.get("instruction") or obj.get("question") or obj.get("input") or ""
            a = obj.get("output") or obj.get("answer") or obj.get("response") or ""
            q, a = str(q).strip(), str(a).strip()
            if q and a:
                records.append((q, a))
            else:
                bad += 1
    if not records:
        raise ValueError(f"{path} 里没有可用的 (instruction, output) 样本。")
    if bad:
        print(f"  ({os.path.basename(path)}: 跳过 {bad} 条空行/缺字段的样本)")
    return records


def build_chat_vocab(base_char2idx, records):
    """
    在预训练词表的基础上扩表，只做「加 token」，不动任何已有 id。

    加两类：
        1) SFT 数据里出现、但预训练词表没有的字（本数据集 107 个，占频次 5.3%）
           —— 不补的话这些字会被静默丢掉，问句和答案都会残缺
        2) chat 模板的 4 个特殊 token：<|user|> <|assistant|> <|end|> <pad>

    已有 id 全部保持不变，所以预训练权重可以逐行原样搬过来，只有新增的行是新的。
    """
    seen = set(base_char2idx)
    oov = set()
    for q, a in records:
        oov.update(q)
        oov.update(a)
    oov -= seen

    chars = list(base_char2idx.keys())          # 顺序即 id 顺序，前 len(base) 个必须不动
    chars += sorted(oov)                        # 新字符排序，保证每次运行 id 一致
    chars += CHAT_SPECIALS

    char2idx = {c: i for i, c in enumerate(chars)}
    idx2char = {i: c for c, i in char2idx.items()}
    return char2idx, idx2char, sorted(oov)


def encode(text, char2idx):
    """文字 → id。此时词表已覆盖 SFT 全部字符，所以不会丢字（丢字会直接报错）。"""
    out = []
    for c in text:
        i = char2idx.get(c)
        if i is None:
            raise KeyError(f"字符 {c!r} 不在词表中，build_chat_vocab 漏了它")
        out.append(i)
    return out


def render_chat(instruction, output, char2idx):
    """
    拼一条训练样本，并同时给出 loss mask。

    结构（与用户给的模板一致）:
        <bos> <|user|> {instruction} <|assistant|> {output} <|end|>

    labels 在 prompt 段（直到 <|assistant|> 为止）全部是 -100，
    回答段（含收尾的 <|end|>）保留真实 id。

    注意角色标记是「一个 token」而不是 4 个字符：这里直接查表取 id，
    不能把 "<|user|>" 拆成 '<' '|' 'u' ... 逐字编码。
    """
    prompt_ids = ([char2idx[BOS]] if BOS in char2idx else []) + [char2idx[U_TAG]] \
                 + encode(instruction, char2idx) + [char2idx[A_TAG]]
    answer_ids = encode(output, char2idx) + [char2idx[END_TAG]]

    ids = prompt_ids + answer_ids
    labels = [-100] * len(prompt_ids) + answer_ids
    return ids, labels


class SFTDataset(Dataset):
    """
    每条样本 = 一个完整对话（单轮）。
    长度不齐靠 collate 动态 padding，不预先截到固定 seq_len —— 数据最长才 114，
    硬截到 128 会白算 40% 的 pad 位置。

    max_len 只用来兜底：超长样本把「回答段」尾部截掉，保住问句的完整性
    （问句被截断，模型学到的就是残缺指令，比答案被截更糟）。
    """

    def __init__(self, records, char2idx, max_len):
        self.items = []
        self.n_truncated = 0
        self.n_dropped = 0
        end_id = char2idx[END_TAG]
        for q, a in records:
            ids, labels = render_chat(q, a, char2idx)
            if len(ids) > max_len:
                prompt_len = sum(1 for v in labels if v == -100)
                if prompt_len + 2 <= max_len:
                    # 砍答案的尾巴，但补一个 <|end|> 上去 ——
                    # 否则模型会学到「话说一半就没了」，收尾反而学坏
                    ids = ids[:max_len - 1] + [end_id]
                    labels = labels[:max_len - 1] + [end_id]
                    self.n_truncated += 1
                else:
                    self.n_dropped += 1
                    continue
            self.items.append((ids, labels))

        if not self.items:
            raise ValueError("所有样本都超长被丢掉了，请调大 --max_len。")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        ids, labels = self.items[idx]
        return torch.tensor(ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

    def supervised_ratio(self):
        """有监督 token 占比 —— SFT 一般落在 40%~60%，明显偏高说明 mask 没生效。"""
        sup = sum(1 for _, labels in self.items for v in labels if v != -100)
        tot = sum(len(labels) for _, labels in self.items)
        return sup / max(1, tot)


def make_collate(pad_id):
    """
    动态 padding 到 batch 内最长，右侧补 <pad>，labels 补 -100。

    注意这里**不传 key_padding_mask**：padding 全在右边，而因果掩码已经保证
    「位置 i 只能看见 j <= i」，所以真实 token 永远看不到右边的 pad，
    右边 pad 也就不可能影响 loss。少传一个 mask 还顺手躲开了
    「整行被 mask 掉 → softmax 全 -inf → NaN」这个坑。
    """
    def collate(batch):
        max_len = max(len(ids) for ids, _ in batch)
        B = len(batch)
        x = torch.full((B, max_len), pad_id, dtype=torch.long)
        y = torch.full((B, max_len), -100, dtype=torch.long)
        for i, (ids, labels) in enumerate(batch):
            n = len(ids)
            x[i, :n] = ids
            y[i, :n] = labels
        return x, y
    return collate


# ─────────────────────── 词表扩容 ───────────────────────

def resize_vocab(model, new_vocab):
    """
    把 tok_embed / fc 从旧词表大小扩到 new_vocab，旧行原样保留。

    新行用「已有行的均值」初始化，而不是随机初始化：
    随机初始化会让新 token 一上来就拿到离谱的 logit（可能把 softmax 顶歪），
    均值初始化则让它们起步时和普通字表现一致，再由 SFT 慢慢学。
    这是结构上唯一的改动，且只是 embedding 矩阵变宽，Transformer 层一个都没动。
    """
    old_v = model.tok_embed.weight.size(0)
    if new_vocab <= old_v:
        return 0

    device, dtype = model.tok_embed.weight.device, model.tok_embed.weight.dtype
    dim = model.tok_embed.embedding_dim

    old_emb = model.tok_embed
    new_emb = nn.Embedding(new_vocab, dim).to(device=device, dtype=dtype)
    with torch.no_grad():
        new_emb.weight[:old_v] = old_emb.weight
        new_emb.weight[old_v:] = old_emb.weight.mean(dim=0, keepdim=True)
    model.tok_embed = new_emb

    old_fc = model.fc
    new_fc = nn.Linear(old_fc.in_features, new_vocab).to(device=device, dtype=dtype)
    with torch.no_grad():
        new_fc.weight[:old_v] = old_fc.weight
        new_fc.bias[:old_v] = old_fc.bias
        new_fc.weight[old_v:] = old_fc.weight.mean(dim=0, keepdim=True)
        new_fc.bias[old_v:] = old_fc.bias.mean()
    model.fc = new_fc

    model.vocab_size = new_vocab
    return new_vocab - old_v


# ─────────────────────── 训练 / 评估 ───────────────────────

def run_epoch(model, loader, optimizer, device, train=True, clip=1.0,
              scheduler=None, epoch=0, total_epochs=0, log_every=0, tag=""):
    """
    和预训练的区别只有一处：loss 只算 labels != -100 的位置。

    先做标准的「错位一位」：
        logits[:, :-1]  预测  labels[:, 1:]
    再交给 CrossEntropyLoss(ignore_index=-100) 自动跳过问句段和 padding 段。
    """
    model.train(train)
    total_loss = 0.0
    total_sup = 0
    total_correct = 0
    n_batches = len(loader)
    t0 = time.time()
    tty = sys.stdout.isatty()

    for i, (x, y) in enumerate(loader, 1):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        logits = model(x)                                  # (B, T, V)
        shift_logits = logits[:, :-1, :]                   # 预测位置 0..T-2
        shift_labels = y[:, 1:]                            # 目标位置 1..T-1
        loss = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            ignore_index=-100,
        )

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        mask = shift_labels != -100
        n_sup = int(mask.sum())
        total_loss += loss.item() * n_sup
        total_sup += n_sup
        with torch.no_grad():
            total_correct += int((shift_logits.argmax(-1)[mask] == shift_labels[mask]).sum())

        if log_every and (i % log_every == 0 or i == n_batches):
            elapsed = time.time() - t0
            done = i / max(1, n_batches)
            avg = total_loss / max(1, total_sup)
            acc = total_correct / max(1, total_sup)
            msg = (f"  {tag} {epoch}/{total_epochs}  {i}/{n_batches} "
                   f"({done * 100:5.1f}%)  loss {avg:.4f}  ppl {math.exp(min(avg, 700)):.1f}  "
                   f"acc {acc * 100:5.1f}%  {total_sup / max(elapsed, 1e-9):,.0f} tok/s  "
                   f"已用 {_fmt_sec(elapsed)}  剩余 {_fmt_sec(elapsed / done - elapsed)}")
            if tty:
                print("\r" + msg.ljust(132), end="", flush=True)
            else:
                print(msg, flush=True)

    if log_every and tty:
        print("\r" + " " * 132 + "\r", end="", flush=True)

    avg_loss = total_loss / max(1, total_sup)
    return avg_loss, math.exp(min(avg_loss, 700)), total_correct / max(1, total_sup)


# ─────────────────────── 主函数 ───────────────────────

def main():
    p = argparse.ArgumentParser()
    # —— 数据 ——
    p.add_argument("--data_dir", default=DEFAULT_DATA_DIR, help="SFT jsonl 所在目录")
    p.add_argument("--train_file", default="sft_train.jsonl")
    p.add_argument("--val_file", default="sft_val.jsonl", help="不存在时从训练集里切 5%")
    p.add_argument("--max_samples", type=int, default=0, help="只用前 N 条训练（0=全部），冒烟测试用")
    p.add_argument("--max_len", type=int, default=0, help="样本最大长度，0=沿用预训练 checkpoint 的 max_len")
    # —— 初始化 ——
    p.add_argument("--init_model", default="best_transformer.pt", help="预训练 checkpoint")
    p.add_argument("--save", default="sft_transformer.pt", help="SFT 权重；相对路径按 SFT 目录解析")
    p.add_argument("--code_dir", default=CODE_DIR,
                   help="预训练工程目录（transformer_lm.py 与 best_transformer.pt 所在处）")
    # —— 优化 ——
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4, help="SFT 比预训练小一档，避免把预训练知识冲掉")
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--dropout", type=float, default=-1.0, help="-1=沿用 checkpoint 的值")
    # —— 杂项 ——
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--log_every", type=int, default=20, help="每多少个 batch 打一次进度，0=关闭")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  model: TRANSFORMER (SFT)")

    # ── 1) 加载预训练权重 ──
    init_path = resolve_input(args.init_model)
    ckpt = torch.load(init_path, map_location="cpu", weights_only=False)
    cfg = ckpt["args"]
    if cfg.get("model") != "transformer" and "tok_embed.weight" not in ckpt["model_state"]:
        raise ValueError(f"{init_path} 不是 Transformer 权重，SFT 需要先跑 transformer_lm.py")

    base_char2idx = ckpt["char2idx"]
    print(f"预训练权重: {os.path.abspath(init_path)}")
    print(f"  预训练词表 {len(base_char2idx)} 字   d_model={cfg['embed_dim']}  "
          f"层数={cfg['num_layers']}  nhead={cfg['nhead']}  max_len={cfg['max_len']}")

    model = TransformerLM(
        vocab_size=len(base_char2idx),
        embed_dim=cfg["embed_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        nhead=cfg["nhead"],
        dropout=cfg.get("dropout", 0.1) if args.dropout < 0 else args.dropout,
        max_len=cfg["max_len"],
        causal=bool(cfg.get("causal", 1)),
    )
    model.load_state_dict(ckpt["model_state"])
    print("  → 预训练权重加载成功（结构与 checkpoint 完全一致）")

    # ── 2) 数据 ──
    data_dir = resolve_input(os.path.join(args.data_dir, args.train_file))
    data_dir = os.path.dirname(data_dir) or "."
    train_path = os.path.join(data_dir, args.train_file)
    val_path = os.path.join(data_dir, args.val_file)

    train_records = load_jsonl(train_path)
    if os.path.exists(val_path):
        val_records = load_jsonl(val_path)
        print(f"数据: 训练 {len(train_records):,} 条  {train_path}")
        print(f"      验证 {len(val_records):,} 条  {val_path}")
    else:
        random.shuffle(train_records)
        n_val = max(1, int(len(train_records) * 0.05))
        val_records, train_records = train_records[:n_val], train_records[n_val:]
        print(f"数据: 未找到 {args.val_file}，从训练集切 {n_val} 条做验证")

    if args.max_samples and args.max_samples < len(train_records):
        # 词表要按「完整数据」建，否则 --max_samples 会让词表随取样变化，很坑
        all_records = train_records + val_records
        train_records = train_records[:args.max_samples]
        print(f"  (--max_samples {args.max_samples}：只用前 {len(train_records)} 条)")
    else:
        all_records = train_records + val_records

    # ── 3) 扩词表（只加 token，不动已有 id）──
    char2idx, idx2char, added = build_chat_vocab(base_char2idx, all_records)
    n_added_rows = resize_vocab(model, len(char2idx))
    print(f"词表: {len(base_char2idx)} → {len(char2idx)}  "
          f"(新增 {len(added)} 个 SFT 独有字符 + {len(CHAT_SPECIALS)} 个 chat token)")
    print(f"  新增字符样例: {''.join(added[:40])}{' ...' if len(added) > 40 else ''}")
    print(f"  embedding 扩容 {n_added_rows} 行（新行用已有行均值初始化）")
    model.to(device)

    # ── 4) 数据集 ──
    max_len = args.max_len or cfg["max_len"]
    if max_len > cfg["max_len"]:
        raise ValueError(f"--max_len({max_len}) 超过位置编码上限 {cfg['max_len']}，"
                         f"加长需要改结构，SFT 不允许。")
    train_ds = SFTDataset(train_records, char2idx, max_len)
    val_ds = SFTDataset(val_records, char2idx, max_len)
    if train_ds.n_truncated or train_ds.n_dropped:
        print(f"  训练集: 截断 {train_ds.n_truncated} 条，丢弃 {train_ds.n_dropped} 条（超 max_len={max_len}）")

    pad_id = char2idx[PAD]
    collate = make_collate(pad_id)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              drop_last=False, collate_fn=collate, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            drop_last=False, collate_fn=collate, num_workers=args.num_workers)

    print(f"\n训练样本 {len(train_ds):,}  验证样本 {len(val_ds):,}  "
          f"每轮 batch 数 {len(train_loader)}")
    print(f"有监督 token 占比: {train_ds.supervised_ratio() * 100:.1f}%  "
          f"（其余是问句段 -100 与 padding，不参与梯度）")

    # 看一眼真实的样本和 mask，确认模板与 mask 没写错
    sample_ids, sample_labels = train_ds[0]
    print("\n样本 0 的 ids / labels（前 40 个位置）:")
    print("  ids   : " + " ".join(idx2char[int(i)] for i in sample_ids[:40]))
    print("  labels: " + " ".join("_" if int(v) == -100 else idx2char[int(v)] for v in sample_labels[:40]))
    print("  （_ 表示 -100，不参与 loss）\n")

    # ── 5) 优化器 ──
    save_path = resolve_output(args.save)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    print(f"模型将保存至: {save_path}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay, betas=(0.9, 0.98))
    total_steps = max(1, len(train_loader) * args.epochs)
    scheduler = make_scheduler(optimizer, total_steps, args.warmup_ratio)

    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Train PPL':>10}  {'Train Acc':>9}  "
          f"{'Val Loss':>10}  {'Val PPL':>10}  {'Val Acc':>8}  {'LR':>9}")
    print("-" * 90)

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_ppl, tr_acc = run_epoch(
            model, train_loader, optimizer, device, train=True, clip=args.grad_clip,
            scheduler=scheduler, epoch=epoch, total_epochs=args.epochs,
            log_every=args.log_every, tag="train")
        with torch.no_grad():
            va_loss, va_ppl, va_acc = run_epoch(
                model, val_loader, optimizer, device, train=False,
                epoch=epoch, total_epochs=args.epochs, log_every=0, tag="val")

        marker = ""
        if va_loss < best_val:
            best_val = va_loss
            marker = "  *"
            # checkpoint 里的 args 必须能独立复原结构：
            # SFT 的 argparse 没有 embed_dim / nhead 这些字段，全部从预训练 cfg 继承，
            # 只覆盖真正变了的 max_len / dropout，再记几个 SFT 自己的元信息。
            save_args = dict(cfg)
            save_args.update({
                "model": "transformer",
                "sft": True,
                "embed_dim": cfg["embed_dim"],
                "hidden_dim": cfg["hidden_dim"],
                "num_layers": cfg["num_layers"],
                "nhead": cfg["nhead"],
                "dropout": model.drop.p,
                "max_len": max_len,
                "causal": int(bool(cfg.get("causal", 1))),
                "seq_len": cfg.get("seq_len", max_len),
                "init_model": os.path.basename(init_path),
                "chat_specials": CHAT_SPECIALS,
                "sft_epochs": args.epochs,
                "sft_lr": args.lr,
                "sft_batch_size": args.batch_size,
            })
            torch.save({
                "model_state": model.state_dict(),
                "char2idx": char2idx,
                "idx2char": idx2char,
                "args": save_args,
            }, save_path)

        print(f"{epoch:>6}  {tr_loss:>10.4f}  {tr_ppl:>10.2f}  {tr_acc * 100:>8.2f}%  "
              f"{va_loss:>10.4f}  {va_ppl:>10.2f}  {va_acc * 100:>7.2f}%  "
              f"{scheduler.get_last_lr()[0]:>9.2e}{marker}")

    print(f"\nSFT 完成。最佳验证 loss: {best_val:.4f}  已保存至 {save_path}"
          if best_val < float("inf") else "\nSFT 完成。没有跑任何 epoch，未保存模型。")
    print(f"下一步: python chat.py --model_path {os.path.basename(save_path)} --question \"什么是人工智能？\"")


if __name__ == "__main__":
    main()
