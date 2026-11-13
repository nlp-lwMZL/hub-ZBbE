"""
transformer_sft.py — 在 transformer_lm.py 的字符级 Transformer 上做 SFT（监督微调）

【这个脚本干什么】
    模型结构和训练流程完全沿用 transformer_lm.py，只改三处：
        1. 数据：从「连续语料切窗口」换成 (问题, 回答) 拼成的对话模板
        2. loss：问题段标签置 -100 不参与计算，只学「回答 + <|end|>」（loss mask）
        3. 推理：输入问题 → 拼模板 → 逐字生成 → 遇到 <|end|> 停止

    这三处就是 SFT 的全部：预训练教模型「会中文」，SFT 教它「按问答格式回答」。

【数据】
    sft_train/sft_train.jsonl  3000 条训练
    sft_train/sft_val.jsonl     200 条验证
    每行：{"instruction": "问题", "output": "回答"}

【用法】
    python transformer_sft.py                                  # 默认从零训练 20 轮，存 sft_transformer.pt
    python transformer_sft.py --epochs 10                      # 只跑 10 轮
    python transformer_sft.py --chat "数字1: 238 数字2: 564"    # 加载权重直接提问
    python transformer_sft.py --init ../30-week04-作业答案/week4/tf_mask.pt   # 加载预训练权重做真微调

【阅读顺序】
    常量/模板 → SFTDataset（loss mask 在哪） → TransformerLM（结构） → train → answer_question
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

# 脚本所在目录：默认数据、权重路径都基于它，从任何目录执行都能找到文件
HERE = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────── 对话模板 ───────────────────────
# 字符级模型没有 tokenizer，这些「标记」就是一串普通字符；
# 建字表时它们会被收进表里，模型于是能学会「谁在说话、说到哪算完」。
USER_TAG = "<|user|>"
ASST_TAG = "<|assistant|>"
END_TAG = "<|end|>"
DIGIT_CHARS = "0123456789."

# padding 占位符：JSONL 里不可能出现 \u0000，所以固定给它 id 0。
# 注意 pad 用「id 0 + 注意力掩码」屏蔽，loss 用「-100」屏蔽，两套机制分工不同。
PAD_CHAR = "\u0000"


# ─────────────────────── 数据 ───────────────────────

def load_jsonl(path):
    """一行一个 JSON；空行跳过。"""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not records:
        raise FileNotFoundError(f"没读到数据: {path}")
    return records


def format_chat(question, answer):
    """把一问一答拼成一条完整对话：<|user|>问题<|assistant|>回答<|end|>"""
    return f"{USER_TAG}{question}{ASST_TAG}{answer}{END_TAG}"


def build_vocab(texts):
    """
    建字符级字表：PAD 固定占 id 0，其余字符按码位排序（顺序稳定 → 可复现）。

    和 transformer_lm.py 的唯一区别：这里必须预留 PAD，
    因为 SFT 的样本长短不一，凑 batch 时要补 pad。
    """
    chars = set()
    for t in texts:
        chars |= set(t)
    chars.discard(PAD_CHAR)
    char2idx = {PAD_CHAR: 0}
    for c in sorted(chars):
        char2idx[c] = len(char2idx)
    return char2idx, {i: c for c, i in char2idx.items()}


class SFTDataset(Dataset):
    """
    一条样本 = 一条完整对话；labels 只在「回答段 + <|end|>」有值，其余是 -100。

        ids:    <|user|> 问题 <|assistant|> 回答 <|end|>
        labels:  -100      -100      -100     答案首字 … <|end|>

    loss mask 的实现就在下面三行：
        · labels[j-1] = ids[j]：位置 j-1 的输出负责预测第 j 个字（和预训练一样右移一位）；
        · 只对 j >= prefix_len 填标签，所以问题段全是 -100；
        · 交叉熵的 ignore_index=-100 会跳过这些位置 —— 问题段不产生梯度。
    """

    def __init__(self, records, char2idx, seq_len):
        self.items = []
        for r in records:
            q, a = r["instruction"], r["output"]
            prefix = f"{USER_TAG}{q}{ASST_TAG}"
            full = f"{prefix}{a}{END_TAG}"

            # 超长时优先裁问题，保住完整的「回答 + <|end|>」
            # （本数据最长对话 139 字 < 默认 seq_len=144，这段不会触发）
            if len(full) > seq_len:
                budget = seq_len - len(USER_TAG) - len(ASST_TAG) - len(a) - len(END_TAG)
                q = q[:max(1, budget)]
                prefix = f"{USER_TAG}{q}{ASST_TAG}"
                full = f"{prefix}{a}{END_TAG}"

            # 字表外字符直接丢弃（本数据集都在表里，这行是兜底）；
            # keep 里带上原位置，才能算出「哪些字符属于前缀」
            keep = [(c, i) for i, c in enumerate(full) if c in char2idx]
            ids = [char2idx[c] for c, _ in keep]
            prefix_len = sum(1 for c, _ in keep if _ < len(prefix))

            # 右移一位：只有回答段（含结束符）产生标签，其余保持 -100
            labels = [-100] * len(ids)
            for j in range(prefix_len, len(ids)):
                labels[j - 1] = ids[j]

            self.items.append((torch.tensor(ids), torch.tensor(labels)))

        self.pad_id = char2idx[PAD_CHAR]
        self.seq_len = seq_len

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def make_collate(pad_id):
    """
    把一个 batch 补到同长（SFT 样本长度不一，不像预训练窗口固定长度）：
        ids    用 pad_id 补
        labels 用 -100 补（pad 位置不算 loss）
        mask   用 True 补（True = 该位置是 pad，注意力里屏蔽掉，不参与任何位置的计算）
    """
    def collate(batch):
        max_len = max(len(x) for x, _ in batch)
        xs, ys, masks = [], [], []
        for x, y in batch:
            n = max_len - len(x)
            xs.append(torch.cat([x, torch.full((n,), pad_id, dtype=torch.long)]))
            ys.append(torch.cat([y, torch.full((n,), -100, dtype=torch.long)]))
            masks.append(torch.cat([torch.zeros(len(x), dtype=torch.bool),
                                    torch.ones(n, dtype=torch.bool)]))
        return torch.stack(xs), torch.stack(ys), torch.stack(masks)

    return collate


# ─────────────────────── 模型（与 transformer_lm.py 一致） ───────────────────────

class TransformerLM(nn.Module):
    """
    字符 embedding + 可学习位置编码 + N 层 TransformerEncoderLayer
    （Pre-LN、GELU、FFN=4×d）+ 与输入 embedding 共享权重的输出头。

    这里同时用两种掩码：
        causal mask  —— 下三角 0、右上 -inf，位置 t 只能看 0..t，这是自回归的前提；
        pad mask     —— batch 补齐出来的位置不参与注意力（SFT 新增，预训练窗口等长所以不需要）。
    """

    def __init__(self, vocab_size, d_model=192, n_head=4, n_layer=4, max_len=512, dropout=0.1):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model)      # 字符 id → d 维向量
        self.pos = nn.Embedding(max_len, d_model)         # 可学习位置编码（比 sin/cos 更适合小模型）
        self.drop = nn.Dropout(dropout)

        # 这一行就是一个 GPT block：Masked Self-Attn → 残差+LN → FFN(4d, GELU) → 残差+LN。
        # norm_first=True 即 Pre-LN：LayerNorm 放在子层之前，残差通路干净，深层不用 warmup 也能训；
        # dim_feedforward=4*d_model 是原论文的固定比例（FFN 占每层约 2/3 参数）。
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head, dim_feedforward=4 * d_model,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        # 堆 N 层 + 最后一层 LayerNorm；PyTorch 会 deepcopy 这个 layer，不存在参数共享
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=n_layer, norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False,      # 带 mask 时关掉嵌套张量优化，行为更直观
        )
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok.weight   # weight tying：输入/输出共用一套 (V×d)，
                                             # 省约 1/4 参数，且两者语义空间本就一致

        # 共享权重时初始化必须小（std=0.02，GPT-2 的做法）：
        # nn.Embedding 默认 std=1 会让初始 logits 过大、初始 loss 飙高
        nn.init.normal_(self.tok.weight, std=0.02)
        nn.init.normal_(self.pos.weight, std=0.02)

    def forward(self, idx, pad_mask=None):
        # idx: (B, T) → logits: (B, T, V)
        x = self.tok(idx)
        x = x + self.pos(torch.arange(idx.size(1), device=idx.device))
        x = self.drop(x)

        # 因果掩码：位置 t 只能看到 0..t，保证训练目标和推理时「只能看前文」一致。
        # 转 bool 是因为要和下面 src_key_padding_mask 的类型统一，否则 PyTorch 会告警。
        mask = nn.Transformer.generate_square_subsequent_mask(
            idx.size(1), device=idx.device).bool()
        # src_key_padding_mask 中 True 的位置被屏蔽 → pad 不参与注意力
        x = self.encoder(x, mask=mask, src_key_padding_mask=pad_mask)
        # 直接输出 logits，不在模型里 softmax：cross_entropy 自带 log_softmax，做两次会掉数值精度
        return self.head(x)


# ─────────────────────── 训练 / 评估 ───────────────────────

@torch.no_grad()
def evaluate(model, loader, device):
    """
    返回回答段的 (loss, ppl)；被 -100 掩码的问题 token 不参与统计。

    用 reduction="sum" 再除以有效 token 数，而不是让 cross_entropy 直接取 mean：
    否则最后一个不满的 batch 会和满 batch 等权，loss 口径不干净。
    """
    model.eval()                      # 关 dropout，否则每次评测结果都在抖
    total, tokens = 0.0, 0
    for x, y, pad in loader:
        x, y, pad = x.to(device), y.to(device), pad.to(device)
        logits = model(x, pad)
        total += float(F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                       y.reshape(-1), ignore_index=-100,
                                       reduction="sum"))
        tokens += int((y != -100).sum())
    loss = total / max(tokens, 1)
    return loss, math.exp(loss)       # ppl = exp(loss)：「每步平均在几个候选字之间犹豫」


@torch.no_grad()
def answer_question(model, char2idx, idx2char, question, device, seq_len,
                    max_new_chars=80, temperature=0.5,
                    repetition_penalty=1.0, allowed_ids=None):
    """
    问题 → 拼前缀 → 逐字生成 → 遇到 <|end|> 停止。

    推理必须逐字：第 t+1 个字还不存在，没法像训练那样一次算完所有位置。
    这里每步都重喂整段上文（没有 KV Cache），只为代码简短。

    repetition_penalty > 1 时降低已生成字的概率（抑制复读）；
    allowed_ids 非空时只在该字符集合内选字（约束解码）。
    """
    model.eval()
    prefix = f"{USER_TAG}{question}{ASST_TAG}"
    ids = [char2idx[c] for c in prefix if c in char2idx]
    out = list(ids)

    for _ in range(max_new_chars):
        # 只喂最近 seq_len 个字：位置编码表只有 seq_len 行，训练时也只见识过这么长的上下文
        logits = model(torch.tensor([out[-seq_len:]], device=device))[0, -1].float()

        if allowed_ids is not None:
            keep = torch.tensor(sorted(allowed_ids), device=logits.device)
            masked = torch.full_like(logits, float("-inf"))
            masked[keep] = logits[keep]
            logits = masked

        # 重复惩罚：把已经出现过的字压下去（正 logit 除以系数，负值乘系数）
        for tid in set(out[len(ids):]):
            logits[tid] = logits[tid] / repetition_penalty if logits[tid] > 0 \
                else logits[tid] * repetition_penalty

        if temperature <= 0.01:
            nxt = int(logits.argmax())                      # 贪心解码，结果确定
        else:
            # 温度采样：T<1 分布更尖（更稳），T>1 更平（更容易出错）
            nxt = int(torch.multinomial(F.softmax(logits / temperature, dim=-1), 1))
        out.append(nxt)
        if "".join(idx2char[i] for i in out[len(ids):]).endswith(END_TAG):
            break

    new_chars = [idx2char[i] for i in out[len(ids):] if idx2char[i] != PAD_CHAR]
    return "".join(new_chars).split(END_TAG)[0]


@torch.no_grad()
def load_pretrained_state(model, ck):
    """
    把 transformer_lm.py 的预训练权重灌进同构的 SFT 模型。

    两处形状不同，只做「尾部扩展」，已训练过的行保持不变：
        词表：预训练 4000 → 这里 4000 + SFT 新字符 + PAD，前 4000 行逐行拷贝；
        位置表：预训练 64 → 这里 seq_len(144)，前 64 行拷贝，新增行保留小随机初始化。
    其余层（encoder / norm）形状一致，直接拷贝。
    """
    sd = ck["model_state"]
    params = dict(model.named_parameters())          # head.weight 与 tok.weight 共享，只出现一次
    n_tok = sd["tok.weight"].size(0)
    params["tok.weight"][:n_tok].copy_(sd["tok.weight"])
    n_pos = min(sd["pos.weight"].size(0), params["pos.weight"].size(0))
    params["pos.weight"][:n_pos].copy_(sd["pos.weight"][:n_pos])
    for k, v in sd.items():
        if k in ("tok.weight", "head.weight", "pos.weight"):
            continue
        params[k].copy_(v)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # ── 读数据 ──
    train_records = load_jsonl(args.data)
    val_records = load_jsonl(args.val)
    if args.max_samples:
        train_records = train_records[:args.max_samples]

    # 字表：默认从「所有拼好的对话」里建；给了 --init 就沿用预训练字表，只在尾部追加新字符
    all_texts = [format_chat(r["instruction"], r["output"]) for r in train_records + val_records]
    ck = None
    if args.init:
        ck = torch.load(args.init, map_location="cpu")
        pa = ck["args"]
        # 结构必须与预训练 ckpt 一致，否则权重对不上，故以 ckpt 记录为准
        args.d_model, args.heads, args.layers = pa["d_model"], pa["heads"], pa["layers"]
        char2idx = dict(ck["char2idx"])
        oov = sorted({c for t in all_texts for c in t if c not in char2idx})
        for c in oov:
            char2idx[c] = len(char2idx)
        char2idx[PAD_CHAR] = len(char2idx)
        idx2char = {i: c for c, i in char2idx.items()}
        print(f"初始化: 预训练字表 {len(ck['char2idx'])}，新增 {len(oov)} 个字符 + PAD，"
              f"合计 {len(char2idx)}")
    else:
        char2idx, idx2char = build_vocab(all_texts)
    V = len(char2idx)
    print(f"device: {device}  训练 {len(train_records)} 条 / 验证 {len(val_records)} 条  字表 {V}")

    train_ds = SFTDataset(train_records, char2idx, args.seq_len)
    val_ds = SFTDataset(val_records, char2idx, args.seq_len)
    collate = make_collate(char2idx[PAD_CHAR])
    # shuffle=True：每轮 batch 组成不同，起一点正则作用；drop_last=True：丢掉不满的尾批，形状固定
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              drop_last=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            drop_last=False, collate_fn=collate)

    # 打印一条样本，肉眼确认 loss 只落在回答段
    ids, labels = train_ds[0]
    print(f"示例: 序列 {len(ids)} 字，参与 loss 的回答 token {int((labels != -100).sum())} 个")

    # ── 模型 ──
    model = TransformerLM(V, args.d_model, args.heads, args.layers,
                          args.seq_len, args.dropout).to(device)
    if ck is not None:
        load_pretrained_state(model, ck)
        print(f"已加载预训练权重: {os.path.basename(args.init)}（词表 {len(ck['char2idx'])} → {V}）")
    n_par = sum(p.numel() for p in model.parameters())
    print(f"模型: TransformerEncoderLayer × {args.layers}（d={args.d_model} h={args.heads} "
          f"FFN={4*args.d_model} Pre-LN）  因果掩码=有  参数量 {n_par/1e6:.2f}M")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    best = float("inf")

    print(f"\n{'Epoch':>6} {'TrainLoss':>10} {'TrainPPL':>10} {'ValLoss':>10} {'ValPPL':>10} {'耗时':>6}")
    print("-" * 62)
    for ep in range(1, args.epochs + 1):
        model.train()                 # 与 evaluate 里的 eval() 成对：训练开 dropout
        st, tot, seen = time.time(), 0.0, 0
        for x, y, pad in train_loader:
            x, y, pad = x.to(device), y.to(device), pad.to(device)
            logits = model(x, pad)
            # ★ loss mask 的核心：ignore_index=-100 让问题段和 pad 位置不产生梯度
            loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1), ignore_index=-100)

            opt.zero_grad()
            loss.backward()
            # 梯度裁剪 1.0：Transformer 偶发梯度尖刺，裁一下防炸，几乎不影响正常收敛
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            n = int((y != -100).sum())              # 按有效 token 数加权，和 evaluate 同口径
            tot += loss.detach().item() * n
            seen += n

        train_loss = tot / max(seen, 1)
        val_loss, val_ppl = evaluate(model, val_loader, device)
        mark = ""
        if val_loss < best:
            best = val_loss
            mark = " *"
            # ckpt 要能「独立重建模型 + 解码」，所以存四样东西：
            # 权重、字表双向映射（字表顺序随数据变！）、结构超参 args
            torch.save({"model_state": model.state_dict(),
                        "char2idx": char2idx, "idx2char": idx2char,
                        "args": vars(args)}, args.save)
        print(f"{ep:>6} {train_loss:>10.4f} {math.exp(train_loss):>10.2f} "
              f"{val_loss:>10.4f} {val_ppl:>10.2f} {time.time()-st:>5.0f}s{mark}")

        # 每 demo_every 轮看一眼生成效果（贪心解码，结果可复现）
        if ep % args.demo_every == 0 or ep == args.epochs:
            q = args.demo_question
            print(f"        问：{q}\n        答：{answer_question(model, char2idx, idx2char, q, device, args.seq_len, temperature=0.0)}")

    print(f"\n训练完成，最佳验证 loss {best:.4f}，已保存 {args.save}")

    # 用保存的最优权重复测几条验证集问题
    ck = torch.load(args.save, map_location=device)
    model.load_state_dict(ck["model_state"])
    print("\n验证集示例：")
    for r in val_records[:5]:
        pred = answer_question(model, char2idx, idx2char, r["instruction"], device,
                               args.seq_len, temperature=args.temperature)
        print(f"  问：{r['instruction']}")
        print(f"  答：{pred}")
        print(f"  参：{r['output']}\n")


def chat(args):
    """加载训练好的权重，输入一个问题，输出回答（dropout 置 0 保证结果稳定）。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location=device)
    c = ck["args"]                    # 用 ckpt 里记下的结构超参重建同构模型
    char2idx, idx2char = ck["char2idx"], ck["idx2char"]
    model = TransformerLM(len(char2idx), c["d_model"], c["heads"], c["layers"],
                          c["seq_len"], 0.0).to(device)
    model.load_state_dict(ck["model_state"])
    allowed = None
    if args.digits_only:
        # 只允许数字、小数点和结束符的字符参与解码，其余 logit 置 -inf
        allowed = {i for ch, i in char2idx.items() if ch in DIGIT_CHARS + END_TAG}
    text = answer_question(model, char2idx, idx2char, args.chat, device,
                           c["seq_len"], args.max_new_chars, args.temperature,
                           args.rep_penalty, allowed)
    print(text)


def main():
    ap = argparse.ArgumentParser(description="字符级 Transformer 的 SFT（监督微调）")
    # 数据与输出
    ap.add_argument("--data", default=os.path.join(HERE, "sft_train.jsonl"))
    ap.add_argument("--val", default=os.path.join(HERE, "sft_val.jsonl"))
    ap.add_argument("--save", default=os.path.join(HERE, "sft_transformer.pt"))
    ap.add_argument("--ckpt", default=os.path.join(HERE, "sft_transformer.pt"))
    ap.add_argument("--init", default=None,
                    help="预训练 ckpt 路径；给了就先加载权重再微调（结构以该 ckpt 为准）")
    # 数据切分
    ap.add_argument("--seq_len", type=int, default=144, help="最长对话长度（字符数）")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_samples", type=int, default=0, help="只用前 N 条训练（快速试跑）")
    # 结构（与 transformer_lm.py 默认一致）
    ap.add_argument("--d_model", type=int, default=192)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    # 训练
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--demo_every", type=int, default=5, help="每几轮打印一次示例回答")
    ap.add_argument("--demo_question", default="给定两个数字，请选择其中的较大数并将其打印出来。 数字1: 238 数字2: 564")
    # 推理
    ap.add_argument("--chat", default=None, help="给一个问题，加载 --ckpt 直接回答")
    ap.add_argument("--max_new_chars", type=int, default=80)
    ap.add_argument("--temperature", type=float, default=0.5, help="0 表示贪心解码")
    ap.add_argument("--rep_penalty", type=float, default=1.0,
                    help="重复惩罚系数，>1 抑制复读（建议 1.1~1.3）")
    ap.add_argument("--digits_only", action="store_true",
                    help="约束解码：只允许输出数字、小数点和结束符")
    # ── 超参数的来历（本数据集实测：对话 46~139 字，均值 88；答案均值 27.1、最长 64；字表 2313）──
    #   seq_len=144：取能覆盖全部样本（最长 139）的最小整数档；它同时是位置编码表长度。
    #   batch_size=64：3000 条 → 每轮 46 步；CPU 上补到 144 的 64×144 前向已占约 1.6 GB。
    #   d_model/heads/layers/dropout：直接沿用 transformer_lm.py 预训练配置（换结构权重就对不上），
    #       本字表下参数量约 2.25M（预训练字表 4000 时是 2.66M）。
    #   epochs=20：3000 条小数据 + 2.25M 参数，20 轮足够收敛；dropout=0.1 抑制过拟合。
    #   lr=3e-4：微调时取预训练 1e-3 的 1/3 左右。实测 1e-4 收敛太慢（3 轮后 val loss 仍
    #       高于从零训练 20 轮的结果），3e-4 三轮就到 2.68；从零训练也可用 5e-4。
    #   max_new_chars=80：答案最长 64 字，留 16 字余量防止停不下来。
    #   temperature=0.5：<1 让分布更尖、回答更稳；训练中的 demo 用 0（贪心）保证可复现。
    args = ap.parse_args()

    if args.chat:
        chat(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
