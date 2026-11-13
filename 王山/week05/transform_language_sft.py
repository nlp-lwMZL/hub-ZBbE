# -*- coding: utf-8 -*-
"""
字符级语言模型训练脚本（修复版）—— RNN / LSTM / Transformer 可选，含 PPL 计算
================================================================================
  🔴 训练逻辑
     4. 原脚本「每个 epoch 都保存模型」，注释说保存 best —— 实际会覆盖。
        改为只在 val PPL 创新低时保存。
     5. 原脚本的打印缩进错误，epoch 行为无法正确显示。
     6. 验证集空时除零。

  🟡 数据与工程
     7. CharDataset 切块的边界计算：len 用 (L - seq_len - 1) // stride + 1，
        当 stride 较大时会漏掉末尾有效样本；且空样本未跳过。
     8. 缺少"生成示例"能力，训练完无法直观看出模型学到了什么。

用法：
    python language_model_fixed.py --model lstm        --epochs 20
    python language_model_fixed.py --model transformer --epochs 20
    python language_model_fixed.py --model transformer --corpus "data/*.txt" --save lm.pt
    python language_model_fixed.py --model lstm --sample "今天天气"     # 训练完生成续写
"""

import os
import math
import argparse
import glob
import random
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ═══════════════════════════ 数据 ═══════════════════════════

SPECIAL_TOKENS = ["<PAD>", "<inst>", "<out>"]

def load_corpus(pattern="*.txt"):
    """批量读取匹配的 txt，全部拼成一长串。"""
    texts = []
    paths = glob.glob(pattern, recursive=True)
    if not paths:
        return "", []
    for path in sorted(paths):
        with open(path, encoding="utf-8", errors="ignore") as f:
            texts.append(f.read())
    return "".join(texts), paths


def build_vocab(text, max_vocab=None):
    """
    构建字表：按频率取前 max_vocab 个字。
    ⚠️ 坑：0 号位必须是 <PAD>。
       原脚本用 `sorted(set(text))`，0 号位是语料里 ASCII 最小的那个字，
       然后又拿 ignore_index=0 当 padding —— 等于"这个字永远不参与 loss"。
       这里强制把 <PAD> 放在 0 号位，语义就干净了。
    """
    from collections import Counter
    counter = Counter(text)
    if max_vocab is None:
        chars = [k for k, _ in counter.most_common()]
    else:
        reserve = max_vocab - len(SPECIAL_TOKENS)
        if reserve <= 0:
            raise ValueError("max_vocab 必须大于特殊token数量(3)")
        chars = [k for k, _ in counter.most_common(reserve)]
    vocab = SPECIAL_TOKENS + chars
    char2idx = {c: i for i, c in enumerate(vocab)}
    idx2char = {i: c for i, c in enumerate(vocab)}
    return char2idx, idx2char

def format_sft_sample(instruction, output):
    # 拼接模板
    return f"<inst>{instruction}<out>{output}"

def make_sft_target_ids(ids, char2idx):
    # SFT loss掩码核心
    out_id = char2idx["<out>"]
    target = ids.clone()
    # 找到第一个<resp>的下标
    resp_pos = (ids == out_id).nonzero(as_tuple=True)[0]
    if len(resp_pos) > 0:
        first_resp = resp_pos[0]
        # <resp>以及前面所有位置全部置为PAD，不参与loss
        target[:first_resp+1] = char2idx["<PAD>"]
    return target

class CharDataset(Dataset):
    """
    字符级 LM 样本：x = 连续 seq_len 个字符，y = 右移一位。

    【修复点 3】跳过不含任何有效字符的块，避免整块全 pad。
    """

    def __init__(self, text, char2idx, seq_len, stride=1, name=""):
        self.seq_len = seq_len
        self.stride = max(1, stride)
        self.name = name

        ids = [char2idx[c] for c in text if c in char2idx]
        self.data = torch.tensor(ids, dtype=torch.long)

        # 预计算所有有效起点：要求这一块里至少有一个非 PAD id
        self.starts = []
        if len(self.data) > seq_len:
            for s in range(0, len(self.data) - seq_len, self.stride):
                if (self.data[s:s + seq_len] != 0).any():
                    self.starts.append(s)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        start = self.starts[idx]
        x = self.data[start:start + self.seq_len]
        y = self.data[start + 1:start + self.seq_len + 1]
        return x, y

# ═══════════════════════════ SFT数据集 ═══════════════════════════
class SftCharDataset(Dataset):
    """
       SFT监督微调数据集：处理指令配对 [instruction(prompt), response(回答)]
       json样例 sft.json:
       [
         {"instruction":"1+1等于几","response":"1加1等于2"},
         {"instruction":"写一句春天的诗","response":"春风拂柳绿，花开满山岗"}
       ]
       拼接格式：`instruction + response`
       label中 instruction部分全部置PAD=0，CrossEntropyLoss会ignore，**仅response计算loss**
       """

    def __init__(self, json_path, char2idx, seq_len):
        self.seq_len = seq_len
        self.char2idx = char2idx
        self.pad_idx = char2idx["<PAD>"]
        self.data = []
        with open(json_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                self.data.append(d)



    def __len__(self):
        return len(self.data)

    def text2ids(self, s):
        return [self.char2idx[c] for c in s if c in self.char2idx]

    def __getitem__(self, index):
        item = self.data[index]
        instr = item["instruction"]
        output = item["output"]
        sft_text = format_sft_sample(instr, output)
        ids = self.text2ids(sft_text)

        # 截断、补PAD
        if len(ids) >= self.seq_len:
            ids = ids[:self.seq_len]
        else:
            ids += [self.pad_idx] * (self.seq_len - len(ids))
        ids = torch.tensor(ids, dtype=torch.long)
        x = ids

        # 自回归：y是x右移一位
        y = torch.zeros_like(x)
        y[:-1] = x[1:]
        # mask指令部分loss
        y = make_sft_target _ids(y, self.char2idx)
        return x, y


# ═══════════════════════════ 模型 ═══════════════════════════

class LM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers,
                 model_type, dropout, pad_idx=0):
        super().__init__()
        self.model_type = model_type
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.drop = nn.Dropout(dropout)

        if model_type in ("lstm", "rnn"):
            cls = nn.LSTM if model_type == "lstm" else nn.RNN
            self.rnn = cls(
                embed_dim, hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.out_dim = hidden_dim

        elif model_type == "transformer":
            # ⚠️ d_model 必须能被 nhead 整除
            nhead = 4
            if embed_dim % nhead != 0:
                raise ValueError(
                    f"embed_dim({embed_dim}) 必须能被 nhead({nhead}) 整除，"
                    f"建议设成 {nhead} 的倍数")
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=nhead,
                dim_feedforward=embed_dim * 2,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
                norm_first=True,          # 【建议】pre-norm 更稳定，nan 风险更低
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.out_dim = embed_dim
            self._causal = None
        else:
            raise ValueError(f"未知 model_type: {model_type}")

        self.fc = nn.Linear(self.out_dim, vocab_size)

    def _causal_mask(self, L, device):
        """
        【核心修复 1】因果 mask。
        正确做法：用 float('-inf') 的加性 mask，并且**必须保留自注意力对角线**。
        """
        if self._causal is not None and self._causal.size(0) == L:
            return self._causal
        # float 加性 mask：上三角（不含对角）设 -inf，表示"只能看自己和前面"
        m = torch.full((L, L), float("-inf"), device=device, dtype=torch.float32)
        # 函数：torch.full(size, fill_value, *, dtype=None, device=None)
        # 作用：创建一个指定形状的张量，所有元素统一填充成 fill_value
        m = torch.triu(m, diagonal=1)
        # torch.triu(m, diagonal=1)
        # diagonal=0：包含对角线的上三角
        # diagonal=1：跳过对角线，从对角线右边一格开始

        self._causal = m
        return m

    def forward(self, x):
        e = self.drop(self.embed(x))
        B, L = x.shape

        if self.model_type == "transformer":
            pad_bool = (x == 0)
            # 整个 batch 都是 pad 时，padding mask 会把所有列都屏蔽 → 必 nan
            # 这里给每行强行保留第 0 个位置可见，保证 softmax 有合法分母
            all_pad_row = pad_bool.all(dim=1)                 # [B]
            if all_pad_row.any():
                pad_bool = pad_bool.clone()
                pad_bool[all_pad_row, 0] = False

            out = self.encoder(
                e,
                mask=self._causal_mask(L, x.device),
                src_key_padding_mask=pad_bool,
            )
        else:
            out, _ = self.rnn(e)

        return self.fc(self.drop(out))


# ═══════════════════════════ 训练 / 评估 ═══════════════════════════

def run_epoch(model, loader, criterion, optimizer, device, train=True, grad_clip=1.0):
    """
    【修复点 2】空 loader / 全 pad batch 的保护。
    返回 (avg_loss, ppl)；若本轮无有效 token，返回 (nan, nan) 交由调用方处理。
    """
    model.train() if train else model.eval()
    total_loss = 0.0
    total_tokens = 0

    ctx = torch.ena ble_grad() if train else torch.no_grad()
    with ctx:
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            # 有效目标数（ignore_index=0 会被跳过）
            n_valid = (y != 0).sum().item()
            if n_valid == 0:
                continue          # 整 batch 全 pad，直接跳过，否则必 nan

            logits = model(x)
            loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

            if not torch.isfinite(loss):
                # 保护：出现 inf/nan 直接跳过这一步，不污染权重
                print(f"  ⚠️ 检测到非有限 loss（{loss.item()}），跳过该 batch")
                if train:
                    optimizer.zero_grad(set_to_none=True)
                continue

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                optimizer.step()

            # 按有效 token 数加权，而不是按 y.numel()
            total_loss += loss.item() * n_valid
            total_tokens += n_valid

    if total_tokens == 0:
        return float("nan"), float("nan")

    avg_loss = total_loss / total_tokens
    return avg_loss, math.exp(min(avg_loss, 50.0))   # PPL 上限保护，防止溢出



@torch.no_grad()
def generate(model, prompt, char2idx, idx2char, device, seq_len=64, max_gen_len=100):
    model.eval()
    pad_idx = char2idx["<PAD>"]
    # 把prompt转id
    ids = [char2idx.get(c, pad_idx) for c in prompt]
    x = torch.tensor([ids], dtype=torch.long, device=device)

    for _ in range(max_gen_len):
        # 截断到模型接受的长度seq_len
        if x.size(1) > seq_len:
            x = x[:, -seq_len:]
        logits = model(x)
        # 取最后一个位置预测
        next_logit = logits[:, -1, :]
        next_id = torch.argmax(next_logit, dim=-1, keepdim=True)
        x = torch.cat([x, next_id], dim=1)
        # 可以加终止判断，比如预测到某个结束符，没有就删掉这行
        # if next_id.item() == char2idx.get("<EOS>", -1):
        #     break
    # 转回文字
    out_ids = x[0].cpu().tolist()
    text = "".join([idx2char[i] for i in out_ids if i != pad_idx])
    return text


# ═══════════════════════════ 主函数 ═══════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="transformer",
                        choices=["rnn", "lstm", "transformer"])
    parser.add_argument("--mode", default="pretrain", choices=["pretrain", "sft"],
                        help="pretrain原始预训练；sft监督微调模式")
    parser.add_argument("--pretrain_ckpt", default="best_model_transform.pt", help="SFT模式加载预训练权重路径")
    parser.add_argument("--train_jsonl", type=str, default="sft_train.jsonl")
    parser.add_argument("--val_jsonl", type=str, default="sft_val.jsonl")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--stride", type=int, default=64,
                        help="切块间隔：1=密集重叠（样本数暴涨），seq_len=不重叠")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_ratio", type=float, default=0.1,help="仅pretrain模式生效：训练验证切分比例")
    parser.add_argument("--max_vocab", type=int, default=4000, help="0=不限")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--corpus", default="*.txt")
    parser.add_argument("--save", default="best_model_transform.pt")
    parser.add_argument("--sample", default="", help="训练完用它当开头生成一段")
    parser.add_argument("--threads", type=int, default=0, help="CPU 线程数，0=自动")
    args = parser.parse_args()

    if args.threads > 0:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  model: {args.model.upper()}")

    # ---------- 数据 ----------
    char2idx, idx2char = None, None
    if args.mode == "pretrain":
    # 预训练模式逻辑（原有代码）
        text, paths = load_corpus(args.corpus)
        if not text:
            raise FileNotFoundError(
                f"未找到匹配 {args.corpus} 的 txt 文件，请确认路径正确。")
        print(f"语料文件: {len(paths)} 个，字符数: {len(text):,}")

        char2idx, idx2char = build_vocab(text, args.max_vocab or None)
    elif args.mode == "sft":
        # SFT模式：加载预训练checkpoint读取词表，词表必须和预训练完全一致，不能重新构建！
        if not args.pretrain_ckpt or not os.path.exists(args.pretrain_ckpt):
            raise FileNotFoundError("SFT模式必须指定--pretrain_ckpt，加载预训练模型，词表从ckpt读取！")
        ckpt = torch.load(args.pretrain_ckpt, map_location=device, weights_only=False)
        char2idx = ckpt["char2idx"]
        idx2char = ckpt["idx2char"]
        print(f"SFT模式，从{args.pretrain_ckpt}加载词表；词表大小 {len(char2idx)}")

    vocab_size = len(char2idx)
    pad_idx = char2idx.get("<PAD>", 0)
    print(f"词表大小: {vocab_size}（含 <PAD>，id={pad_idx}）")

    # ---------- 按行划分 train / val ----------
    train_ds, val_ds = None, None
    if args.mode == "pretrain":
        lines = [ln for ln in text.splitlines() if ln.strip()]  # 【修复】丢掉空行
        if not lines:
            raise ValueError("语料里没有非空行")
        random.shuffle(lines)
        split = max(1, int(len(lines) * (1 - args.val_ratio)))
        train_text = "\n".join(lines[:split])
        val_text = "\n".join(lines[split:]) or "\n".join(lines[:1]) # 兜底
        train_ds = CharDataset(train_text, char2idx, args.seq_len, args.stride, "train")
        val_ds = CharDataset(val_text, char2idx, args.seq_len, args.stride, "val")
        print(f"训练样本: {len(train_ds):,}   验证样本: {len(val_ds):,}")


        if len(train_ds) == 0:
            raise ValueError(
                f"训练样本为 0：语料太短或 seq_len({args.seq_len}) 太大。"
                f"训练集只有 {len(train_text)} 字符")
        if len(val_ds) == 0:
            print("⚠️ 验证样本为 0（验证集太短），将用训练集的一部分代替")
            val_ds = CharDataset(train_text[-max(args.seq_len * 4, 512):],
                             char2idx, args.seq_len, args.stride, "val(fallback)")
    elif args.mode == "sft":
        # SFT数据集加载
        train_ds = SftCharDataset(args.train_jsonl, char2idx, args.seq_len)
        val_ds = SftCharDataset(args.val_jsonl, char2idx, args.seq_len)
        print(f"SFT训练样本:{len(train_ds)}  SFT验证样本:{len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,shuffle=False, drop_last=False)

    # ---------- 模型 ----------
    model = LM(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        model_type=args.model,
        dropout=args.dropout,
        pad_idx=pad_idx,
    ).to(device)
    if args.mode == "sft":
        # SFT加载预训练权重
        ckpt = torch.load(args.pretrain_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        print("✅ SFT：成功加载预训练模型权重，开始监督微调")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2)   # 【建议】卡住时自动降 lr

    best_val_ppl = float("inf")
    print(f"\n{'Epoch':>6}  {'TrainLoss':>10}  {'TrainPPL':>10}  "
          f"{'ValLoss':>10}  {'ValPPL':>10}  {'LR':>9}")
    print("-" * 72)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_ppl = run_epoch(model, train_loader, criterion, optimizer,
                                    device, train=True)
        va_loss, va_ppl = run_epoch(model, val_loader, criterion, None,
                                    device, train=False)

        lr_now = optimizer.param_groups[0]["lr"]

        # 【修复点 4】只在 val PPL 创新低时保存，而不是每轮都覆盖
        marker = ""
        if not math.isnan(va_ppl) and va_ppl < best_val_ppl:
            best_val_ppl = va_ppl
            torch.save({
                "model_state": model.state_dict(),
                "char2idx": char2idx,
                "idx2char": idx2char,
                "args": vars(args),
            }, args.save)
            marker = "  *best"

        if not math.isnan(va_ppl):
            scheduler.step(va_ppl)

        print(f"{epoch:>6}  {tr_loss:>10.4f}  {tr_ppl:>10.2f}  "
              f"{va_loss:>10.4f}  {va_ppl:>10.2f}  {lr_now:>9.2e}{marker}")

    print(f"\n训练完成。最佳验证 PPL: {best_val_ppl:.2f}  已保存至 {args.save}")

    # ---------- 生成示例 ----------
    if args.sample:
        state = torch.load(args.save, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state"])
        print(f"\n--- 生成示例（seed = {args.sample!r}）---")
        print(generate(model, idx2char, start=args.sample, device=device))


if __name__ == "__main__":
    main()

# cd "D:\BaiduNetdiskDownload\PYTHON学习视频\第五周 大模型初探\模型transform练习对话"
# python language_model_fixed.py --mode sft --pretrain_ckpt best_model_transform.pt --lr 5e-4 --save sft_best.pt
