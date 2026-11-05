"""
字符级语言模型训练脚本（Transformer 版），含 PPL 计算。

用法:
    python transformer_model.py --epochs 20
    python transformer_model.py --embed_dim 256 --hidden_dim 1024 --num_layers 4 --num_heads 8
"""

import math
import time
import argparse
import glob
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ─────────────────────────── 数据 ───────────────────────────

def load_corpus(pattern="*.txt"):
    texts = []
    for path in glob.glob(pattern):
        with open(path, encoding="utf-8", errors="ignore") as corpus_file:
            texts.append(corpus_file.read())
    return "".join(texts)


def build_vocab(text, max_vocab=None):
    """
    构建字表：按出现频率取前 max_vocab 个字（None = 全部字符）。
    语料里的生僻字超出字表后会被 CharDataset 过滤掉 —— 字表越小模型越小，
    但覆盖不足会丢信息，这是词表大小的取舍。
    """
    from collections import Counter
    counter = Counter(text)
    if max_vocab:
        chars = [char for char, _count in counter.most_common(max_vocab)]
    else:
        chars = sorted(set(text))
    char2idx = {char: index for index, char in enumerate(chars)}
    idx2char = {index: char for char, index in char2idx.items()}
    return char2idx, idx2char


class CharDataset(Dataset):
    """
    字符级语言模型的训练样本：输入 inputs = 连续 seq_len 个字符，目标 targets = 往后错一位。

    stride 控制切块间隔：
        stride=1  —— 每个字符位置都当一个样本（样本大量重叠，一期语料能切出上百万个样本）
        stride=seq_len —— 不重叠切块（标准做法，样本数 = 语料长度 / seq_len）
    语料一大，stride=1 会让一轮训练变得极慢，这不是模型慢，而是样本数被放大了 seq_len 倍。
    """

    def __init__(self, text, char2idx, seq_len, stride=1):
        self.seq_len = seq_len
        self.stride = max(1, stride)
        token_ids = [char2idx[char] for char in text if char in char2idx]
        self.data = torch.tensor(token_ids, dtype=torch.long)

    def __len__(self):
        return max(0, (len(self.data) - self.seq_len - 1) // self.stride + 1)

    def __getitem__(self, index):
        start = index * self.stride
        inputs = self.data[start: start + self.seq_len]
        targets = self.data[start + 1: start + self.seq_len + 1]
        return inputs, targets


# ─────────────────────────── 模型 ───────────────────────────

class LM(nn.Module):
    """
      1. 位置嵌入 —— 自注意力对输入顺序不敏感，不注入位置信息，「猫追狗」和「狗追猫」
         对模型就没区别了。
      2. 因果 mask —— nn.Transformer 默认双向（每个位置能看整句），语言模型必须让
         位置 i 只能看 j ≤ i，否则预测下一个字时就直接偷看了答案。
         mask 是 bool 型，PyTorch 约定 True = 屏蔽，所以取严格上三角。
    """

    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, num_heads, dropout, seq_len):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim 必须能被注意力头数整除"

        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos = nn.Embedding(seq_len, embed_dim)          # 可学习位置嵌入

        # 完整 Transformer：编码器 1 层 + 解码器 num_layers 层。
        # 每层内部已含多头注意力、FFN、残差连接、LayerNorm、dropout，不用自己写。
        self.transformer = nn.Transformer(
            d_model=embed_dim,              # 模型宽度 = 词嵌入维度
            nhead=num_heads,
            num_encoder_layers=1,           # 解码器才是主角，编码器只需存在
            num_decoder_layers=num_layers,
            dim_feedforward=hidden_dim,     # FFN 内部维度，一般取 embed_dim 的 4 倍
            dropout=dropout,
            activation="gelu",
            batch_first=True,               # 输入按 (batch, 序列长度, 维度)，和我们的数据形状一致
            norm_first=True,                # Pre-LN，比原论文的 Post-LN 好训
        )

        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_dim, vocab_size)

        # 因果 mask 只和 seq_len 有关，建一次复用；forward 里按当前长度切片
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask", causal_mask, persistent=False)

    def forward(self, token_ids):
        # token_ids: (batch, 序列长度) 字符 id
        max_context = self.pos.num_embeddings
        token_ids = token_ids[:, -max_context:]   # 上下文超过 seq_len 时只保留最近的一段
        seq_length = token_ids.shape[1]

        embeddings = self.embed(token_ids)
        embeddings = embeddings + self.pos(torch.arange(seq_length, device=token_ids.device))
        embeddings = self.drop(embeddings)

        # 解码器标准用法是 src=编码器输出、tgt=已生成序列；
        # 这里 src 和 tgt 是同一段序列，三处 mask 全因果，保证任何位置都看不到未来。
        causal_mask = self.causal_mask[:seq_length, :seq_length]
        outputs = self.transformer(
            src=embeddings,
            tgt=embeddings,
            src_mask=causal_mask,
            tgt_mask=causal_mask,
            memory_mask=causal_mask,
        )

        return self.fc(self.drop(outputs))   # (batch, 序列长度, 词表大小)


# ─────────────────────────── 训练 / 评估 ───────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, train=True, amp_dtype=None):
    model.train(train)
    # 累积损失留在设备上，只在整轮结束时 .item() 一次。
    # 每步都 loss.item() 会强制一次 GPU→CPU 同步，把异步执行的优势全部抵消。
    total_loss = torch.zeros((), device=device)
    total_tokens = 0

    # non_blocking 只有在锁页内存（CUDA + pin_memory）上才安全；
    # MPS 上异步拷贝会和 CPU 侧缓冲区复用打架，导致模型读到的其实是别的 batch
    non_blocking = device.type == "cuda"

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=non_blocking)
        targets = targets.to(device, non_blocking=non_blocking)

        # autocast：矩阵乘走低精度，LayerNorm/softmax 等仍保持 fp32
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = model(inputs)
            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            # 梯度裁剪：Transformer 偶尔出尖峰梯度，不裁可能一步就把权重带崩
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # criterion 返回的是本 batch 的平均损失，乘 token 数还原成总和再累加
        total_loss = total_loss + loss.detach() * targets.numel()
        total_tokens += targets.numel()

    average_loss = (total_loss / max(1, total_tokens)).item()   # 语料太小、切不出样本时避免除零
    perplexity = math.exp(min(average_loss, 50))                # 防止极端 loss 让 exp 溢出
    return average_loss, perplexity


# ─────────────────────────── 主函数 ───────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=20)
    parser.add_argument("--seq_len",    type=int,   default=64, help="上下文长度，也是位置嵌入上限")
    parser.add_argument("--stride",     type=int,   default=64, help="切块间隔：1=密集重叠，等于 seq_len=不重叠")
    parser.add_argument("--batch_size", type=int,   default=256, help="GPU 上 batch 越大利用率越高，128→256 实测吞吐 +15%")
    parser.add_argument("--embed_dim",  type=int,   default=128, help="词嵌入维度，同时也是 Transformer 的模型宽度")
    parser.add_argument("--hidden_dim", type=int,   default=512, help="Transformer FFN 内部维度，一般是 embed_dim 的 4 倍")
    parser.add_argument("--num_layers", type=int,   default=2,   help="Transformer 层数")
    parser.add_argument("--num_heads",  type=int,   default=8,   help="注意力头数，需整除 embed_dim")
    parser.add_argument("--dropout",    type=float, default=0.1)
    parser.add_argument("--lr",         type=float, default=3e-4, help="Transformer 比 RNN 对学习率更敏感，默认调小一号")
    parser.add_argument("--val_ratio",  type=float, default=0.05)
    parser.add_argument("--max_vocab",  type=int,   default=4000, help="字表最大字数，0=不限")
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--corpus",     default="*.txt")
    parser.add_argument("--save",       default="best_model_transformer.pt")
    parser.add_argument("--amp",        default="auto", choices=["auto", "off", "bf16"],
                        help="混合精度：auto=仅在 CUDA 上开 bf16（MPS 上实测反而更慢）")
    parser.add_argument("--compile",    action="store_true",
                        help="用 torch.compile 编译模型（仅 CUDA 有效，MPS 后端不支持）")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # CUDA 上允许 TF32：fp32 矩阵乘改用 Tensor Core，通常能再快 1.5~2 倍，精度损失很小
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    amp_dtype = None
    if args.amp == "bf16" or (args.amp == "auto" and device.type == "cuda"):
        amp_dtype = torch.bfloat16

    print(f"device: {device}  model: TRANSFORMER  amp: {amp_dtype or 'off'}")

    # 数据准备
    text = load_corpus(args.corpus)
    if not text:
        raise FileNotFoundError("未找到任何 .txt 文件，请确认路径正确。")
    print(f"语料字符数: {len(text):,}")

    char2idx, idx2char = build_vocab(text, args.max_vocab or None)
    vocab_size = len(char2idx)
    print(f"词表大小: {vocab_size}")

    lines = text.splitlines()
    random.shuffle(lines)
    split = int(len(lines) * (1 - args.val_ratio))
    train_text = "\n".join(lines[:split])
    val_text   = "\n".join(lines[split:])

    train_dataset = CharDataset(train_text, char2idx, args.seq_len, args.stride)
    val_dataset   = CharDataset(val_text,   char2idx, args.seq_len, args.stride)

    # pin_memory 把 batch 放在锁页内存里，配合 non_blocking=True 可与 GPU 计算重叠（CUDA 上有效）
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              drop_last=True, pin_memory=pin_memory)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=True,
                              drop_last=True, pin_memory=pin_memory)
    if len(val_loader) == 0:
        # 语料/验证集太小（切不出一个完整 batch）时退回训练集评估，否则 PPL 会显示成 1.00
        print("提示：验证集太小，改用训练集评估")
        val_loader = train_loader

    # 模型
    model = LM(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        seq_len=args.seq_len,
    ).to(device)

    # torch.compile 把众多小算子融合成少量大 kernel；inductor 后端目前不支持 MPS，只在 CUDA 上启用
    if args.compile:
        if device.type == "cuda":
            model = torch.compile(model)
            print("已启用 torch.compile（首轮会有几十秒编译开销）")
        else:
            print("提示：torch.compile 的 inductor 后端不支持 MPS/CPU，已跳过")

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"模型参数量: {parameter_count:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_ppl = float("inf")

    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Train PPL':>10}  {'Val Loss':>10}  {'Val PPL':>10}  {'Time':>7}s")
    print("-" * 64)

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        train_loss, train_ppl = run_epoch(model, train_loader, criterion, optimizer, device,
                                          train=True, amp_dtype=amp_dtype)
        with torch.no_grad():
            val_loss, val_ppl = run_epoch(model, val_loader, criterion, optimizer, device,
                                          train=False, amp_dtype=amp_dtype)
        epoch_seconds = time.perf_counter() - epoch_start

        marker = "  *" if val_ppl < best_val_ppl else ""
        if val_ppl < best_val_ppl:
            best_val_ppl = val_ppl
            torch.save({
                "model": "transformer",
                "model_state": model.state_dict(),
                "char2idx": char2idx,
                "idx2char": idx2char,
                "args": vars(args),
            }, args.save)

        print(f"{epoch:>6}  {train_loss:>10.4f}  {train_ppl:>10.2f}  {val_loss:>10.4f}  "
              f"{val_ppl:>10.2f}  {epoch_seconds:>7.1f}{marker}")

    print(f"\n训练完成。最佳验证 PPL: {best_val_ppl:.2f}  已保存至 {args.save}")


if __name__ == "__main__":
    main()
