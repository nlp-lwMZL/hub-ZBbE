import pandas as pd
import os
import csv
import math
import glob
import argparse
import random
import time
from collections import Counter
from torch.utils.data import DataLoader, TensorDataset
from torch import nn, optim
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VACAB_FILE = os.path.join(BASE_DIR, "vocab.csv")
default_corpus = os.path.join(BASE_DIR, "corpus_train.txt")
UNK_ID = 1
PAD_ID = 0

ap = argparse.ArgumentParser(description="字符级中文语言模型训练（Decoder-only Transformer）")
ap.add_argument("--seed", type=int, default=42, help="随机种子")
ap.add_argument("--model", choices=["transformer", "lstm"], default="transformer",
                help="主干网络：只换这一个参数即可对比，数据切分/损失/指标完全不变")
ap.add_argument("--tie", action="store_true",
                help="输入/输出 embedding 共享权重（省参数；共享后按 GPT-2 方式小方差初始化）")
ap.add_argument("--corpus", default=default_corpus, help="语料路径（支持通配符）")
ap.add_argument("--max_chars", type=int, default=0, help="只用前 N 个字符（CPU 慢时截断）")
ap.add_argument("--max_vocab", type=int, default=4000, help="字表最多保留多少个字（0=不限）")
ap.add_argument("--batch-size", type=int, default=256, help="批大小")
ap.add_argument("--epochs", type=int, default=6, help="训练轮数")
ap.add_argument("--d_model", type=int, default=192, help="词嵌入维度 d_model")
ap.add_argument("--num-heads", type=int, default=4, help="注意力头数")
ap.add_argument("--num-layers", type=int, default=4, help="层数（Transformer 块数 / LSTM 层数）")
ap.add_argument("--lstm-hidden", type=int, default=0,
                help="LSTM 隐藏维度（0=与 d_model 相同）；用于复刻基线 LSTM 的 embed=128/hidden=256 结构")
ap.add_argument("--dropout", type=float, default=0.1, help="dropout 比例")
ap.add_argument("--max-length", type=int, default=64, help="序列最大长度")
ap.add_argument("--stride", type=int, default=64,
                help="滑窗步长：>0 整段文本滑窗切块（=max_length 不重叠，=1 几乎逐字）；0 = 按行补齐")
ap.add_argument("--learning-rate", type=float, default=1e-3, help="学习率")
ap.add_argument("--val_ratio", type=float, default=0.05, help="验证集比例")
ap.add_argument("--no-mask", dest="no_mask", action="store_true", help="去掉因果掩码（泄漏实验）")
ap.add_argument("--no-pos", dest="no_pos", action="store_true", help="去掉位置编码")
ap.add_argument("--save", default="tf_best.pt", help="模型权重保存路径")
ap.add_argument("--ckpt", default=None, help="续写时读取的权重路径（默认与 --save 相同）")
ap.add_argument("--temperature", type=float, default=0.8, help="续写采样温度（<=0.01 退化为贪心解码）")
ap.add_argument("--generate", default=None, help="给一段开头，续写文本")
ap.add_argument("--n_chars", type=int, default=40, help="续写多少个字")
args = ap.parse_args()

SEED = args.seed
ARCH = args.model
TIE = args.tie
CORPUS = args.corpus
MAX_CHARS = args.max_chars
MAX_VOCAB = args.max_vocab
BATCH_SIZE = args.batch_size
EPOCH_SIZE = args.epochs
D_MODEL = args.d_model
NUM_OF_HEADS = args.num_heads
NUM_LAYERS = args.num_layers
LSTM_HIDDEN = args.lstm_hidden or args.d_model
DROPOUT = args.dropout
MAX_LENGTH = args.max_length
STRIDE = args.stride
LEARNING_RATE = args.learning_rate
VAL_RATIO = args.val_ratio
USE_CAUSAL = not args.no_mask
USE_POS = not args.no_pos
SAVE_FILE = args.save
CKPT_FILE = args.ckpt or args.save
TEMPERATURE = args.temperature
N_CHARS = args.n_chars

def load_corpus_lines(pattern, max_chars=0):
    """
    按通配符读取语料，返回非空行列表；max_chars > 0 时只保留前 N 个字符
    """
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"未找到语料文件：{pattern}")
    lines = []
    total = 0
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                if max_chars:
                    remain = max_chars - total
                    if remain <= 0:
                        return lines
                    line = line[:remain]
                lines.append(line)
                total += len(line)
    return lines

def build_vocab_file():
    """
    用训练语料构造字符表，按字符出现次数由高到低构造
    """
    lines = load_corpus_lines(CORPUS, MAX_CHARS)
    counter = Counter()
    for line in lines:
        counter.update(line)
    # 每行末尾一个换行符：滑窗把行拼成流时它就是句界符，必须进字表
    counter["\n"] += len(lines)
    with open(VACAB_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["<PAD>", 0])
        writer.writerow(["<UNK>", 1])
        for char, count in counter.most_common():
            writer.writerow([char, count])

def load_vacab_file():
    """
    读取字符表文件并构造字符表
    """
    chars = pd.read_csv(VACAB_FILE, header=None, usecols=[0])[0]
    return {char: idx for idx, char in enumerate(chars)}

def get_vocab():
    """
    获取字符表；max_vocab > 0 时只保留词表中频率最高的前 N 项（vocab.csv 本身按频率降序）
    """
    if not os.path.exists(VACAB_FILE):
        build_vocab_file()
    vocab = load_vacab_file()
    if MAX_VOCAB and MAX_VOCAB > 0:
        keep = max(MAX_VOCAB, 2)
        vocab = {char: idx for idx, char in enumerate(list(vocab.keys())[:keep])}
    if STRIDE > 0 and "\n" not in vocab:
        # 滑窗模式需要句界符：追加到末尾，已有字的 id 全部不变（不影响旧权重）
        vocab["\n"] = len(vocab)
    return vocab, len(vocab)

def load_data(lines, vocab, max_len=MAX_LENGTH, shuffle=True):
    """
    把文本行转换为 torch.DataLoader：每行补齐/截断到 max_len，x/y 同行错一位
    """
    seqs = [torch.LongTensor([vocab.get(c, UNK_ID) for c in t]) for t in lines]
    pad_sequence = []
    for seq in seqs:
        seq_len = len(seq)
        padding_count = max_len - seq_len
        if padding_count > 0:
            padding_array = torch.full((max_len,), PAD_ID, dtype=torch.long)
            padding_array[0:seq_len] = seq
            pad_sequence.append(padding_array)
        elif padding_count < 0:
            pad_sequence.append(seq[0:max_len])
        else:
            pad_sequence.append(seq)
    data = torch.stack(pad_sequence)
    x = data[:, :-1]
    y = data[:, 1:]
    return DataLoader(TensorDataset(x, y), batch_size=BATCH_SIZE, shuffle=shuffle)

def load_stream_data(lines, vocab, max_len=MAX_LENGTH, stride=STRIDE, shuffle=True):
    """
    滑窗切块：把所有行拼成一条 id 流，每隔 stride 取一个长度 max_len 的窗口，窗内 x/y 错一位。
    stride = max_len 时窗口不重叠；stride < max_len 时窗口重叠（样本变多，但相邻样本高度相关）。
    """
    ids = [torch.LongTensor([vocab.get(c, UNK_ID) for c in t]) for t in lines]
    sep_id = vocab.get("\n", UNK_ID)
    parts = []
    for seq in ids:
        parts.append(seq)
        parts.append(torch.LongTensor([sep_id]))   # 行尾补句界符：窗口跨行时能看见「这里换行了」
    stream = torch.cat(parts) if parts else torch.LongTensor([])
    if len(stream) <= max_len:
        return None
    starts = range(0, len(stream) - max_len + 1, stride)
    windows = torch.stack([stream[s:s + max_len] for s in starts])
    x = windows[:, :-1]
    y = windows[:, 1:]
    return DataLoader(TensorDataset(x, y), batch_size=BATCH_SIZE, shuffle=shuffle)

def build_loaders(vocab):
    """
    读取语料并按 val_ratio 切分训练集/验证集。
    stride = 0：按行补齐（每行一个样本）；stride > 0：整段文本滑窗（按字符流尾部切分，避免窗口泄漏）
    """
    lines = load_corpus_lines(CORPUS, MAX_CHARS)
    split = int(len(lines) * (1 - VAL_RATIO))
    if STRIDE > 0:
        train_loader = load_stream_data(lines[:split], vocab, MAX_LENGTH, STRIDE, shuffle=True)
        dev_loader = load_stream_data(lines[split:], vocab, MAX_LENGTH, STRIDE, shuffle=False) if split < len(lines) else None
        if train_loader is None:
            raise ValueError("语料太短，切不出任何窗口：请增大 --max_chars 或减小 --max-length")
        n_train = sum(len(t) for t in lines[:split])
        n_dev = sum(len(t) for t in lines[split:])
        print(f"切分: 滑窗(stride={STRIDE}, 窗宽={MAX_LENGTH})  训练 {n_train:,} 字 / {len(train_loader.dataset):,} 样本  "
              f"验证 {n_dev:,} 字 / {0 if dev_loader is None else len(dev_loader.dataset):,} 样本")
        return train_loader, dev_loader
    random.Random(SEED).shuffle(lines)
    split = int(len(lines) * (1 - VAL_RATIO))
    train_lines = lines[:split]
    dev_lines = lines[split:]
    train_loader = load_data(train_lines, vocab, shuffle=True)
    dev_loader = load_data(dev_lines, vocab, shuffle=False) if dev_lines else None
    print(f"切分: 按行补齐(窗宽={MAX_LENGTH})  训练 {len(train_lines):,} 行  验证 {len(dev_lines):,} 行")
    return train_loader, dev_loader

class LanguageMode(nn.Module):
    """
    主干可切换：transformer / lstm。
    换主干时只动 __init__ 里这一个分支和 forward 里的这一行，其余代码（数据切分、
    损失、指标）完全不变，所以两种架构的 PPL 是可以直接对比的。
    """
    def __init__(self, input_size, d_model, num_layers, num_heads, max_length,
                 dropout=0.1, use_pos=True, causal=True, arch="transformer", tie=False,
                 lstm_hidden=0):
        super().__init__()
        hidden = lstm_hidden or d_model
        if tie and hidden != d_model:
            raise ValueError("--tie 要求输出维度与 embed 维度相同，不能与 --lstm-hidden 同时使用")
        self.embedding = nn.Embedding(input_size, d_model, padding_idx=PAD_ID)
        self.use_pos = use_pos
        if use_pos:
            self.pos_embedding = nn.Embedding(max_length, d_model)
        self.arch = arch
        if arch == "lstm":
            # LSTM 自身按时间步递推，天然是因果的，不需要 mask
            self.rnn = nn.LSTM(
                d_model, hidden, num_layers=num_layers, batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
        else:
            transformer_encode = nn.TransformerEncoderLayer(
                d_model, num_heads, dim_feedforward=4 * d_model, dropout=dropout, activation="gelu", batch_first=True, norm_first=True
            )
            self.transformer = nn.TransformerEncoder(transformer_encode, num_layers, norm=nn.LayerNorm(d_model), enable_nested_tensor=False)
        self.liner = nn.Linear(hidden, input_size, bias=False) if tie else nn.Linear(hidden, input_size)
        if tie:
            # weight tying：输出头直接复用 embedding 那块参数（同一个 Parameter，不是拷贝）
            # 必须 bias=False，否则输出不再是纯点积 h·E[v]，共享关系被破坏
            self.liner.weight = self.embedding.weight
            with torch.no_grad():
                # 共享后梯度两路叠加，logits 量级 ≈ sqrt(d)·std，用默认 std=1 会直接饱和
                # （初始 loss 会飙到 19 而不是 ln(V)），所以按 GPT-2 用小方差初始化
                nn.init.normal_(self.embedding.weight, std=0.02)
                if use_pos:
                    nn.init.normal_(self.pos_embedding.weight, std=0.02)
                # padding_idx 对应的行必须保持全零（重新初始化会把它写脏，而它不会再有梯度）
                self.embedding.weight[PAD_ID].zero_()
        self.causal = causal
        self.max_length = max_length
        self.loss = nn.functional.cross_entropy

    def input_embedding(self, x):
        y_em = self.embedding(x)
        if not self.use_pos:
            return y_em
        pos = torch.arange(y_em.size(1), device=y_em.device)
        return y_em + self.pos_embedding(pos)

    def generate_causal_mask(self, seq_len, device):
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        return mask  # True 表示屏蔽

    def forward(self, x, y=None):
        y_em = self.input_embedding(x)
        if self.arch == "lstm":
            y_trans, _ = self.rnn(y_em)
        else:
            pad_mask = (x == PAD_ID)
            causal_mask = self.generate_causal_mask(y_em.size(1), y_em.device) if self.causal else None
            y_trans = self.transformer(y_em, causal_mask, pad_mask)
        y_pred = self.liner(y_trans)
        if y is None:
            return y_pred
        else:
            V = y_pred.size(-1)
            loss = self.loss(y_pred.reshape(-1, V), y.reshape(-1), ignore_index=PAD_ID)
            return loss

def eval(mode, validation_loader):
    """
    一次前向同时给出两种 PPL：
    全位置 = 所有非 PAD 位置的交叉熵平均（常说的 PPL）
    末位   = 每个窗口最后一个有效位置的平均（"只看下一个字"的口径）
    """
    mode.eval()
    total_loss = 0.0
    total_tokens = 0
    last_loss = 0.0
    last_tokens = 0
    with torch.no_grad():
        for x_batch, y_batch in validation_loader:
            logits = mode(x_batch)
            V = logits.size(-1)
            per_token = nn.functional.cross_entropy(
                logits.reshape(-1, V), y_batch.reshape(-1),
                ignore_index=PAD_ID, reduction="none",
            ).reshape(y_batch.shape)
            valid = (y_batch != PAD_ID)
            total_loss += per_token.sum().item()
            total_tokens += int(valid.sum())
            # 每个样本最后一个有效位置的下标（滑窗模式下即窗口末位）
            last_idx = valid.sum(1) - 1
            ok = last_idx >= 0
            rows = torch.arange(y_batch.size(0))[ok]
            picked = per_token[rows, last_idx[ok]]
            last_loss += picked.sum().item()
            last_tokens += int(ok.sum())
    average_loss = total_loss / max(1, total_tokens)
    last_average = last_loss / max(1, last_tokens)
    return {
        "loss": average_loss, "ppl": math.exp(average_loss),
        "last_loss": last_average, "last_ppl": math.exp(last_average),
    }

def save_checkpoint(mode, vocab):
    """
    保存权重时一并记录结构超参，load 时才能重建同构模型
    """
    torch.save({
        "state_dict": mode.state_dict(),
        "vocab": vocab,
        "d_model": D_MODEL,
        "num_layers": NUM_LAYERS,
        "lstm_hidden": LSTM_HIDDEN,
        "num_heads": NUM_OF_HEADS,
        "arch": ARCH,
        "tie": TIE,
        "max_length": MAX_LENGTH,
        "dropout": DROPOUT,
        "use_pos": USE_POS,
        "causal": USE_CAUSAL,
    }, SAVE_FILE)
    print(f"已保存权重 → {SAVE_FILE}")

def generate():
    """
    续写：把 args.generate 这段开头喂进去，逐字往外蹦 n_chars 个新字
    """
    if not os.path.exists(CKPT_FILE):
        print(f"未找到权重文件 {CKPT_FILE}，请先训练再续写。")
        return
    ckpt = torch.load(CKPT_FILE, map_location="cpu", weights_only=False)
    vocab = ckpt["vocab"]
    idx2char = {idx: char for char, idx in vocab.items()}
    mode = LanguageMode(
        len(vocab), ckpt["d_model"], ckpt["num_layers"], ckpt["num_heads"],
        ckpt["max_length"], ckpt["dropout"], ckpt["use_pos"], ckpt["causal"],
        arch=ckpt.get("arch", "transformer"),
        tie=ckpt.get("tie", False),
        lstm_hidden=ckpt.get("lstm_hidden", 0),
    )
    mode.load_state_dict(ckpt["state_dict"])
    mode.eval()
    seq_len = ckpt["max_length"] - 1
    ids = [vocab.get(ch, UNK_ID) for ch in args.generate]
    seq = ids[-seq_len:]
    generated = []
    with torch.no_grad():
        for _ in range(N_CHARS):
            x = torch.LongTensor([seq])
            logits = mode(x)[0, -1]
            # PAD 不参与训练，共享权重时它的 logit 恒为 0，不屏蔽就有可能被采样出来
            logits[PAD_ID] = float("-inf")
            if TEMPERATURE <= 0.01:
                # 温度趋近 0 等价于贪心解码，结果完全确定
                nxt = int(logits.argmax())
            else:
                # 温度采样：logits/T 再 softmax，T<1 更尖（保守），T>1 更平（发散）
                probs = torch.softmax(logits / TEMPERATURE, dim=-1)
                nxt = int(torch.multinomial(probs, 1))
            generated.append(nxt)
            seq.append(nxt)
            if len(seq) > seq_len:
                seq.pop(0)
    new_text = "".join(idx2char.get(i, "") for i in generated)
    print(f"输入: {args.generate}")
    print(f"续写: {new_text}")
    print(f"成文: {args.generate}{new_text}")

def main():
    torch.manual_seed(SEED)
    random.seed(SEED)
    vocab, vocab_length = get_vocab()
    train_loader, validation_loader = build_loaders(vocab)
    mode = LanguageMode(vocab_length, D_MODEL, NUM_LAYERS, NUM_OF_HEADS, MAX_LENGTH,
                        DROPOUT, USE_POS, USE_CAUSAL, arch=ARCH, tie=TIE,
                        lstm_hidden=args.lstm_hidden)
    total_params = sum(p.numel() for p in mode.parameters())
    print(f"语料: {CORPUS}  词表: {vocab_length}  参数: {total_params:,}  主干: {ARCH}  共享权重: {'有' if TIE else '无'}  "
          f"位置编码: {'有' if USE_POS else '无'}  因果掩码: {'有' if USE_CAUSAL else '无'}")
    optimizer = optim.Adam(mode.parameters(), lr=LEARNING_RATE)
    best_ppl = float("inf")
    last_epoch = {}
    for epoch in range(EPOCH_SIZE):
        mode.train()
        total_loss = 0.0
        total_tokens = 0
        t0 = time.time()
        for x_batch, y_batch in train_loader:
            loss = mode(x_batch, y_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            n_tokens = int((y_batch != PAD_ID).sum())
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens
        elapsed = time.time() - t0
        average_loss = total_loss / max(1, total_tokens)
        perplexity = math.exp(average_loss)
        tok_per_s = total_tokens / max(1e-9, elapsed)
        print(f"第{epoch + 1}轮训练集损失值：{average_loss:.5f}，困惑度：{perplexity}  "
              f"吞吐：{tok_per_s:,.0f} tok/s  用时 {elapsed:.1f}s")
        if validation_loader is not None:
            m = eval(mode, validation_loader)
            print(f"第{epoch + 1}轮验证集损失值：{m['loss']:.5f}，困惑度：{m['ppl']:.2f}  "
                  f"末位困惑度：{m['last_ppl']:.2f}")
            last_epoch = {**m, "params": total_params, "tok_per_s": tok_per_s, "epochs": epoch + 1}
            if m["ppl"] < best_ppl:
                best_ppl = m["ppl"]
                save_checkpoint(mode, vocab)
        else:
            last_epoch = {"ppl": perplexity, "last_ppl": float("nan"),
                          "params": total_params, "tok_per_s": tok_per_s, "epochs": epoch + 1}
            save_checkpoint(mode, vocab)
    if last_epoch:
        print(f"汇总 | {ARCH} | 参数 {last_epoch['params']:,} | 全位置 PPL {last_epoch['ppl']:.2f} "
              f"| 末位 PPL {last_epoch['last_ppl']:.2f} | 吞吐 {last_epoch['tok_per_s']:,.0f} tok/s "
              f"| {last_epoch['epochs']} 轮")
    if best_ppl < float("inf"):
        print(f"最佳验证困惑度：{best_ppl:.4f}")

if __name__ == "__main__":
    # 一条命令两用：带 --generate 就只做续写（不再训练），否则进入训练
    if args.generate:
        generate()
    else:
        main()
