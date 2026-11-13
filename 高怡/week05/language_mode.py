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
default_corpus = os.path.join(BASE_DIR, "corpus_general.txt")
SFT_TRAIN = os.path.join(BASE_DIR, "sft_train.jsonl")
SFT_VAL = os.path.join(BASE_DIR, "sft_val.jsonl")

# answer 模式用的集外问题：刻意避开 SFT 训练集的题型，用来看是泛化还是只会背
OOD_QUESTIONS = [
    "中国的首都是哪里？",
    "3 加 5 等于多少？",
    "把“今天天气很好”翻译成英文。",
    "请用一句话说明什么是光合作用。",
    "为什么天空是蓝色的？",
]
UNK_ID = 1
PAD_ID = 0
IGNORE_ID = -100   # label 侧的「不算损失」哨兵：问题段 + 填充位都用它，x 侧仍用 PAD_ID

ap = argparse.ArgumentParser(description="字符级中文语言模型训练（Decoder-only Transformer）")
ap.add_argument("--seed", type=int, default=42, help="随机种子")
ap.add_argument("--model", choices=["transformer", "lstm"], default="transformer",
                help="主干网络：只换这一个参数即可对比，数据切分/损失/指标完全不变")
ap.add_argument("--tie", action="store_true",
                help="输入/输出 embedding 共享权重（省参数；共享后按 GPT-2 方式小方差初始化）")
ap.add_argument("--corpus", default=default_corpus, help="语料路径（支持通配符）")
ap.add_argument("--max-chars", type=int, default=0, help="只用前 N 个字符（CPU 慢时截断）")
ap.add_argument("--max-vocab", type=int, default=0, help="字表最多保留多少个字（0=不限，全量）")
ap.add_argument("--batch-size", type=int, default=256, help="批大小")
ap.add_argument("--epochs", type=int, default=6, help="训练轮数")
ap.add_argument("--d-model", type=int, default=192, help="词嵌入维度 d_model")
ap.add_argument("--num-heads", type=int, default=4, help="注意力头数")
ap.add_argument("--num-layers", type=int, default=4, help="层数（Transformer 块数 / LSTM 层数）")
ap.add_argument("--lstm-hidden", type=int, default=0,
                help="LSTM 隐藏维度（0=与 d_model 相同）；用于复刻基线 LSTM 的 embed=128/hidden=256 结构")
ap.add_argument("--dropout", type=float, default=0.1, help="dropout 比例")
ap.add_argument("--max-length", type=int, default=260, help="序列最大长度")
ap.add_argument("--sft-max-length", type=int, default=64, help="微调时问题/回答各自的最大长度（整条序列最长 2N+3）")
ap.add_argument("--stride", type=int, default=64,
                help="滑窗步长：>0 整段文本滑窗切块（=max_length 不重叠，=1 几乎逐字）；0 = 按行补齐")
ap.add_argument("--learning-rate", type=float, default=1e-3, help="学习率")
ap.add_argument("--val-ratio", type=float, default=0.05, help="验证集比例")
ap.add_argument("--no-mask", dest="no_mask", action="store_true", help="去掉因果掩码（泄漏实验）")
ap.add_argument("--no-pos", dest="no_pos", action="store_true", help="去掉位置编码")
ap.add_argument("--save", default=None, help="模型权重保存路径（默认：微调 sft_best.pt，其余 tf_best.pt）")
ap.add_argument("--ckpt", default=None, help="续写时读取的权重路径（默认与 --save 相同）")
ap.add_argument("--sft-ckpt", default="tf_best.pt", help="微调时加载的预训练权重路径")
ap.add_argument("--mode", choices=["generate", "sft", "answer"], default=None,
                help="运行模式：generate=续写，sft=微调，answer=检验微调问答效果；不传则预训练")
ap.add_argument("--n-answer", type=int, default=5, help="answer 模式下从验证集抽几条问题")
ap.add_argument("--prompt", default="", help="续写开头（配合 --mode generate）")
ap.add_argument("--temperature", type=float, default=0.8, help="续写采样温度（<=0.01 退化为贪心解码）")
ap.add_argument("--n-chars", type=int, default=40, help="续写多少个字")
ap.add_argument("--top-p", type=float, default=0.8,
                help="answer 模式 top-p 核采样阈值（0~1，<=0 关闭核采样）")
ap.add_argument("--beam", type=int, default=3,
                help="answer 模式 beam search 宽度（1=纯贪心，与 --top-p 联合使用）")
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
SFT_MAX_LENGTH = args.sft_max_length
STRIDE = args.stride
LEARNING_RATE = args.learning_rate
VAL_RATIO = args.val_ratio
USE_CAUSAL = not args.no_mask
USE_POS = not args.no_pos
SAVE_FILE = args.save or ("sft_best.pt" if args.mode in ("sft", "answer") else "tf_best.pt")
CKPT_FILE = args.ckpt or SAVE_FILE
SFT_CKPT_FILE = args.sft_ckpt
TEMPERATURE = args.temperature
N_CHARS = args.n_chars
PROMPT = args.prompt

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
    用训练语料构造字符表，按字符出现次数由高到低构造。
    SFT 的问答文本也统计进来：空格、运算符、方括号这类字符在语料里没有，
    不加进来微调时它们全都会变成 <UNK>，问题直接被截断成残缺输入。
    """
    lines = load_corpus_lines(CORPUS, MAX_CHARS)
    counter = Counter()
    for line in lines:
        counter.update(line)
    # 每行末尾一个换行符：滑窗把行拼成流时它就是句界符，必须进字表
    counter["\n"] += len(lines)
    for path in (SFT_TRAIN, SFT_VAL):
        if os.path.exists(path):
            for texts in load_json_lines(path):
                for text in texts:
                    counter.update(text)
    with open(VACAB_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["<PAD>", 0])
        writer.writerow(["<UNK>", 1])
        writer.writerow(["<|user|>", 2])
        writer.writerow(["<|assistant|>", 3])
        writer.writerow(["<|end|>", 4])
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
    y = data[:, 1:].clone()
    y[y == PAD_ID] = IGNORE_ID   # 填充位不参与 loss（x 侧仍保留 PAD，供 attention mask 使用）
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

def load_json_lines(path):
    df = pd.read_json(path, lines=True)
    instruction = df["instruction"].to_list()
    output = df["output"].to_list()
    return instruction, output

def build_sft_data_loader(path, vocab, max_len=SFT_MAX_LENGTH, shuffle=True):
    """
    把 (instruction, output) 拼成对话样本：<|user|> 问题 <|assistant|> 回答 <|end|>
    问题与回答各自最长 max_len，加 3 个特殊 token，整条序列最长 2*max_len+3，不能超过模型窗口 MAX_LENGTH。
    x/y 同行错一位；问题段和填充段的 label 都是 IGNORE_ID(-100)，只在回答上算损失
    """
    if 2 * max_len + 3 > MAX_LENGTH:
        raise ValueError(f"问题/回答各 {max_len} 字 + 3 个特殊 token 超出了模型窗口，请把 --max-length 设为 >= {2 * max_len + 3}")
    instruction, output = load_json_lines(path)
    user_id = vocab.get("<|user|>", UNK_ID)
    assistant_id = vocab.get("<|assistant|>", UNK_ID)
    end_id = vocab.get("<|end|>", UNK_ID)
    seq_len = 2 * max_len + 3
    pad_sequence = []
    label_sequence = []
    for question, answer in zip(instruction, output):
        question_ids = [vocab.get(c, UNK_ID) for c in question][:max_len]
        prompt_ids = [user_id] + question_ids + [assistant_id]
        answer_ids = [vocab.get(c, UNK_ID) for c in answer][:max_len] + [end_id]
        ids = prompt_ids + answer_ids
        labels = [IGNORE_ID] * len(prompt_ids) + answer_ids
        padding_count = seq_len - len(ids)
        if padding_count > 0:
            ids = ids + [PAD_ID] * padding_count
            labels = labels + [IGNORE_ID] * padding_count
        pad_sequence.append(torch.LongTensor(ids))
        label_sequence.append(torch.LongTensor(labels))
    data = torch.stack(pad_sequence)
    labels = torch.stack(label_sequence)
    return DataLoader(TensorDataset(data[:, :-1], labels[:, 1:]), batch_size=BATCH_SIZE, shuffle=shuffle)

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
            loss = self.loss(y_pred.reshape(-1, V), y.reshape(-1), ignore_index=IGNORE_ID)
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
                ignore_index=IGNORE_ID, reduction="none",
            ).reshape(y_batch.shape)
            valid = (y_batch != IGNORE_ID)
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

def save_checkpoint(mode, vocab, path=SAVE_FILE):
    """
    保存权重时一并记录结构超参，load 时才能重建同构模型。
    所有超参从 mode 实例本身推断，避免 sft() 等二次训练场景与全局常量不一致。
    """
    liner = mode.liner
    is_tie = (liner.bias is None)
    emb = mode.embedding
    d_model = emb.embedding_dim
    max_length = mode.max_length           # 模型自身已存
    use_pos = mode.use_pos
    causal = mode.causal
    arch = mode.arch                        # "transformer" / "lstm"
    if arch == "lstm":
        num_layers = mode.rnn.num_layers
        num_heads = 0
        dropout = mode.rnn.dropout
        lstm_hidden = mode.rnn.hidden_size
    else:
        # TransformerEncoder: .layers 是 ModuleList[TransformerEncoderLayer]
        num_layers = len(mode.transformer.layers)
        num_heads = mode.transformer.layers[0].self_attn.num_heads
        # TransformerEncoderLayer 的 dropout 挂在 .dropout1 上（self_attn 的 dropout）
        dropout = mode.transformer.layers[0].dropout1.p if num_layers > 0 else 0.0
        lstm_hidden = 0
    torch.save({
        "state_dict": mode.state_dict(),
        "vocab": vocab,
        "d_model": d_model,
        "num_layers": num_layers,
        "lstm_hidden": lstm_hidden,
        "num_heads": num_heads,
        "arch": arch,
        "tie": is_tie,
        "max_length": max_length,
        "dropout": dropout,
        "use_pos": use_pos,
        "causal": causal,
    }, path)
    print(f"已保存权重 → {path}")

def generate():
    """
    续写：把 args.prompt 这段开头喂进去，逐字往外蹦 n_chars 个新字
    """
    if not PROMPT:
        print("未给出续写开头，请用 --prompt 指定")
        return
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
    ids = [vocab.get(ch, UNK_ID) for ch in PROMPT]
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
    print(f"输入: {PROMPT}")
    print(f"续写: {new_text}")
    print(f"成文: {PROMPT}{new_text}")

def answer():
    """
    检验 SFT 效果：从验证集抽几条（带参考答案）+ 几条集外问题
    BeamSearch解码 + 核采样，遇到 <|end|> 停，最多生成 SFT_MAX_LENGTH 个字
    """
    if not os.path.exists(CKPT_FILE):
        print(f"未找到权重文件 {CKPT_FILE}，请先 --mode sft 微调")
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
    user_id = vocab.get("<|user|>", UNK_ID)
    assistant_id = vocab.get("<|assistant|>", UNK_ID)
    end_id = vocab.get("<|end|>", UNK_ID)
    window = ckpt["max_length"] - 1

    def reply(question):
        q_ids = [vocab.get(c, UNK_ID) for c in question][:SFT_MAX_LENGTH]
        prompt = ([user_id] + q_ids + [assistant_id])[-window:]
        unknown = sum(1 for i in q_ids if i == UNK_ID)
        beam_w = max(1, args.beam)
        top_p = args.top_p

        def top_p_filter(probs, p):
            """标准 nucleus：累积概率首次超过 p 的位置也要保留。
            返回 (allowed_ids, allowed_probs)，两者等长，allowed_probs 已重新归一化。"""
            sorted_p, sorted_idx = torch.sort(probs, descending=True)
            cum = torch.cumsum(sorted_p, dim=-1)
            keep = cum <= p
            first_over = int((~keep).nonzero(as_tuple=True)[0][0]) if (~keep).any() else keep.size(-1)
            keep[:first_over + 1] = True
            allowed_ids = sorted_idx[keep]        # 原始词表 id
            allowed_probs = sorted_p[keep]
            allowed_probs = allowed_probs / allowed_probs.sum().clamp_min(1e-9)
            return allowed_ids, allowed_probs

        def finalize(beams):
            """长度归一化打分：score = Σ log prob / length^α，α=0.6 是 Google NMT 推荐值"""
            alpha = 0.6
            scored = []
            for seq, raw_score, finished in beams:
                out_len = len(seq) - len(prompt)
                if out_len <= 0:
                    continue
                norm_score = raw_score / (out_len ** alpha)
                scored.append((seq, norm_score, finished, out_len))
            # 优先选 finished；finished 之间按归一化分数排序
            finished = [b for b in scored if b[2]]
            ongoing = [b for b in scored if not b[2]]
            if finished:
                finished.sort(key=lambda t: t[1], reverse=True)
                return finished[0]
            ongoing.sort(key=lambda t: t[1], reverse=True)
            return ongoing[0] if ongoing else None

        with torch.no_grad():
            # beams: list of (seq_list, raw_score, finished_flag)
            beams = [(list(prompt), 0.0, False)]
            for step in range(SFT_MAX_LENGTH):
                candidates = []
                all_finished = True
                for seq, score, finished in beams:
                    if finished:
                        candidates.append((seq, score, finished))
                        continue
                    all_finished = False
                    logits = mode(torch.LongTensor([seq]))[0, -1]
                    logits[PAD_ID] = float("-inf")
                    probs = torch.softmax(logits, dim=-1)
                    if top_p > 0.0 and top_p < 1.0:
                        allowed_ids, allowed_probs = top_p_filter(probs, top_p)
                        sample_size = min(beam_w, allowed_ids.size(0))
                        sampled_pos = torch.multinomial(allowed_probs, sample_size, replacement=False)
                        sampled_ids = allowed_ids[sampled_pos]
                        sampled_lp = torch.log(allowed_probs[sampled_pos].clamp_min(1e-9))
                    else:
                        # 无 top-p 时走 top-k = beam_w（确定性）
                        k = min(beam_w, logits.size(-1))
                        topk_lp, topk_ids = torch.topk(torch.log(probs.clamp_min(1e-9)), k)
                        sampled_ids = topk_ids
                        sampled_lp = topk_lp
                    for nxt_id, lp in zip(sampled_ids.tolist(), sampled_lp.tolist()):
                        new_seq = seq + [nxt_id]
                        if len(new_seq) > window:
                            new_seq = new_seq[-window:]
                        done = (nxt_id == end_id)
                        candidates.append((new_seq, score + lp, done))
                if all_finished:
                    break
                # 按 raw score 保留 top beam_w（展开阶段用 raw score 即可）
                candidates.sort(key=lambda t: t[1], reverse=True)
                beams = candidates[:beam_w]

        chosen = finalize(beams)
        if chosen is None:
            return "", unknown
        out = chosen[0][len(prompt):]
        if end_id in out:
            out = out[:out.index(end_id)]
        return "".join(idx2char.get(i, "") for i in out), unknown

    instruction, output = load_json_lines(SFT_VAL)
    # 固定抽样，保证不同权重之间的对比是同一批题目
    random.seed(SEED)
    torch.manual_seed(SEED)
    picks = random.sample(range(len(instruction)), min(args.n_answer, len(instruction)))
    print(f"权重: {CKPT_FILE}  词表: {len(vocab)}")
    for n, i in enumerate(picks, 1):
        text, unknown = reply(instruction[i])
        print(f"\n【验证集 {n}】{instruction[i]}")
        print(f"  参考: {output[i]}")
        print(f"  生成: {text}" + (f"   （问题里 {unknown} 个字不在词表）" if unknown else ""))
    print("\n" + "-" * 18 + " 集外问题 " + "-" * 18)
    for n, question in enumerate(OOD_QUESTIONS, 1):
        text, unknown = reply(question)
        print(f"\n【集外 {n}】{question}")
        print(f"  生成: {text}" + (f"   （问题里 {unknown} 个字不在词表）" if unknown else ""))

def main():
    torch.manual_seed(SEED)
    random.seed(SEED)
    if args.ckpt:
        # 续训：结构超参和字表都以 ckpt 为准，否则权重形状对不上
        if not os.path.exists(CKPT_FILE):
            print(f"未找到权重文件 {CKPT_FILE}，请检查 --ckpt 路径")
            return
        ckpt = torch.load(CKPT_FILE, map_location="cpu", weights_only=False)
        vocab = ckpt["vocab"]
        vocab_length = len(vocab)
        if ckpt["max_length"] != MAX_LENGTH:
            raise ValueError(f"权重窗口 {ckpt['max_length']} 与 --max-length {MAX_LENGTH} 不一致，续训需要相同窗口")
        train_loader, validation_loader = build_loaders(vocab)
        mode = LanguageMode(vocab_length, ckpt["d_model"], ckpt["num_layers"], ckpt["num_heads"],
                            ckpt["max_length"], ckpt["dropout"], ckpt["use_pos"], ckpt["causal"],
                            arch=ckpt.get("arch", "transformer"), tie=ckpt.get("tie", False),
                            lstm_hidden=ckpt.get("lstm_hidden", 0))
        mode.load_state_dict(ckpt["state_dict"])
        print(f"已加载权重 ← {CKPT_FILE}")
    else:
        vocab, vocab_length = get_vocab()
        train_loader, validation_loader = build_loaders(vocab)
        mode = LanguageMode(vocab_length, D_MODEL, NUM_LAYERS, NUM_OF_HEADS, MAX_LENGTH,
                            DROPOUT, USE_POS, USE_CAUSAL, arch=ARCH, tie=TIE,
                            lstm_hidden=args.lstm_hidden)
    total_params = sum(p.numel() for p in mode.parameters())
    print(f"语料: {CORPUS}  词表: {vocab_length}  参数: {total_params:,}  主干: {mode.arch}  共享权重: {'有' if mode.liner.bias is None else '无'}  "
          f"位置编码: {'有' if mode.use_pos else '无'}  因果掩码: {'有' if mode.causal else '无'}")
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
            n_tokens = int((y_batch != IGNORE_ID).sum())
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

def sft():
    torch.manual_seed(SEED)
    random.seed(SEED)
    if not os.path.exists(SFT_CKPT_FILE):
        print(f"未找到预训练权重 {SFT_CKPT_FILE}，请先预训练或用 --sft-ckpt 指定")
        return
    # 结构超参和字表都以 ckpt 为准：微调必须和预训练用同一套 id 与窗口，否则权重对不上
    ckpt = torch.load(SFT_CKPT_FILE, map_location="cpu", weights_only=False)
    vocab = ckpt["vocab"]
    vocab_length = len(vocab)
    if ckpt["max_length"] < 2 * SFT_MAX_LENGTH + 3:
        raise ValueError(f"预训练窗口 {ckpt['max_length']} 装不下 2*{SFT_MAX_LENGTH}+3，请减小 --sft-max-length")
    train_loader = build_sft_data_loader(SFT_TRAIN, vocab)
    val_loader = build_sft_data_loader(SFT_VAL, vocab, shuffle=False)
    mode = LanguageMode(vocab_length, ckpt["d_model"], ckpt["num_layers"], ckpt["num_heads"],
                        ckpt["max_length"], ckpt["dropout"], ckpt["use_pos"], ckpt["causal"],
                        arch=ckpt.get("arch", "transformer"), tie=ckpt.get("tie", False),
                        lstm_hidden=ckpt.get("lstm_hidden", 0))
    mode.load_state_dict(ckpt["state_dict"])
    total_params = sum(p.numel() for p in mode.parameters())
    print(f"已加载预训练权重 ← {SFT_CKPT_FILE}")
    print(f"语料: {SFT_TRAIN}  词表: {vocab_length}  参数: {total_params:,}  主干: {ckpt.get('arch', 'transformer')}  "
          f"共享权重: {'有' if ckpt.get('tie', False) else '无'}  窗口: {ckpt['max_length']}")
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
            n_tokens = int((y_batch != IGNORE_ID).sum())
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens
        elapsed = time.time() - t0
        average_loss = total_loss / max(1, total_tokens)
        perplexity = math.exp(average_loss)
        tok_per_s = total_tokens / max(1e-9, elapsed)
        print(f"第{epoch + 1}轮训练集损失值：{average_loss:.5f}，困惑度：{perplexity}  "
              f"吞吐：{tok_per_s:,.0f} tok/s  用时 {elapsed:.1f}s")
        if val_loader is not None:
            m = eval(mode, val_loader)
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
    # --mode generate 续写，--mode sft 微调，--mode answer 检验问答，不传则预训练
    if args.mode == "generate":
        generate()
    elif args.mode == "sft":
        sft()
    elif args.mode == "answer":
        answer()
    else:
        main()
