
import math
import argparse
import glob
import os
import random
import time
import json

import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformer_lm import TransformerLM

HERE = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────── SFT 特殊 token ───────────────────────────
# SFT（监督微调）和纯 LM 的本质区别：数据是「(指令, 回答) 配对」，
# 训练时只在「回答」部分算 loss，「指令」部分用 -100 屏蔽掉。
# 这些特殊 token 用来标记对话结构，模型靠它们分清「哪里是用户说的、哪里该我接话」。
# 它们被放在字表最前面（id 0~5），保证任何语料字符的 id 都大于它们，
# 这样 build_vocab 时不会冲突。
SPECIAL_TOKENS = {
    "pad":       "<|pad|>",        # 0: 填充，补齐 batch 长度，loss 里也屏蔽
    "begin":     "<|begin|>",      # 1: 一段对话的开头
    "system":    "<|system|>",     # 2: 系统提示（人设/全局规则），可选
    "user":      "<|user|>",       # 3: 用户指令开始
    "assistant": "<|assistant|>",  # 4: 助手回答开始（也是 loss 开始计算的位置）
    "end":       "<|end|>",        # 5: 一轮结束（回答以它收尾，生成时遇到它就停）
}
SPECIAL_TOKEN_LIST = ["<|pad|>", "<|begin|>", "<|system|>", "<|user|>", "<|assistant|>", "<|end|>"]

# ─────────────────────────── 数据 ───────────────────────────

def load_corpus(pattern="*.txt"):
    texts = []
    for path in glob.glob(pattern):
        with open(path, encoding="utf-8", errors="ignore") as f:
            texts.append(f.read())
    return "".join(texts)


def build_vocab(text, max_vocab=None, add_special=True):
    """
    构建字表：按出现频率取前 max_vocab 个字（None = 全部字符）。
    语料里的生僻字超出字表后会被 CharDataset 过滤掉 —— 字表越小模型越小，
    但覆盖不足会丢信息，这是词表大小的取舍。

    add_special=True 时把 6 个 SFT 特殊 token 放在字表最前面（id 0~5），
    普通字符从 id 6 开始。这样做的好处：
      1. 特殊 token 的 id 固定，不会因为语料变化而漂移；
      2. max_vocab 限制的是「普通字符」的数量，特殊 token 不占这个配额
         （否则字表一大，特殊 token 可能被挤出前 max_vocab，模型就认不出对话标记了）。
    """
    from collections import Counter
    counter = Counter(text)
    chars = [c for c, _ in counter.most_common(max_vocab)] if max_vocab else sorted(set(text))
    if add_special:
        # 特殊 token 在前，普通字符在后。enumerate 从 0 开始，所以特殊 token 拿到 0~5。
        all_tokens = SPECIAL_TOKEN_LIST + chars
    else:
        all_tokens = chars
    char2idx = {c: i for i, c in enumerate(all_tokens)}
    idx2char = {i: c for c, i in char2idx.items()}
    return char2idx, idx2char


def load_sft_data(path):
    """
    加载 SFT 数据：JSONL 格式，每行一个 JSON，至少包含 instruction 和 output 两个字段。
    也兼容 input/output 字段（有的数据集把指令叫 input）。
    可选 system 字段作为系统提示。

    示例一行：
        {"instruction": "解释什么是SFT", "output": "SFT是监督微调，用标注好的指令-回答对训练模型..."}

    返回 [(system, instruction, output), ...]，system 可能为空字符串。
    """
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            instruction = obj.get("instruction", obj.get("input", "")).strip()
            output = obj.get("output", obj.get("response", "")).strip()
            system = obj.get("system", "").strip()
            if not output:   # 没有回答的样本跳过（SFT 必须有监督信号）
                continue
            pairs.append((system, instruction, output))
    return pairs


class CharDataset(Dataset):
    """
    字符级语言模型的训练样本：输入 x = 连续 seq_len 个字符，目标 y = 往后错一位。

    stride 控制切块间隔：
        stride=1  —— 每个字符位置都当一个样本（样本大量重叠，一期语料能切出上百万个样本）
        stride=seq_len —— 不重叠切块（标准做法，样本数 = 语料长度 / seq_len）
    语料一大，stride=1 会让一轮训练变得极慢，这不是模型慢，而是样本数被放大了 seq_len 倍。
    """

    def __init__(self, text, char2idx, seq_len, stride=1):
        self.seq_len = seq_len
        self.stride = max(1, stride)
        ids = [char2idx[c] for c in text if c in char2idx]
        self.data = torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        return max(0, (len(self.data) - self.seq_len - 1) // self.stride + 1)

    def __getitem__(self, idx):
        start = idx * self.stride
        x = self.data[start: start + self.seq_len]
        y = self.data[start + 1: start + self.seq_len + 1]
        return x, y


class SFTDataset(Dataset):
    """
    SFT 数据集：每个样本是一段对话，格式为

        <|begin|> [<|system|>系统提示<|end|>] <|user|>指令<|end|> <|assistant|>回答<|end|>

    【SFT 和 LM 的核心差别就在这里 —— loss 掩码】
        LM 模式：整段文本每个位置都算 loss（预测下一个字）。
        SFT 模式：只在「回答」部分算 loss，指令/系统提示/对话标记全部置为 -100。
        这样模型不会浪费容量去「背用户问了什么」，而是专注学「用户这么问时该怎么答」。
        这也是为什么 SFT 几百条样本就能让模型学会对话格式，而纯 LM 得喂海量文本。

    【一个样本的构造细节】
        假设 instruction="你好"，output="你好呀"，无 system：
        完整序列 S = [begin, user, 你, 好, end, assistant, 你, 好, 呀, end]
        x = S[:-1]  = [begin, user, 你, 好, end, assistant, 你, 好, 呀]
        y = S[1:]   = [user, 你, 好, end, assistant, 你, 好, 呀, end]
        labels =    [-100, -100, -100, -100, -100,  你,  好,  呀,  end]
                                          ↑ assistant 在 y 中的位置，从这里开始算 loss
        即：模型看到 <|assistant|> 后，要能依次预测出「你 好 呀 <|end|>」。

    【为什么 assistant 标记本身不算 loss】
        推理时我们手动把 <|assistant|> 拼在指令后面喂给模型，它不需要「自己决定何时输出
        assistant 标记」。所以训练时也不让它为这个标记付 loss，只监督真正的回答内容 + <|end|>。
        这和 HuggingFace SFTTrainer 里 response_template 的处理一致。
    """

    def __init__(self, pairs, char2idx, seq_len):
        self.seq_len = seq_len
        self.pad_id = char2idx[SPECIAL_TOKENS["pad"]]
        self.begin_id = char2idx[SPECIAL_TOKENS["begin"]]
        self.system_id = char2idx[SPECIAL_TOKENS["system"]]
        self.user_id = char2idx[SPECIAL_TOKENS["user"]]
        self.assistant_id = char2idx[SPECIAL_TOKENS["assistant"]]
        self.end_id = char2idx[SPECIAL_TOKENS["end"]]
        # 预编码所有样本：SFT 样本数通常不多（几百到几万条），一次性编码完省得 __getitem__ 里反复做
        self.samples = []
        for system, instruction, output in pairs:
            self.samples.append(self._encode(system, instruction, output, char2idx))

    def _encode(self, system, instruction, output, char2idx):
        """把一条 (system, instruction, output) 编成 (x, labels)，长度均为 seq_len。"""
        # 按对话格式拼完整 id 序列
        seq = [self.begin_id]
        if system:
            seq += [self.system_id]
            seq += [char2idx[c] for c in system if c in char2idx]
            seq += [self.end_id]
        seq += [self.user_id]
        seq += [char2idx[c] for c in instruction if c in char2idx]
        seq += [self.end_id]
        # 记录 assistant 标记在 seq 中的位置：它之后的第一个 token 就是回答的开头
        assistant_pos = len(seq)
        seq += [self.assistant_id]
        seq += [char2idx[c] for c in output if c in char2idx]
        seq += [self.end_id]

        # 截断：超过 seq_len+1 就从右边砍（保留指令，回答被截断）。
        # seq_len+1 是因为 x 取前 seq_len 个、y 取后 seq_len 个（错开一位），需要 seq_len+1 个原始 token。
        if len(seq) > self.seq_len + 1:
            seq = seq[: self.seq_len + 1]

        # 补齐到 seq_len+1（用 pad）
        pad_len = (self.seq_len + 1) - len(seq)
        seq += [self.pad_id] * pad_len

        # x = 前 seq_len 个，y = 后 seq_len 个（错开一位，这和 LM 模式一样）
        x = seq[: self.seq_len]
        y = seq[1 : self.seq_len + 1]

        # 构造 labels：默认全 -100，只把「回答部分」设成真实 id
        labels = [-100] * self.seq_len
        # assistant_pos 是 assistant 标记在 seq 中的下标。
        # y = seq[1:]，所以 seq[assistant_pos]（assistant 标记）在 y 中的下标是 assistant_pos - 1。
        # 回答的第一个 token 是 seq[assistant_pos + 1]，对应 y 中的下标 assistant_pos。
        # 所以 labels 从下标 assistant_pos 开始填真实值。
        for i in range(assistant_pos, self.seq_len):
            if y[i] != self.pad_id:   # pad 位置保持 -100，不参与 loss
                labels[i] = y[i]

        return (torch.tensor(x, dtype=torch.long),
                torch.tensor(labels, dtype=torch.long))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ─────────────────────────── 模型 ───────────────────────────

class LM(nn.Module):
    def __init__(self,vocab_size,  d_model=192, n_head=4, n_layer=4, max_len=512,
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
            enable_nested_tensor=False,  # 带 mask 时关掉嵌套张量优化，行为更直观
        )
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok.weight   # 输入/输出 embedding 共享
        nn.init.normal_(self.tok.weight, std=0.02)
        if self.pos is not None:
            nn.init.normal_(self.pos.weight, std=0.02)

    def forward(self, idx):
        x = self.tok(idx)  # (B, T, d) 查表
        if self.use_pos:
            x = x + self.pos(torch.arange(idx.size(1), device=idx.device))
        x = self.drop(x)
        mask = None
        if self.causal:
            mask = nn.Transformer.generate_square_subsequent_mask(
                idx.size(1), device=idx.device)   # 下三角 0、右上 -inf
        x = self.encoder(x, mask=mask)
        return self.head(x)


# ─────────────────────────── 训练 / 评估 ───────────────────────────

@torch.no_grad()
def evaluate(model, loader, device, last_only=False):
    """
    评估：返回 (loss, ppl)。
    兼容 LM 和 SFT 两种模式——loader 吐出的第二个元素在 LM 里是 y（真实下一字），
    在 SFT 里是 labels（回答部分为真实 id、其余为 -100）。F.cross_entropy 默认
    ignore_index=-100，所以同一份代码两种数据都能跑，只需在统计 token 数时跳过 -100。
    """
    model.eval()
    total, tokens = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        if last_only:
            logits, y = logits[:, -1, :], y[:, -1]
        # reduction="sum"：先把所有有效位置的 loss 加起来，再除以有效 token 数。
        # ignore_index=-100 是 PyTorch 的默认值，SFT 里被掩码的位置会自动跳过。
        total += float(F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                       y.reshape(-1), reduction="sum", ignore_index=-100))
        # 只统计「没被掩码」的 token 数，否则 SFT 模式下 PPL 的分母会掺进去大量 -100，
        # 算出来的 loss 会被稀释、PPL 虚低。
        tokens += int((y != -100).sum().item())
    if tokens == 0:
        return float("inf"), float("inf")
    loss = total / tokens
    return loss, math.exp(loss)


def train(args):
    """
    训练入口：建数据 → 建模型 → 逐 epoch 前向/反向 → 每轮在验证集上报 PPL → 存最优。

    支持两种模式（由 --mode 控制）：
      lm  —— 原语言模型：纯文本语料 + 滑动窗口，整段算 loss。
      sft —— 监督微调：JSONL 指令-回答对 + 对话格式，只在回答部分算 loss。

    两种模式共用同一个 TransformerLM 模型和同一个训练循环，差别只在「数据怎么切」
    和「loss 在哪里算」——这正好说明 SFT 不是换了个模型，而是换了种「教」的方式。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    is_sft = (args.mode == "sft")

    # ── 数据 + 字表 ──────────────────────────────────────────────
    if is_sft:
        # SFT：从 JSONL 读 (system, instruction, output) 三元组
        pairs = load_sft_data(args.sft_data)
        if args.max_chars:
            # 截断样本数（不是字符数），快速跑通用
            pairs = pairs[:args.max_chars]
        # 用字表构建：把所有出现过的字（指令+回答+系统提示）合并起来统计
        all_text = "".join(s + i + o for s, i, o in pairs)
        char2idx, idx2char = build_vocab(all_text, args.max_vocab, add_special=True)
        V = len(char2idx)
        print(f"device: {device}  SFT 样本 {len(pairs):,} 条  字表 {V}（含 {len(SPECIAL_TOKEN_LIST)} 个特殊 token）")

        # 按「样本」打乱再切分（不是按字符），保证验证集的问答对训练时没见过
        random.shuffle(pairs)
        split = int(len(pairs) * (1 - args.val_ratio))
        tr = SFTDataset(pairs[:split], char2idx, args.seq_len)
        va = SFTDataset(pairs[split:], char2idx, args.seq_len)
    else:
        # LM：原逻辑
        text = load_corpus(args.corpus)
        if args.max_chars and len(text) > args.max_chars:
            text = text[:args.max_chars]
        char2idx, idx2char = build_vocab(text, args.max_vocab, add_special=False)
        V = len(char2idx)
        print(f"device: {device}  语料 {len(text):,} 字  字表 {V}")
        lines = text.splitlines()
        random.shuffle(lines)
        split = int(len(lines) * (1 - args.val_ratio))
        tr = CharDataset("\n".join(lines[:split]), char2idx, args.seq_len, args.stride)
        va = CharDataset("\n".join(lines[split:]), char2idx, args.seq_len, args.stride)

    train_loader = DataLoader(tr, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(va, batch_size=args.batch_size, shuffle=False, drop_last=True)

    # ── 模型 ─────────────────────────────────────────────────────
    model = TransformerLM(V, args.d_model, args.heads, args.layers, args.seq_len,
                          args.dropout, use_pos=not args.no_pos, causal=not args.no_mask).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    mode_str = "SFT（监督微调，仅回答部分算 loss）" if is_sft else "LM（语言模型，全位置算 loss）"
    print(f"模式: {mode_str}")
    print(f"模型: TransformerEncoderLayer × {args.layers}（d={args.d_model} h={args.heads} "
          f"FFN={4*args.d_model} Pre-LN）  位置编码={'有' if not args.no_pos else '无'}  "
          f"因果掩码={'有' if not args.no_mask else '**无（泄漏实验）**'}  参数量 {n_par/1e6:.2f}M")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    best = float("inf")
    print(f"\n{'Epoch':>6} {'TrainLoss':>10} {'ValPPL(回答)':>14} {'ValPPL(末位)':>13} {'tok/s':>9}")
    print("-" * 60)
    for ep in range(1, args.epochs + 1):
        model.train()
        st, seen, tot = time.time(), 0, 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            # ★ SFT 的核心就这一处变化：cross_entropy 加 ignore_index=-100。
            #   LM 模式下 y 里没有 -100，这一行等价于原来的写法；
            #   SFT 模式下 y 里指令部分全是 -100，loss 自动只在回答上算，
            #   模型因此只学「怎么答」，不学「怎么问」。
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1), ignore_index=-100)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            # 统计有效 token（SFT 里要排除 -100），这样 TrainLoss 才是「每个回答 token 的平均 loss」
            n_valid = int((y != -100).sum().item())
            tot += float(loss) * n_valid
            seen += n_valid
        dt = max(time.time() - st, 1e-6)   # 防止 epoch 太快导致 dt=0 进而 seen/dt 除零
        # 验证集 PPL：SFT 下「全位置」其实就是「回答部分」（因为指令被掩码了），
        # 所以列名改成 ValPPL(回答) 更准确；末位 PPL 依然是「最后一个回答 token」的预测难度。
        _, v_ppl = evaluate(model, val_loader, device, last_only=False)
        _, v_ppl_last = evaluate(model, val_loader, device, last_only=True)
        mark = ""
        # 保存条件：验证 PPL 创新低，或者 best 还是 inf（第一轮 / 验证集太小没拿到有效 PPL）。
        # 加第二个条件是为了避免「验证集太小导致 PPL 一直是 inf，结果一个 ckpt 都没存」的情况。
        if v_ppl < best or math.isinf(best):
            best = v_ppl
            mark = " *"
            # ckpt 里存 mode，generate 时据此决定用哪种格式拼输入
            torch.save({"model_state": model.state_dict(), "char2idx": char2idx,
                        "idx2char": idx2char, "args": vars(args)}, args.save)
        print(f"{ep:>6} {tot/seen if seen else 0:>10.4f} {v_ppl:>14.2f} {v_ppl_last:>13.2f} {seen/dt:>9,.0f}{mark}")

    print(f"\n最佳验证 PPL: {best:.2f}  已保存 {args.save}")


@torch.no_grad()
def generate(args):
    """
    生成：根据 ckpt 的 mode 走两种路径
      lm  —— 给一段开头，续写文本（原逻辑）。
      sft —— 把输入当作用户指令，按对话格式拼好后生成助手回答，遇到 <|end|> 停止。

    为什么推理必须「逐 token」而不能像训练那样一次算完全部位置：
        训练时每个位置 t 的标签 x_{t+1} 已经在输入序列里了，所以能一次并行算 B×T 个 loss；
        生成时第 t+1 个字还不存在 —— 必须先采样出它，才能拿它当前文去预测第 t+2 个。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location=device)
    c = ck["args"]
    char2idx, idx2char = ck["char2idx"], ck["idx2char"]
    is_sft = c.get("mode", "lm") == "sft"

    model = TransformerLM(len(char2idx), c["d_model"], c["heads"], c["layers"], c["seq_len"], 0.0,
                          use_pos=not c.get("no_pos", False),
                          causal=not c.get("no_mask", False)).to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()

    if is_sft:
        # SFT：把用户指令按训练时的格式拼好
        #   <|begin|><|user|>{指令}<|end|><|assistant|>
        # 然后从 <|assistant|> 之后开始生成，遇到 <|end|> 就停。
        begin_id = char2idx[SPECIAL_TOKENS["begin"]]
        user_id = char2idx[SPECIAL_TOKENS["user"]]
        end_id = char2idx[SPECIAL_TOKENS["end"]]
        assistant_id = char2idx[SPECIAL_TOKENS["assistant"]]
        ids = [begin_id, user_id]
        ids += [char2idx[ch] for ch in args.generate if ch in char2idx]
        ids += [end_id, assistant_id]
        prompt_len = len(ids)        # 记下前缀长度，输出时只打新生成的部分
        out = list(ids)
        for _ in range(args.n_chars):
            logits = model(torch.tensor([out[-c["seq_len"]:]], device=device))[0, -1]
            if args.temperature <= 0.01:
                nxt = int(logits.argmax())
            else:
                nxt = int(torch.multinomial(F.softmax(logits / args.temperature, dim=-1), 1))
            if nxt == end_id:        # 遇到结束标记就停，回答到此为止
                break
            out.append(nxt)
        print(f"用户: {args.generate}")
        print(f"助手: {''.join(idx2char[i] for i in out[prompt_len:])}")
    else:
        # LM：原续写逻辑
        ids = [char2idx[ch] for ch in args.generate if ch in char2idx]
        out = list(ids)
        for _ in range(args.n_chars):
            logits = model(torch.tensor([out[-c["seq_len"]:]], device=device))[0, -1]
            if args.temperature <= 0.01:
                nxt = int(logits.argmax())
            else:
                nxt = int(torch.multinomial(F.softmax(logits / args.temperature, dim=-1), 1))
            out.append(nxt)
        print(f"输入: {args.generate}")
        print(f"续写: {''.join(idx2char[i] for i in out[len(ids):])}")


def main():
    ap = argparse.ArgumentParser()
    # ── 模式开关 ────────────────────────────────────────────────
    # sft：监督微调，用 JSONL 指令-回答对训练，只在回答部分算 loss
    # lm ：原语言模型，用纯文本语料训练，全位置算 loss
    ap.add_argument("--mode", choices=["sft", "lm"], default="sft",
                    help="训练模式：sft=监督微调（默认），lm=语言模型")
    default_corpus = os.path.join(HERE, "corpus_general.txt")
    ap.add_argument("--corpus", default=default_corpus, help="LM 模式的语料路径（支持通配符）")
    default_sft = os.path.join(HERE, "sft_sample.jsonl")
    ap.add_argument("--sft_data", default=default_sft, help="SFT 模式的 JSONL 数据路径")
    ap.add_argument("--max_chars", type=int, default=0,
                    help="截断：LM 模式截前 N 字符；SFT 模式截前 N 条样本")
    ap.add_argument("--max_vocab", type=int, default=4000)
    ap.add_argument("--seq_len", type=int, default=64)
    # stride 默认等于 seq_len（不重叠）。调小会增加样本量但让相邻窗口高度相关；
    # 和 LSTM 基线对比时必须保持 64，否则切分口径不一致，PPL 不可比
    ap.add_argument("--stride", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-3)
    # 默认这套超参（d=192 / 4 层 / 4 头）参数量约 2.66M，正好和本周 LSTM 基线的 2.46M 对齐在 ±10% 内 ——
    # 作业要求「先对齐参数量再比 PPL」，否则比出来的不是结构差异而是模型大小的差异
    ap.add_argument("--d_model", type=int, default=192)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--val_ratio", type=float, default=0.05)
    # seed 决定「按行打乱」的结果，也就是训练/验证的划分本身。
    # 复算 PPL（demo_continuation.py）必须用同一个 seed 才能还原同一份验证集
    ap.add_argument("--seed", type=int, default=42)
    # ⚠ --no-mask 不是「更快的写法」，而是泄漏实验：模型能看到未来 → 学会抄输入。
    # 打开后请务必同时看「末位 PPL」和续写效果，否则会被 1.07 的全位置 PPL 骗过去
    ap.add_argument("--no-mask", dest="no_mask", action="store_true",
                    help="去掉因果掩码（泄漏实验）")
    # --no-pos 是位置编码消融：自注意力本身对位置是「置换等变」的（打乱输入顺序，输出只会跟着打乱），
    # 没有位置编码它就退化成一个「词袋 + 上下文加权」，连「上一个字是什么」这种最强特征都拿不到。
    # 实测 PPL 从 24.95 掉到 27.42（seq_len 拉长到 256 时差距会更明显）
    ap.add_argument("--no-pos", dest="no_pos", action="store_true", help="去掉位置编码")
    ap.add_argument("--save", default="tf_best.pt")
    ap.add_argument("--generate", default=None, help="给一段开头/指令，生成回答")
    ap.add_argument("--ckpt", default="tf_best.pt")
    ap.add_argument("--n_chars", type=int, default=40)
    ap.add_argument("--temperature", type=float, default=0.8)
    args = ap.parse_args()

    # 一条命令两用：带 --generate 就只做生成（不再训练），否则进入训练。
    if args.generate:
        generate(args)
    else:
        train(args)


if __name__ == "__main__":
    main()

