"""
transformer_lm.py — 字符级 Transformer 语言模型（自回归），第 4 周作业配套实验代码

【这个脚本干什么】
    把本周课上用 RNN/LSTM 训的「字符级语言模型」，换成 Transformer（GPT 那一套结构）重训一遍，
    在**同一份语料、同一套切分、同一个验证集**上和 LSTM 比 PPL；再用两个开关做消融：
        --no-mask   去掉因果掩码（看"泄漏"是怎么发生的）
        --no-pos    去掉位置编码（看注意力失去位置信息后有多差）

【Transformer 层用 torch 自带封装】
    nn.TransformerEncoderLayer(norm_first=True, batch_first=True, activation="gelu")
    nn.TransformerEncoder(...)
    因果掩码用 nn.Transformer.generate_square_subsequent_mask(T)：
        下三角（含对角线）为 0 = 可以看，右上角为 -inf = 不能看，
        也就是「位置 t 只能看 0..t」，这正是 GPT 的自回归约束。

【三个必答的问题（作业要回答的）】
    1. 因果掩码去掉后，训练 loss 为什么会掉到接近 0？为什么"末位 PPL"没有跟着掉？
    2. 位置编码去掉后，PPL 变差多少？为什么注意力本身提供不了位置信息？
    3. 参数量对齐（~2.5M）的情况下，Transformer 和 LSTM 谁赢？

【为什么可以不和课件耦合】
    数据准备、字表、切分、评测口径都写在本文件里，语料用同目录的 corpus_sample.txt，
    也可以 --corpus 指向课件里那份完整的 corpus_general.txt。

用法:
    python transformer_lm.py --epochs 6                     # 基线：因果掩码 + 位置编码
    python transformer_lm.py --epochs 6 --no-mask            # 泄漏实验
    python transformer_lm.py --epochs 6 --no-pos             # 去掉位置编码
    python transformer_lm.py --generate "今天天气" --ckpt tf_best.pt
    python transformer_lm.py --corpus "D:/.../corpus_general.txt" --epochs 6   # 用完整语料
"""

# ═══════════════════ 阅读指引（先看这 6 条，再进代码） ═══════════════════
# 这份文件是「参考实现」，目的不是教你从零写 Transformer，而是让你自己在代码里
# 找到下面这几个问题的答案 —— 所以注释重点都标在「为什么」上：
#
#   1) 为什么必须是「因果」掩码？           → 见 TransformerLM.forward 里 generate_square_subsequent_mask 那段
#   2) 为什么训练时能一口气算完所有位置？   → 见 train 里那行 F.cross_entropy(...reshape(-1, V))
#      而生成时必须一个字一个字来？         → 见 generate 里的 for 循环
#   3) 为什么 --no-mask 的「全位置 PPL」好看得离谱？ → 见 evaluate 的 last_only 参数
#   4) 为什么位置编码是「加」而不是「拼」？ → 见 forward 里 x = x + self.pos(...)
#   5) Pre-LN / weight tying / 4 倍 FFN 是什么、为什么要这么设？ → 见 TransformerLM.__init__
#   6) 为什么「PPL 低」不代表「会说话」？    → 跑一遍 --no-mask，看末位 PPL 和续写就懂了
#
# 本机实测（RTX 4060，同一份语料、同一套切分，详见同目录 README.md / 演示说明.md）：
#   · 2.66M 参数（d=192 / 4 层 / 4 头）：
#       正常        全位置 PPL 24.95 ｜ 末位 22.49
#       去掉 mask   全位置 PPL 1.07（假象）｜ 末位 98.23（真实水平）
#       去掉 pos    全位置 PPL 27.42
#       LSTM 对照（2.46M）               全位置 PPL 38.32
#   · 12.21M 参数（d=384 / 6 层 / 8 头，课堂演示用的 *_big.pt）：
#       有 mask     全位置 15.06 ｜ 末位 13.19
#       无 mask     全位置 1.07  ｜ 末位 88.50，续写退化成「们们们们…」复读 40 个字
#   注意：这些数字是「同一口径下的相对比较」，换语料/换 epoch 数就不能直接比。

import argparse
import glob          # 语料路径支持通配符（比如 "corpus_*.txt" 一次读多份）
import math          # 只用来把交叉熵换成 PPL：ppl = exp(loss)
import os
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F   # 只用它的 cross_entropy / softmax，比 nn.CrossEntropyLoss 更好嵌进函数式写法
from torch.utils.data import DataLoader, Dataset   # 切好的样本是 Dataset，凑 batch / 打乱交给 DataLoader

# 本文件所在目录。后面拼默认语料、默认 ckpt 路径都用它，
# 这样无论你从哪个目录执行 python，都能找到同目录的 corpus_sample.txt（否则会 FileNotFoundError）
HERE = os.path.dirname(os.path.abspath(__file__))


# ─────────────────────── 数据（与本周 LSTM 版同口径） ───────────────────────

def load_corpus(pattern):
    """
    读语料：pattern 可以是单个文件，也可以是通配符（内部用 glob 展开，按文件名排序后拼接）。

    两处容错是故意留的：
        errors="ignore"  —— 语料里偶尔混进坏字节时不要让整个训练崩掉，直接跳过解不出的字符；
        sorted(...)      —— 保证同一台机器上多次运行读到的是同一个顺序，实验才可复现
                            （glob 本身不保证顺序，换个文件系统顺序可能不一样）。
    """
    text = ""
    for p in sorted(glob.glob(pattern)):
        with open(p, encoding="utf-8", errors="ignore") as f:
            text += f.read()
    if not text:
        # 空语料会让后面 build_vocab 得到空字表、Embedding(0) 直接报错，
        # 所以这里提前给出「是不是路径写错了」的明确提示，比让 torch 抛一堆形状错误好排查
        raise FileNotFoundError(f"没读到语料: {pattern}")
    return text


def build_vocab(text, max_vocab):
    """
    建字表：字符 → id 的双向映射（字符级模型没有 tokenizer，字表就是全部词表）。

    max_vocab 的取舍：
        max_vocab > 0  —— 只保留出现频率最高的 max_vocab 个字（most_common 返回的是「按频率降序」）。
                          中文语料里长尾字符极多，砍掉后 softmax 的类别数变小，模型更好训、也更快；
                          代价是生僻字全变成「查不到」→ 在 CharDataset 里被直接丢掉。
        max_vocab = 0  —— 用 sorted(set(text)) 取全部字符，顺序稳定（按 unicode 码位），但字表可能上万。
    注意 id 的顺序在这里不影响语义（模型学的是 embedding，不是 id 的大小），但影响可复现性。
    """
    from collections import Counter
    chars = [c for c, _ in Counter(text).most_common(max_vocab)] if max_vocab else sorted(set(text))
    char2idx = {c: i for i, c in enumerate(chars)}
    return char2idx, {i: c for c, i in char2idx.items()}


class CharDataset(Dataset):
    """
    x = 连续 seq_len 个字符，y = 往后错一位；stride 控制切块间隔（=seq_len 即不重叠）

    为什么 y 要「错一位」：自回归语言模型在位置 t 的目标永远是「下一个字」，
    所以喂进去 [今, 天, 天]，监督信号就是 [天, 天, 气]。
    一个长度为 T 的窗口因此能同时贡献 T 个训练样本（每个位置都是一个预测题），
    这也是 Transformer 训练效率高的根本原因 —— T 个位置是并行算的，不靠时间步串行。

    stride（步长）的取舍：
        stride = seq_len  —— 相邻窗口不重叠，样本少但彼此独立；
        stride < seq_len  —— 窗口互相重叠，样本变多（同一次前向能覆盖更多上下文边界），
                             但相邻样本高度相关，等于变相放大了这几段文本的权重。
    本作业统一用 stride=64=seq_len，和 LSTM 版保持同一口径，PPL 才能横向比。
    """

    def __init__(self, text, char2idx, seq_len, stride):
        # 字表外的字符（被 max_vocab 砍掉的生僻字）直接丢弃、不做 UNK：
        # 这样模型从来没见过「未知」这个符号，也就不会把它学成一个高频垃圾类别；
        # 副作用是文本会「被缝合」—— 丢掉几个字后前后文接在一起，这是可以接受的近似。
        ids = [char2idx[c] for c in text if c in char2idx]
        self.data = torch.tensor(ids, dtype=torch.long)
        self.seq_len = seq_len
        self.stride = max(1, stride)      # 防呆：stride=0 会让 __getitem__ 永远取到同一段

    def __len__(self):
        # 要拿到 seq_len 个输入 + 1 个「下一位」目标，所以长度里多减 1；
        # 用 (可用长度)//stride + 1 数窗口数，和 __getitem__ 的起点 i*stride 严格对应
        return max(0, (len(self.data) - self.seq_len - 1) // self.stride + 1)

    def __getitem__(self, i):
        s = i * self.stride
        # 注意两个切片都取 seq_len 个：x 从 s 开始，y 从 s+1 开始，天然错一位
        return self.data[s:s + self.seq_len], self.data[s + 1:s + self.seq_len + 1]


# ─────────────────────── 模型（torch 自带封装） ───────────────────────

class TransformerLM(nn.Module):
    """
    可学习位置编码 + N 层 TransformerEncoderLayer + 输出头（与输入 embedding 共享权重）。

    causal=True 时给编码器传下三角掩码，等价 GPT 的因果注意力：
        mask[i][j] = 0     (j <= i，可以看)
        mask[i][j] = -inf  (j >  i，不能看)
    causal=False 就是 BERT 那种双向注意力 —— 训练 loss 会崩到接近 0，但那是"抄输入"。

    结构对照（读代码时心里有个 GPT 的图）：
        nn.Embedding(V, d)                      ← 字符查表，等价 GPT 的 token embedding
        + nn.Embedding(max_len, d)（可学习位置）  ← 等价 GPT 的 position embedding
        nn.TransformerEncoder(layer × N)        ← 注意！它名字叫 Encoder，但加了因果掩码之后
                                                  每一层就是「Masked Self-Attn + FFN」，
                                                  也就等于 GPT 的一个 decoder block
        nn.Linear(d, V, bias=False)             ← 输出头，把 d 维向量映射回词表大小的 logits
    和 GPT 的唯一结构差别：这里没有 cross-attention（没有 encoder 端输入），
    因为纯语言模型只需要「看着前文预测下一个字」。
    """

    def __init__(self, vocab_size, d_model=192, n_head=4, n_layer=4, max_len=512,
                 dropout=0.1, use_pos=True, causal=True):
        super().__init__()
        # 这两个开关就是消融实验的全部：causal 对应 --no-mask，use_pos 对应 --no-pos。
        # 存成属性（而不是在 forward 里判断参数）是为了让同一份类既能跑两次前向对比，
        # 也方便 generate 时从 ckpt 的 args 里原样还原出「当初是怎么训的」
        self.causal = causal
        self.use_pos = use_pos
        self.tok = nn.Embedding(vocab_size, d_model)      # 字符 id → d 维向量，训练中是会学的
        # 位置编码用「可学习」而不是原论文那种固定的 sin/cos：
        # 参数多不了多少（max_len × d），但对小模型、短序列通常更好训；
        # --no-pos 时置 None，此时 forward 里连这一步都不做，才是干净的消融
        self.pos = nn.Embedding(max_len, d_model) if use_pos else None
        self.drop = nn.Dropout(dropout)                  # 在 embedding 之后再加一层 dropout，是 GPT 的常规做法

        # ── 一个 Transformer block ──
        # 这一行展开就是学生作业里的第 1 题：
        #   Self-Attention（q/k/v 投影 + 多头拼接 + 输出投影）
        #   → 残差 + LayerNorm → FFN（Linear(d,4d) + GELU + Linear(4d,d)）→ 残差 + LayerNorm
        # dim_feedforward=4*d_model 里的 4 倍不是玄学：原论文就这么设，而且 FFN 是「逐位置」的，
        # 它是每层参数的大头（约占一层的 2/3），这个比例决定了模型容量花在「混合信息(注意力)」
        # 还是「变换表示(FFN)」上
        # activation="gelu"：比 ReLU 平滑、负半轴不硬截断，BERT/GPT-2 起的标配
        # batch_first=True：张量形状是 (batch, seq, d)，和 PyTorch 的常规习惯一致；
        #                   不开的话是 (seq, batch, d)，后面 logits[:, -1, :] 就取错维度了
        # norm_first=True → Pre-LN：LayerNorm 放在子层「输入」之前，残差主干是一条干净的
        #                   恒等通路，梯度能从最后一层直通到第一层。原论文是 Post-LN（norm 在
        #                   残差相加之后），深层时必须配 warmup 才训得动；Pre-LN 对学习率宽容得多，
        #                   所以现在 99% 的开源 LLM 都是 Pre-LN
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head, dim_feedforward=4 * d_model,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        # nn.TransformerEncoder = 把同一个 layer 堆 N 次 + 最后再来一个 LayerNorm。
        # 注意 PyTorch 会 deepcopy 传进去的 layer，所以不是「共享一层参数」
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=n_layer, norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False,      # 带 mask 时关掉嵌套张量优化，行为更直观
        )
        # 输出头不加 bias：字符级词表很大时 bias 几乎没用，去掉能省参数；
        # bias=False 也是「和输入 embedding 共享权重」的前提（形状必须严格一致）
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok.weight   # 输入/输出 embedding 共享
        # ↑ weight tying（权重共享）：
        #   1) 省参数：这一份 (V × d) 就有 4000×192 ≈ 0.77M，占整个 2.66M 模型的近 1/3；
        #   2) 提效果：输入侧学的是「这个字长什么样」，输出侧要判断的也是「下一个字是哪个」，
        #      两者本来就在同一个语义空间里，绑在一起等于给输出头加了很强的共享先验，
        #      PPL 通常会稳定好一点。GPT-2 / GPT-3 都这么干。
        #   3) 代价：两边的梯度会叠加到同一份权重上，所以初始化必须小心（见下两行）。

        # embedding 初始化成小方差（GPT 的常规做法）：nn.Embedding 默认 std=1，
        # 和输出头共享权重时 logits 会非常大，初始 loss 直接飙到 19（≈ln4000 的两倍多）
        # 这里 std=0.02 正是 GPT-2 的初始化；位置编码同理（两者相加后量级要匹配）
        nn.init.normal_(self.tok.weight, std=0.02)
        if self.pos is not None:
            nn.init.normal_(self.pos.weight, std=0.02)

    def forward(self, idx):
        # idx: (batch, T) 的字符 id；输出: (batch, T, vocab_size) 的未归一化 logits
        x = self.tok(idx)                     # (B, T, d) 查表
        if self.use_pos:
            # 位置编码「相加」而不是「拼接」，原因有三：
            #   1) 形状不用变，后续所有子层（注意力、FFN）都不用改；
            #   2) 残差连接要求输入输出同维；
            #   3) 相加等价于把「我是第几个字」写进同一个 d 维空间，模型靠不同的子空间维度
            #      同时表示「是什么字」和「在第几位」，注意力里 q·k 就能同时比较内容和相对位置。
            # arange(T) 是当前窗口的绝对位置 0..T-1（每段都从头开始编号，所以位置的含义是
            # 「段内第几个字」，这与训练时每段都从 0 开始一致 —— 生成时也必须这样对齐）。
            x = x + self.pos(torch.arange(idx.size(1), device=idx.device))
        x = self.drop(x)
        mask = None
        if self.causal:
            # 这就是整个「自回归」的开关。generate_square_subsequent_mask(T) 生成一个 T×T 矩阵：
            #     [[ 0, -inf, -inf, ...],
            #      [ 0,   0, -inf, ...],
            #      ...                     ]
            # 注意力算的是 softmax(QK^T/√d)·V，把某个位置加上 -inf 后，exp(-inf)=0，
            # 于是那个位置的权重被彻底清零 —— 位置 t 的注意力只能落在 0..t 上，看不到未来。
            # 没有它，位置 t 的输出就能直接看到 x_t 本身（也就是它要预测的那个标签的一部分），
            # 模型会立刻学会「把当前输入抄到输出」，loss 掉到 0.07 —— 详见 evaluate 的 last_only 说明。
            # 另外注意：这也是 BERT（双向、无掩码）不能直接拿来做生成的根本原因。
            mask = nn.Transformer.generate_square_subsequent_mask(
                idx.size(1), device=idx.device)   # 下三角 0、右上 -inf
        x = self.encoder(x, mask=mask)
        # 输出层直接给 logits，不在模型里做 softmax：
        #   F.cross_entropy 内部自带 log_softmax，归一化两次会让数值变差、梯度变小（常见坑）。
        # 只有 generate 采样时才显式 softmax，因为那里确实要概率分布。
        return self.head(x)


# ─────────────────────── 训练 / 评估 ───────────────────────

@torch.no_grad()
def evaluate(model, loader, device, last_only=False):
    """返回 (loss, ppl)；last_only=True 时只看每段最后一个位置（这个位置抄不到答案）"""
    # model.eval() 和 @torch.no_grad() 是两件事，缺一不可：
    #   model.eval()  —— 把 Dropout 关掉（推理时随即丢弃会让同一个输入的输出每次都变，
    #                    PPL 就没意义了）、把 BatchNorm 换成用运行统计量（本模型没有 BN，但习惯要养成）；
    #   no_grad()     —— 不建计算图，显存/速度都省不少，评测几十个 batch 时差别很明显。
    # 如果忘了 eval()，dropout 会让 PPL 系统性偏高，而且每次跑出来的数都不一样 —— 很多人对不上数就是这个原因。
    model.eval()
    total, tokens = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        if last_only:
            # 只取每段「最后一个位置」的预测：
            # 训练窗口是 [s, s+T)，最后一个位置的输入是 x[s+T-1]，目标是 x[s+T]，而这一个目标字
            # 从来没出现在输入窗口里 —— 所以这个位置永远「抄不到答案」。
            # 这就是识破 --no-mask 泄漏实验的关键指标：无 mask 时全位置 PPL 能到 1.07，
            # 但末位 PPL 高达 98.23（12M 版是 88.50），后者才反映真实语言能力。
            logits, y = logits[:, -1, :], y[:, -1]
        # 用 reduction="sum" 再自己除以 token 数，而不是让 cross_entropy 直接取 mean：
        #   mean 是「每个 batch 内先平均」，最后一个不满的 batch 会和满 batch 权重相同，
        #   算出来的值会和「按 token 加权」有细微差异。PPL 是要写进作业表里横向比的，口径必须干净。
        total += float(F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                       y.reshape(-1), reduction="sum"))
        tokens += y.numel()
    loss = total / tokens
    # 交叉熵是以 e 为底的平均负对数似然，exp 回来就是 PPL（困惑度）：
    # 「模型在每一步平均在多少个候选字之间犹豫」。字表 4000 时，瞎猜的 PPL ≈ 4000，越好越接近 1。
    return loss, math.exp(loss)


def train(args):
    """
    训练入口：建数据 → 建模型 → 逐 epoch 前向/反向 → 每轮在验证集上按两种口径报 PPL → 存最优。

    这里没有写 LR schedule、没有早停、没有梯度累积，全部刻意保持最小实现 ——
    因为作业的对比是「同一套训练流程只换模型结构」，流程越简单，差异越能归因到结构上。
    """
    # 有 CUDA 就用 CUDA。字符级小模型在 CPU 上也能跑完，只是慢（默认参数下要更久）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 固定随机种子：语料按行打乱、参数初始化、DataLoader 的 shuffle 都受它影响。
    # 论文里「同一划分、同一 seed」的对比才有意义，作业评分也看这一点
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    text = load_corpus(args.corpus)
    if args.max_chars and len(text) > args.max_chars:
        # 只在没 GPU 时用来快速跑通流程的近似手段；注意截断后 PPL 不能和全量语料的数字比
        text = text[:args.max_chars]
    char2idx, idx2char = build_vocab(text, args.max_vocab)
    V = len(char2idx)      # 词表大小：Embedding 行数、输出头列数、softmax 类别数，三者必须一致
    print(f"device: {device}  语料 {len(text):,} 字  字表 {V}")

    # 按「行」打乱再切分，而不是按字符位置硬切：
    #   语料是很多条独立句子/段落拼起来的，按字符硬切很容易让同一段文字同时出现在
    #   训练段和验证段里（相邻窗口重叠 / 同一篇文章跨界），验证集就泄漏了，PPL 会虚低。
    #   先打乱行、再各取一部分，能保证「验证集里的句子训练时没见过」。
    # 这也是为什么 demo_continuation.py 复算 PPL 时必须用 ckpt 里记下的 seed 重放同样的打乱。
    lines = text.splitlines()
    random.shuffle(lines)
    split = int(len(lines) * (1 - args.val_ratio))     # 默认 5% 留作验证
    tr = CharDataset("\n".join(lines[:split]), char2idx, args.seq_len, args.stride)
    va = CharDataset("\n".join(lines[split:]), char2idx, args.seq_len, args.stride)
    # 训练集 shuffle=True：每个 epoch 的 batch 组成不同，等价于做了一点数据层面的正则；
    # drop_last=True：丢掉最后一个不满的 batch，避免 batch 很小时（比如只剩 1 条）梯度过噪，
    #                 也让每个 batch 的 shape 固定
    train_loader = DataLoader(tr, batch_size=args.batch_size, shuffle=True, drop_last=True)
    # 验证集 shuffle=False：不需要打乱（评测不看顺序），关掉能省一点开销、也让结果可复现
    val_loader = DataLoader(va, batch_size=args.batch_size, shuffle=False, drop_last=True)

    # 两个开关在这里汇合：--no-pos → use_pos=False，--no-mask → causal=False。
    # 其余超参一律来自命令行，保证三次消融实验「除开关外完全一致」——这是消融实验的基本要求
    model = TransformerLM(V, args.d_model, args.heads, args.layers, args.seq_len,
                          args.dropout, use_pos=not args.no_pos, causal=not args.no_mask).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"模型: TransformerEncoderLayer × {args.layers}（d={args.d_model} h={args.heads} "
          f"FFN={4*args.d_model} Pre-LN）  位置编码={'有' if not args.no_pos else '无'}  "
          f"因果掩码={'有' if not args.no_mask else '**无（泄漏实验）**'}  参数量 {n_par/1e6:.2f}M")

    # Adam + 固定学习率就够了（小模型，6 个 epoch），没有 warmup / cosine：
    # Pre-LN 的稳定性就是体现在「省掉这些也训得动」上，这点值得在作业里点一句
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    best = float("inf")
    print(f"\n{'Epoch':>6} {'TrainLoss':>10} {'ValPPL(全位置)':>15} {'ValPPL(末位)':>13} {'tok/s':>9}")
    print("-" * 60)
    for ep in range(1, args.epochs + 1):
        # train() 与 evaluate() 里的 eval() 成对出现：训练要开 dropout（正则），评测要关
        model.train()
        st, seen, tot = time.time(), 0, 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            # ★ 训练能「并行」的核心就这一行：
            #   一个 (B, T) 的 batch 里，每个位置 t 都是一个独立的预测任务，一次前向同时算出
            #   B×T 个预测的 loss，再一起反向。所以 Transformer 的样本吞吐和序列长度成正比，
            #   而 LSTM 必须沿时间步一格一格递推（t 的隐状态依赖 t-1），并行不了。
            #   reshape(-1, V) 把 (B, T, V) 拉平成 (B*T, V) 交给交叉熵，是这个「并行算全位置」的写法。
            #   注意：这只在「答案已经在输入里」的训练阶段成立；推理时下一个字还不存在，只能逐 token 生成。
            loss = F.cross_entropy(model(x).reshape(-1, V), y.reshape(-1))
            opt.zero_grad()
            loss.backward()
            # 梯度裁剪 1.0：Transformer 早期偶尔会出一两个特别大的梯度（共享权重的输出头
            # 叠加了输入侧梯度，更容易冲），裁一下能防住偶发的 loss 尖刺，几乎不影响正常收敛
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            # 按 token 数加权累加 loss（和 evaluate 里同一个口径），这样打印的 TrainLoss 才能和
            # 验证集的 loss 直接对照看
            tot += float(loss) * y.numel()
            seen += y.numel()
        dt = time.time() - st
        # 每轮评两次：全位置的（和被 LSTM 对比的常规口径）+ 只取末位的（防自欺的诚实口径）。
        # 带 --no-mask 时这两列会拉开一个数量级（1.07 vs 98.23），这个反差就是实验的全部看点
        _, v_ppl = evaluate(model, val_loader, device, last_only=False)
        _, v_ppl_last = evaluate(model, val_loader, device, last_only=True)
        mark = ""
        if v_ppl < best:
            best = v_ppl
            mark = " *"
            # 存 ckpt 时把「重建模型需要的一切」都塞进去：
            #   model_state —— 权重；
            #   char2idx / idx2char —— 字表，缺了它没法把 id 变回汉字（字表顺序是随语料变的！）；
            #   args —— 结构超参 + 那两个开关，generate 和 demo 靠它原样还原模型结构，
            #           否则 state_dict 的形状对不上，load_state_dict 会直接报错。
            # 注意：判断「最优」用的是全位置 PPL。所以 --no-mask 的那次训练里，
            # 这个指标本身在奖励「抄答案」，存下来的 ckpt 恰恰是最会复读的那个 —— 这点课上可以当彩蛋讲
            torch.save({"model_state": model.state_dict(), "char2idx": char2idx,
                        "idx2char": idx2char, "args": vars(args)}, args.save)
        print(f"{ep:>6} {tot/seen:>10.4f} {v_ppl:>15.2f} {v_ppl_last:>13.2f} {seen/dt:>9,.0f}{mark}")

    print(f"\n最佳验证 PPL（全位置）: {best:.2f}  已保存 {args.save}")


@torch.no_grad()
def generate(args):
    """
    续写：把 args.generate 这段开头喂进去，逐字往外蹦 n_chars 个新字。

    为什么推理必须「逐 token」而不能像训练那样一次算完全部位置：
        训练时每个位置 t 的标签 x_{t+1} 已经在输入序列里了，所以能一次并行算 B×T 个 loss；
        生成时第 t+1 个字还不存在 —— 必须先采样出它，才能拿它当前文去预测第 t+2 个。
        这就是自回归模型的串行本质（也是 KV Cache、投机解码等推理优化存在的原因）。
    这里每一步都把整段上文重新喂一遍（没有 KV Cache），纯粹为了代码短好读；
    真要快的话应该缓存每层的 K/V，让老位置只算一次。参数一多、序列一长，这个差别非常大。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # map_location：让「在 GPU 上训的 ckpt 能在 CPU 机器上加载」，学生换机器不会卡在这里
    ck = torch.load(args.ckpt, map_location=device)
    c = ck["args"]
    char2idx, idx2char = ck["char2idx"], ck["idx2char"]
    # 用 ckpt 里记下的超参重建同构模型（含 no_mask / no_pos 两个开关）。
    # dropout 传 0.0：生成过程中本来就该关 dropout，这里顺手传 0 双保险
    model = TransformerLM(len(char2idx), c["d_model"], c["heads"], c["layers"], c["seq_len"], 0.0,
                          use_pos=not c.get("no_pos", False),
                          causal=not c.get("no_mask", False)).to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()      # 关 dropout，否则同一个开头每次生成的风格都会漂
    # 开头里不在字表的字直接跳过（否则 KeyError）；如果一个都没有，out 会是空列表
    ids = [char2idx[ch] for ch in args.generate if ch in char2idx]
    out = list(ids)
    for _ in range(args.n_chars):
        # 只喂最近 seq_len 个 token：位置编码最多只有 seq_len 行（max_len 就是按 seq_len 建的），
        # 而且训练时模型也只见过长度 ≤ seq_len 的上下文，超长输入反而会让位置分布跑出训练区间。
        # [0, -1] 取 batch 里第一条、最后一个位置的 logits —— 那是对「下一个字」的打分
        logits = model(torch.tensor([out[-args.seq_len:]], device=device))[0, -1]
        if args.temperature <= 0.01:
            # 温度趋近 0 等价于贪心解码（argmax）：结果完全确定，适合课上反复讲同一个例子。
            # 用阈值而不是 ==0，是为了兼容命令行传 0 时可能出现的浮点误差，也避免下面除零
            nxt = int(logits.argmax())
        else:
            # 温度采样：logits / T 再 softmax。T<1 分布更尖（更保守、更像贪心），
            # T>1 更平（更发散、更容易出语法错误）。0.8 是个常用折中
            nxt = int(torch.multinomial(F.softmax(logits / args.temperature, dim=-1), 1))
        out.append(nxt)
    print(f"输入: {args.generate}")
    # 只打印「新生成的部分」（out 里前 len(ids) 个是用户给的开头）
    print(f"续写: {''.join(idx2char[i] for i in out[len(ids):])}")


def main():
    ap = argparse.ArgumentParser()
    # 默认语料指向脚本同目录的 corpus_sample.txt：80 万字样例语料，学生拿到就能跑通，
    # 不用先去课件目录找 corpus_general.txt
    default_corpus = os.path.join(HERE, "corpus_sample.txt")
    ap.add_argument("--corpus", default=default_corpus, help="语料路径（支持通配符）")
    ap.add_argument("--max_chars", type=int, default=0, help="只用前 N 个字符（CPU 慢时截断）")
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
    ap.add_argument("--generate", default=None, help="给一段开头，续写文本")
    ap.add_argument("--ckpt", default="tf_best.pt")
    ap.add_argument("--n_chars", type=int, default=40)
    ap.add_argument("--temperature", type=float, default=0.8)
    args = ap.parse_args()

    # 一条命令两用：带 --generate 就只做续写（不再训练），否则进入训练。
    # 这样学生不用记两个脚本名；注意 --generate 模式下其余训练参数（--epochs 等）会被忽略
    if args.generate:
        generate(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
