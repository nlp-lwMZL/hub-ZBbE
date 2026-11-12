"""
基于字符级语言模型的拼音输入法。
用法:
    python pinyin_ime.py
    python pinyin_ime.py --model_path best_model.pt --topk 8 --beam 10
"""

import argparse
import glob
import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# 脚本所在目录：拼音表、模型权重、语料等输入文件优先按这里解析，
# 避免「在别的目录执行 python d:\...\pinyin_ime.py」时找不到同目录的文件
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def resolve_input(pattern):
    """相对路径的解析顺序：脚本所在目录 → 当前工作目录。绝对路径原样返回。"""
    if os.path.isabs(pattern):
        return pattern
    local = os.path.join(BASE_DIR, pattern)
    return local if glob.glob(local) else pattern


def resolve_file(path):
    """单个文件版：cwd 里已有就用 cwd 的，否则退回脚本目录。"""
    if os.path.isabs(path) or os.path.exists(path):
        return path
    local = os.path.join(BASE_DIR, path)
    return local if os.path.exists(local) else path

# ─────────────────────── 拼音 → 候选汉字映射表 ───────────────────────

def _load_pinyin_map(path):
    # 从 JSON 文件读取 {音节: [候选字, ...]} 映射表
    path = resolve_file(path)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"找不到拼音映射表文件: {path}\n"
            f"  脚本目录: {BASE_DIR}  当前工作目录: {os.getcwd()}"
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)

# 模块加载时为空，main() 中调用 _load_pinyin_map() 后填充
PINYIN_MAP = {}

# ─────────────────────── 模型定义 ───────────────────────
# 与 language_model.py 中的 LM 保持完全一致，以便正确加载 checkpoint

class LM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, model_type, dropout):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        # 根据 model_type 参数在 RNN 和 LSTM 之间切换
        rnn_cls = nn.LSTM if model_type == "lstm" else nn.RNN
        self.rnn = rnn_cls(
            embed_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            # 单层时 PyTorch 不允许设置 dropout，需显式置 0
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        # x: (B, T) 字符 id 序列
        e = self.drop(self.embed(x))       # (B, T, embed_dim)
        out, _ = self.rnn(e)               # (B, T, hidden_dim)，丢弃 hidden state
        return self.fc(self.drop(out))     # (B, T, vocab_size)


# ─────────────────────── 拼音分词 ───────────────────────

# 按音节长度降序排列，保证贪心匹配时优先命中较长的音节
# 例如 "zhuang" 应整体匹配，而非 "zh" + "uang"
# 模块级先置空列表，main() 加载 PINYIN_MAP 后重建
_SYLLABLES = []

def segment(pinyin_str):
    """
    将拼音字符串切分为音节列表。
    支持带空格（"huang jin"）和连续输入（"huangjin"）两种格式。 
    对每个空格分隔的 token 使用贪心最长匹配，无法识别的字符直接跳过。
    """
    syllables = []
    for token in pinyin_str.strip().lower().split():
        i = 0
        while i < len(token):
            # 在已排序的音节表中找第一个能匹配当前位置的音节
            matched = next((s for s in _SYLLABLES if token[i:].startswith(s)), None)
            if matched:
                syllables.append(matched)
                i += len(matched)
            else:
                i += 1  # 当前字符无法匹配任何音节，跳过
    return syllables


# ─────────────────────── 无上文时的字频先验 ───────────────────────
# 输入「wo ai ni」时第一个音节还没有上文，模型无法为「我」打分。
# 此时退回语料里的字频（unigram）—— 这也是最古老的语言模型：P(字)。
# 不加这一步，第一个音节的所有候选字得分相同，只能按拼音表的顺序瞎挑。

UNI_LOGPROB = None


def build_unigram_logprob(pattern, char2idx, device):
    """从语料统计字频，返回词表上的 log 概率（拉普拉斯平滑）。"""
    import collections

    counter = collections.Counter()
    for path in glob.glob(resolve_input(pattern)):
        with open(path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                counter.update(c for c in line if c in char2idx)

    probs = torch.ones(len(char2idx), dtype=torch.float)
    for c, n in counter.items():
        probs[char2idx[c]] += n
    probs = probs / probs.sum()
    return torch.log(probs).to(device)


# ─────────────────────── 束搜索 ───────────────────────

def beam_search(syllables, prefix, model, char2idx, idx2char, beam_size, device):
    """
    对音节列表逐字做束搜索，返回按得分降序排列的候选列表。

    参数:
        syllables : 本轮待转换的音节列表，如 ["huang", "jin"]
        prefix    : 已确认的历史文字，作为语言模型的上文
        beam_size : 束宽，每步保留的最优路径数量

    返回:
        [(累计log_prob, 转换结果字符串), ...]

    原理:
        每处理一个音节，就把当前所有 beam 与该音节的候选汉字做笛卡尔积展开，
        用语言模型对「prefix + 已生成部分」预测下一字的 log prob 作为得分增量，
        展开后按总得分取 top-beam_size 条路径继续。
    """
    beams = [(0.0, "")]  # 初始只有一条空路径，得分为 0

    for syllable in syllables:
        # 过滤掉不在训练词表中的候选字（模型无法为其打分），并去重
        # （拼音表里同一个字可能因为来源不同出现两次，不去重会在候选里出现重复行）
        candidates = list(dict.fromkeys(c for c in PINYIN_MAP.get(syllable, []) if c in char2idx))
        if not candidates:
            continue  # 该音节无可用候选，跳过（不终止整个搜索）

        new_beams = []
        for score, partial in beams:
            # 拼接历史上文与当前已生成部分，送入模型
            context = prefix + partial
            if context:
                ids = [char2idx[c] for c in context if c in char2idx]
                x = torch.tensor([ids], dtype=torch.long, device=device)
                with torch.no_grad():
                    logits = model(x)   # (1, T, vocab_size)
                # 只取最后一个时间步的输出，作为"下一字"的概率分布
                log_probs = F.log_softmax(logits[0, -1, :], dim=-1)
            else:
                # 没有上文 → 用字频先验（unigram）给候选字打分
                log_probs = UNI_LOGPROB

            for char in candidates:
                lp = log_probs[char2idx[char]].item()
                new_beams.append((score + lp, partial + char))

        # 按累计得分降序，保留 beam_size 条最优路径
        new_beams.sort(reverse=True)
        beams = new_beams[:beam_size]

    return beams


# ─────────────────────── 交互主循环 ───────────────────────

def run(model, char2idx, idx2char, topk, beam_size, device):
    """
    交互式输入法主循环。
    用户每轮输入一段拼音，程序展示 topk 个候选转换结果；
    用户选择编号后，结果追加到已确认文字，作为下一轮的上文。
    """
    print("=" * 52)
    print("  拼音输入法（字符级语言模型）")
    print("  输入拼音回车 → 选候选编号追加到已输入文字")
    print("  r = 重置  q = 退出")
    print("=" * 52)

    confirmed = ""  # 已确认的文字，累积作为语言模型上文

    while True:
        print(f"\n已输入: 「{confirmed}」" if confirmed else "\n已输入: （空）")
        raw = input("拼音> ").strip()

        if not raw:
            continue
        if raw == "q":
            print("退出。")
            break
        if raw == "r":
            confirmed = ""
            continue

        syllables = segment(raw)
        if not syllables:
            print("无法识别任何音节，请检查拼音拼写。")
            continue

        print(f"音节: {' '.join(syllables)}")
        results = beam_search(syllables, confirmed, model, char2idx, idx2char, beam_size, device)

        if not results:
            print("无候选结果。")
            continue

        print("候选:")
        for i, (score, text) in enumerate(results[:topk]):
            print(f"  [{i}] {text}  ({score:.2f})")

        choice = input("选择编号 (回车跳过): ").strip()
        if choice.isdigit():
            idx = int(choice)
            if 0 <= idx < len(results):
                confirmed += results[idx][1]
            else:
                print("编号超出范围。")


# ─────────────────────── 入口 ───────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="best_model.pt")
    parser.add_argument("--pinyin_map", default="pinyin_map.json", help="拼音映射表 JSON 文件")
    parser.add_argument("--topk",       type=int, default=5,  help="展示候选数")
    parser.add_argument("--beam",       type=int, default=10, help="束搜索宽度")
    parser.add_argument("--unigram",    default="*.txt", help="统计字频先验用的语料（无上文的第一个字）")
    args = parser.parse_args()

    # 加载拼音映射表，并重建音节排序列表
    global PINYIN_MAP, _SYLLABLES
    PINYIN_MAP = _load_pinyin_map(args.pinyin_map)
    _SYLLABLES = sorted(PINYIN_MAP.keys(), key=len, reverse=True)
    print(f"拼音表: {args.pinyin_map}  ({len(PINYIN_MAP)} 个音节)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 从 checkpoint 中恢复词表和模型超参
    ckpt     = torch.load(resolve_file(args.model_path), map_location=device)
    char2idx = ckpt["char2idx"]
    idx2char = ckpt["idx2char"]
    cfg      = ckpt["args"]

    model = LM(
        vocab_size = len(char2idx),
        embed_dim  = cfg["embed_dim"],
        hidden_dim = cfg["hidden_dim"],
        num_layers = cfg["num_layers"],
        model_type = cfg["model"],
        dropout    = 0.0,   # 推理阶段关闭 dropout，保证输出确定性
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # 无上文的第一个字：用字频先验（unigram）
    global UNI_LOGPROB
    UNI_LOGPROB = build_unigram_logprob(args.unigram, char2idx, device)
    print(f"字频先验: {args.unigram}")

    print(f"模型: {args.model_path}  ({cfg['model'].upper()}, 词表 {len(char2idx)} 字)")
    run(model, char2idx, idx2char, args.topk, args.beam, device)


if __name__ == "__main__":
    main()
