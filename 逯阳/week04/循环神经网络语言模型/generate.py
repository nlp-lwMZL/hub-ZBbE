"""
文本生成验证脚本 —— 用来检验 transformer_lm.py 训练出的模型是否真的学会了语言。

做两件事：
    1) 掩码自检：扰动序列末位字符，检查前面位置的输出是否变化。
       因果掩码生效 → 前面位置看不到末位，输出不变（差值 ≈ 0）；
       双向注意力   → 前面位置能看到末位，输出改变（差值明显 > 0）。
    2) 自回归生成：给定开头文字，逐字采样续写，并报告生成片段的平均 log 概率 / PPL。
       PPL 明显低于随机猜测（≈ 词表大小）说明模型确实学到了字与字之间的规律。

关于文本结尾：
    训练时每篇文档编码成 <bos> + 正文 + <eos>，所以模型自己知道哪里该收尾。
    生成时预测到 <eos> 就停（除非 --stop_at_eos 0），并在输出里报告是否正常收尾。
    提示为空时从 <bos> 起笔，相当于「让模型自己开一篇新的」。

用法:
    python generate.py --model_path best_transformer.pt --prompt "春天来了"
    python generate.py --prompt "春天来了" --temperature 0.8 --top_k 40 --max_new_tokens 200
    python generate.py --prompt "春天" --num_samples 3 --top_p 0.9
    python generate.py --prompt "" --num_samples 3                 # 让模型自己开头 + 自己收尾
    python generate.py --model_path best_model.pt --prompt "我们"   # 也能加载旧的 RNN/LSTM 权重
"""

import argparse
import glob
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# 脚本所在目录：权重等输入文件优先按这里解析，避免在别的目录执行时找不到
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 与训练脚本一致的特殊 token；老 checkpoint 里没有它们，取不到就退化成原来的行为
BOS, EOS = "<bos>", "<eos>"


def resolve_file(path):
    """相对路径：cwd 里已有就用 cwd 的，否则退回脚本目录。"""
    if os.path.isabs(path) or os.path.exists(path):
        return path
    local = os.path.join(BASE_DIR, path)
    return local if os.path.exists(local) else path


# ─────────────────────── 模型定义 ───────────────────────
# 必须与 transformer_lm.py / language_model.py 中的定义逐字段一致，否则权重加载会失败。

class TransformerLM(nn.Module):
    """字符级 Transformer 语言模型（与 transformer_lm.py 保持完全一致）。"""

    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers,
                 nhead, dropout, max_len, causal=True):
        super().__init__()
        if embed_dim % nhead != 0:
            raise ValueError(f"embed_dim({embed_dim}) 必须能被 nhead({nhead}) 整除")

        self.max_len = max_len
        self.causal = causal
        self.vocab_size = vocab_size

        self.tok_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(max_len, embed_dim)
        self.drop = nn.Dropout(dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(
            layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, vocab_size)

        self._mask_cache = {}

    def _causal_mask(self, T, device):
        key = (T, str(device))
        mask = self._mask_cache.get(key)
        if mask is None:
            mask = torch.full((T, T), float("-inf"), device=device).triu(1)
            self._mask_cache[key] = mask
        return mask

    def forward(self, x):
        B, T = x.shape
        if T > self.max_len:
            raise ValueError(f"序列长度 {T} 超过 max_len={self.max_len}")
        pos = torch.arange(T, device=x.device)
        h = self.drop(self.tok_embed(x) + self.pos_embed(pos))
        mask = self._causal_mask(T, x.device) if self.causal else None
        h = self.blocks(h, mask=mask)
        return self.fc(self.norm(h))


class RNNLM(nn.Module):
    """旧版 RNN/LSTM 语言模型，仅为兼容已有 checkpoint（如 best_model.pt）。"""

    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, model_type, dropout):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        rnn_cls = nn.LSTM if model_type == "lstm" else nn.RNN
        self.rnn = rnn_cls(
            embed_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        # 供生成脚本统一读取
        self.max_len = None
        self.causal = True

    def forward(self, x):
        e = self.drop(self.embed(x))
        out, _ = self.rnn(e)
        return self.fc(self.drop(out))


def build_model(ckpt, device):
    """按 checkpoint 里存的超参复原网络结构。"""
    cfg = ckpt["args"]
    vocab_size = len(ckpt["char2idx"])
    kind = cfg.get("model", "lstm")

    # 保险：老 checkpoint 里可能没有 "model" 字段，直接看权重名字判断结构
    if "tok_embed.weight" in ckpt["model_state"]:
        kind = "transformer"

    if kind == "transformer":
        model = TransformerLM(
            vocab_size = vocab_size,
            embed_dim  = cfg["embed_dim"],
            hidden_dim = cfg["hidden_dim"],
            num_layers = cfg["num_layers"],
            nhead      = cfg["nhead"],
            dropout    = 0.0,                    # 推理关闭 dropout，保证确定性
            max_len    = cfg["max_len"],
            causal     = bool(cfg.get("causal", 1)),
        )
    else:
        model = RNNLM(
            vocab_size = vocab_size,
            embed_dim  = cfg["embed_dim"],
            hidden_dim = cfg["hidden_dim"],
            num_layers = cfg["num_layers"],
            model_type = kind,
            dropout    = 0.0,
        )

    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, cfg, kind


# ─────────────────────── 掩码自检 ───────────────────────

def check_causal_mask(model, vocab_size, device, seq_len=16):
    """
    验证「未来信息是否泄漏」：
    把序列最后一个字符换成别的字符，看前 seq_len-1 个位置的输出是否变化。
        因果掩码 → 变化量 ≈ 0（前面看不到后面）
        无掩码   → 变化量 > 0（前面看到了后面）
    """
    model.eval()
    x = torch.randint(0, vocab_size, (1, seq_len), device=device)
    x2 = x.clone()
    x2[0, -1] = (x2[0, -1] + 1) % vocab_size      # 只改末位

    with torch.no_grad():
        a = model(x)
        b = model(x2)

    diff = (a[0, :-1] - b[0, :-1]).abs().max().item()
    return diff


# ─────────────────────── 采样 ───────────────────────

def apply_top_k_top_p(logits, top_k=0, top_p=1.0):
    """top-k / nucleus 过滤，把不可能的字直接删掉再归一化。"""
    if top_k and top_k > 0:
        k = min(top_k, logits.size(-1))
        kth = torch.topk(logits, k).values[..., -1, None]
        logits = logits.masked_fill(logits < kth, float("-inf"))

    if top_p and 0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cum = probs.cumsum(dim=-1)
        # 保留累计概率刚好越过 top_p 的最短前缀（至少保留 1 个）
        remove = cum - probs > top_p
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)

    return logits


def apply_repetition_penalty(logits, generated_ids, penalty=1.0):
    """对已出现过的字做惩罚，抑制字符级模型常见的死循环。penalty=1.0 表示关闭。"""
    if penalty == 1.0 or not generated_ids:
        return logits
    ids = torch.tensor(sorted(set(generated_ids)), device=logits.device)
    sel = logits[ids]
    logits[ids] = torch.where(sel < 0, sel * penalty, sel / penalty)
    return logits


@torch.no_grad()
def generate(model, char2idx, idx2char, prompt, max_new_tokens,
             temperature=1.0, top_k=0, top_p=1.0, repetition_penalty=1.0,
             device="cpu", greedy=False, stop_at_newline=False, stop_at_eos=True):
    """
    自回归生成：每次只把「最后一个位置的输出」当作下一个字的分布。
    上下文超过 max_len 时，滑动窗口只保留最近的 max_len 个字符。

    有 <bos>/<eos> 的模型：从 <bos> + 提示 起笔，预测到 <eos> 就停。
    老 checkpoint 没有这两个 token 时，退化成从提示（或换行符）起笔、按长度停。
    """
    bos_id = char2idx.get(BOS)
    eos_id = char2idx.get(EOS)

    unk = [c for c in prompt if c not in char2idx]
    if unk:
        print(f"  (提示里有 {len(unk)} 个字符不在字表中，已忽略: {''.join(unk)[:20]})")

    ids = [bos_id] if bos_id is not None else []
    ids += [char2idx[c] for c in prompt if c in char2idx]

    if len(ids) == (1 if bos_id is not None else 0):
        # 没有可用提示：有 <bos> 就从它起笔（让模型自己开一篇），否则退回换行符
        if bos_id is None:
            ids = [char2idx.get("\n", 0)]
            print("  (提示为空或全被过滤，改用换行符作为起始)")
        else:
            print(f"  (提示为空，从 {BOS} 起笔，让模型自己开头)")

    ctx_limit = model.max_len if model.max_len else 10 ** 9
    logprob_sum = 0.0
    n_scored = 0
    newline_id = char2idx.get("\n")
    hit_eos = False

    for _ in range(max_new_tokens):
        ctx = ids[-ctx_limit:]
        x = torch.tensor([ctx], dtype=torch.long, device=device)
        logits = model(x)[0, -1, :].float()          # (V,) 下一个字的未归一化分数

        logits = apply_repetition_penalty(logits, ids, repetition_penalty)

        if greedy:
            next_id = int(torch.argmax(logits))
            logprob_sum += F.log_softmax(logits, dim=-1)[next_id].item()
        else:
            scaled = apply_top_k_top_p(logits / max(temperature, 1e-6), top_k, top_p)
            log_probs = F.log_softmax(scaled, dim=-1)
            next_id = int(torch.multinomial(log_probs.exp(), num_samples=1))
            logprob_sum += log_probs[next_id].item()

        # 模型自己判断说完了 —— 这是 <eos> 训练带来的直接收益
        if stop_at_eos and eos_id is not None and next_id == eos_id:
            hit_eos = True
            break

        n_scored += 1
        ids.append(next_id)

        if stop_at_newline and next_id == newline_id:
            break

    text = "".join(idx2char.get(i, "") for i in ids if i != bos_id and i != eos_id)
    return text, logprob_sum, n_scored, hit_eos


# ─────────────────────── 主函数 ───────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="best_transformer.pt")
    parser.add_argument("--prompt",     default="春天来了", help="开头文字（空字符串则随机起笔）")
    parser.add_argument("--max_new_tokens", type=int, default=200, help="续写长度")
    parser.add_argument("--temperature", type=float, default=0.9, help="<1 更保守，>1 更发散")
    parser.add_argument("--top_k",      type=int, default=40, help="每步只从概率最高的 k 个字里采样，0=关闭")
    parser.add_argument("--top_p",      type=float, default=0.95, help="nucleus 采样阈值，1.0=关闭")
    parser.add_argument("--repetition_penalty", type=float, default=1.1, help="重复惩罚，1.0=关闭")
    parser.add_argument("--greedy",     type=int, default=0, choices=[0, 1], help="1=贪心解码（最保守，容易复读）")
    parser.add_argument("--num_samples", type=int, default=1, help="生成几段做对比")
    parser.add_argument("--stop_at_newline", type=int, default=0, choices=[0, 1], help="1=遇到换行就停")
    parser.add_argument("--stop_at_eos", type=int, default=1, choices=[0, 1], help="1=模型输出 <eos> 就停（需模型训练时带特殊 token）")
    parser.add_argument("--check_mask", type=int, default=1, choices=[0, 1], help="1=运行因果掩码自检")
    parser.add_argument("--seed",       type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(resolve_file(args.model_path), map_location=device)
    model, cfg, kind = build_model(ckpt, device)
    char2idx, idx2char = ckpt["char2idx"], ckpt["idx2char"]
    vocab_size = len(char2idx)
    has_special = BOS in char2idx and EOS in char2idx

    causal = bool(getattr(model, "causal", True))
    print("=" * 64)
    print(f"模型: {args.model_path}")
    print(f"类型: {kind.upper()}"
          + (f"  层数={cfg['num_layers']}  d_model={cfg['embed_dim']}  "
             f"nhead={cfg['nhead']}  max_len={cfg['max_len']}" if kind == "transformer"
             else f"  层数={cfg['num_layers']}  hidden={cfg['hidden_dim']}"))
    print(f"词表: {vocab_size} 字   参数: {sum(p.numel() for p in model.parameters()):,}")
    print(f"causal 掩码: {'开' if causal else '关'}")
    print(f"特殊 token: {'有 <bos>/<eos>（模型会自己收尾）' if has_special else '无（按长度停，旧 checkpoint）'}")
    print("=" * 64)

    # —— 1) 掩码自检 ——
    if args.check_mask:
        seq_len = min(16, model.max_len or 16)
        diff = check_causal_mask(model, vocab_size, device, seq_len)
        print(f"\n[掩码自检] 扰动末位字符后，前面位置的输出最大变化量 = {diff:.3e}")
        if diff < 1e-4:
            print("  → 前文对末位改动不敏感：因果掩码生效，没有未来信息泄漏。")
        else:
            print("  → 前文随末位改变：注意力是双向的，存在未来信息泄漏。")
            print("    这种模型训练时能看到答案，生成质量会明显下降，建议用 --causal 1 重训。")

    if kind == "transformer" and not causal:
        print("\n!! 警告：该 checkpoint 训练时未加因果掩码，下列生成结果仅供参考。")

    # —— 2) 文本生成 ——
    print(f"\n[生成] 提示: 「{args.prompt}」  temperature={args.temperature}  "
          f"top_k={args.top_k}  top_p={args.top_p}  "
          f"greedy={bool(args.greedy)}  长度={args.max_new_tokens}\n")

    for i in range(args.num_samples):
        text, logprob_sum, n, hit_eos = generate(
            model, char2idx, idx2char, args.prompt, args.max_new_tokens,
            temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
            repetition_penalty=args.repetition_penalty, device=device,
            greedy=bool(args.greedy), stop_at_newline=bool(args.stop_at_newline),
            stop_at_eos=bool(args.stop_at_eos),
        )
        avg_nll = -logprob_sum / max(1, n)
        ppl = math.exp(min(avg_nll, 700))
        ending = "模型自己输出了 <eos>，正常收尾" if hit_eos else "未触发 <eos>，到长度上限截断"
        print(f"--- 样本 {i + 1} （{n} 字，平均 PPL {ppl:.1f}，随机基线 {vocab_size}）---")
        print(text)
        print(f"  [{ending}]\n")


if __name__ == "__main__":
    main()
