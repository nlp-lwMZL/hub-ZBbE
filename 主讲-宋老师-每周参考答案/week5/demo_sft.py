"""
demo_sft.py — 课堂演示 / 作业自查：同一道题，看「SFT 前」和「SFT 后」的区别

左边是基座（只会续写的接龙器），右边是 SFT 之后的模型（在 <|assistant|> 后面作答、说完就停）。
两边跑的是同一批**验证集**问题 —— 也就是训练时没见过的题。

用法:
    python demo_sft.py                          # 默认看 5 条
    python demo_sft.py --n 8 --temperature 0    # 贪心解码，结果确定，适合课堂复现
"""

import argparse
import json
import os

import torch
import torch.nn.functional as F

from sft import build_prompt, generate
from transformer_lm import SPECIAL_TOKENS, TransformerLM, encode

HERE = os.path.dirname(os.path.abspath(__file__))
USER, ASSISTANT, END = SPECIAL_TOKENS


def load(path, device):
    """载入一个 ckpt，按里面记下的超参重建模型（ckpt 存了结构 + 字表，所以能独立加载）。"""
    ck = torch.load(path, map_location=device)
    c = ck["args"]
    model = TransformerLM(len(ck["char2idx"]), c["d_model"], c["heads"], c["layers"],
                          c["seq_len"], 0.0,          # 推理时 dropout 必须关掉
                          use_pos=not c.get("no_pos", False),
                          causal=not c.get("no_mask", False)).to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()
    return model, ck["char2idx"], ck["idx2char"], c["seq_len"]


@torch.no_grad()
def base_continue(model, char2idx, idx2char, text, seq_len, n_chars=30, temperature=0.6):
    """基座的用法：直接喂一句话让它接着写 —— 没有模板，也没有「答完就停」的概念。"""
    device = next(model.parameters()).device
    ids = encode(text, char2idx)
    out = list(ids)
    for _ in range(n_chars):
        logits = model(torch.tensor([out[-seq_len:]], device=device))[0, -1]
        nxt = int(logits.argmax()) if temperature <= 0.01 else \
            int(torch.multinomial(F.softmax(logits / temperature, dim=-1), 1))
        out.append(nxt)
    return "".join(idx2char[i] for i in out[len(ids):]).replace("\n", " ")


@torch.no_grad()
def answer_ppl(model, rows, char2idx, seq_len, device):
    """
    验证集上只算「回答段」的 PPL：模板和问题位置的标签是 -100，不参与。
    这是 SFT 的标准评测口径 —— 我们关心的是「在回答位置上写得好不好」。
    """
    total, n_tok = 0.0, 0
    end_id = char2idx[END]
    for q, a in rows:
        p = encode(build_prompt(q), char2idx)
        ids = (p + encode(a + END, char2idx))[:seq_len]
        if len(ids) < len(p) + 2:
            continue
        x = torch.tensor([ids], device=device)
        y = torch.full((1, len(ids)), -100, dtype=torch.long, device=device)
        for t in range(len(p) - 1, len(ids) - 1):
            y[0, t] = ids[t + 1]
        logits = model(x)
        total += float(F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                       y.reshape(-1), ignore_index=-100, reduction="sum"))
        n_tok += int((y != -100).sum())
    loss = total / max(1, n_tok)
    return loss, torch.exp(torch.tensor(loss)).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.path.join(HERE, "base_lm.pt"), help="SFT 之前的基座")
    ap.add_argument("--sft", default=os.path.join(HERE, "sft_mask.pt"), help="SFT 之后的模型")
    ap.add_argument("--val", default=os.path.join(HERE, "sft_val.jsonl"))
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.6)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m_base, c2i, i2c, seq = load(args.base, device)
    m_sft, _, _, _ = load(args.sft, device)

    rows = []
    with open(args.val, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                rows.append((rec["instruction"], rec["output"]))
    rows = rows[: args.n]

    print(f"解码：{'贪心' if args.temperature <= 0.01 else f'温度 {args.temperature}'}"
          f"   题目：验证集前 {len(rows)} 条（训练时没见过）\n")
    for i, (q, a) in enumerate(rows, 1):
        print(f"── {i}. 问：{q}")
        print(f"   参考答案：{a}")
        print(f"   SFT 前  ：{base_continue(m_base, c2i, i2c, q, seq, temperature=args.temperature)}")
        print(f"   SFT 后  ：{generate(m_sft, c2i, i2c, q, seq, temperature=args.temperature)}")
        print()

    print("=" * 60)
    print("验证集「回答段」PPL（越低说明在回答位置上的措辞越贴训练数据的风格）")
    for name, m in (("SFT 前", m_base), ("SFT 后", m_sft)):
        loss, ppl = answer_ppl(m, rows, c2i, seq, device)
        print(f"  {name:8s} loss={loss:6.3f}  PPL={ppl:9.2f}")
    print("\n注：基座从没学过「回答」这个动作，所以它的 PPL 高得多 —— 这不是模型差，"
          "而是它擅长的不是这件事。")


if __name__ == "__main__":
    main()
