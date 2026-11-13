"""
fit_check.py — 「集内拟合」检查：同一套问题，模型在**训练集**上答得出来吗？

用途：验证「SFT 能不能把训练集背下来」。三件事要一起看：
    1) teacher forcing 准确率 —— 喂 gold 答案，看每个位置 argmax 对不对（考的是记忆）
    2) 贪心生成 exact match —— 自己往下写，看能不能复现参考答案（考的是「记忆能不能被用出来」）
    3) 验证集上的同一指标 —— 作为对照；训练集高、验证集低，就是典型的「记住了但没学会」

⚠ 前提：训练时要 `--save-best last`。默认的 `--save-best val` 只会存验证集最优的那轮，
   过拟合后验证集一路变差，你拿到手的其实是第 1~2 轮的模型，看起来「训练 loss 都 0.004 了还不会答」。

用法:
    python fit_check.py --ckpt sft_fit.pt --train        # 看训练集（前 200 条）
    python fit_check.py --ckpt sft_fit.pt --val          # 看验证集
"""

import argparse
import json
import os
from collections import Counter

import torch

from demo_sft import load
from sft import build_prompt, generate, read_jsonl
from transformer_lm import SPECIAL_TOKENS, encode

HERE = os.path.dirname(os.path.abspath(__file__))
END = SPECIAL_TOKENS[2]


def char_f1(pred, gold):
    """字符集合 F1：不要求顺序，给「答得像不像」一个宽松指标。"""
    a, b = Counter(pred), Counter(gold)
    same = sum((a & b).values())
    if same == 0:
        return 0.0
    p, r = same / max(1, sum(a.values())), same / max(1, sum(b.values()))
    return 2 * p * r / (p + r)


@torch.no_grad()
def tf_accuracy(model, rows, char2idx, idx2char, seq_len, device):
    """teacher forcing：把 gold 答案喂进去，数一数每个回答位置 argmax 猜对没有。"""
    hit, tot = 0, 0
    for q, a in rows:
        p = encode(build_prompt(q), char2idx)
        ids = (p + encode(a + END, char2idx))[:seq_len]
        if len(ids) <= len(p) + 1:
            continue
        x = torch.tensor([ids], device=device)
        pred = model(x)[0].argmax(-1)
        for t in range(len(p) - 1, len(ids) - 1):
            hit += int(int(pred[t]) == ids[t + 1])
            tot += 1
    return hit / max(1, tot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--train", action="store_true", help="在训练集上评估（默认）")
    ap.add_argument("--val", action="store_true", help="在验证集上评估")
    ap.add_argument("--n", type=int, default=200, help="抽多少条来评")
    ap.add_argument("--show", type=int, default=5)
    args = ap.parse_args()

    path = os.path.join(HERE, "sft_val.jsonl" if args.val else "sft_train.jsonl")
    rows = read_jsonl(path)[: args.n]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, c2i, i2c, seq = load(args.ckpt, device)

    tf_acc = tf_accuracy(model, rows, c2i, i2c, seq, device)
    em, f1s = 0, []
    for i, (q, gold) in enumerate(rows):
        pred = generate(model, c2i, i2c, q, seq, n_chars=min(seq - 8, 64), temperature=0.01)
        em += int(pred.strip() == gold.strip())
        f1s.append(char_f1(pred, gold))
        if i < args.show:
            print(f"  Q: {q[:38]}")
            print(f"     gold: {gold[:50]}")
            print(f"     pred: {pred[:50]}")

    n = len(rows)
    print(f"\n{'训练集' if not args.val else '验证集'}（{n} 条，ckpt={os.path.basename(args.ckpt)}）")
    print(f"  teacher-forcing 准确率 : {tf_acc:.1%}")
    print(f"  贪心生成 exact match  : {em}/{n} = {em/n:.1%}")
    print(f"  字符 F1               : {sum(f1s)/n:.3f}")


if __name__ == "__main__":
    main()
