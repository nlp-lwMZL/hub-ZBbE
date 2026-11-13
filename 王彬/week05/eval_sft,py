"""离线评估 SFT 模型：不同解码策略下的准确率与生成质量。

用法：
    python eval_sft.py                          # 默认 100 条，温度 0/0.1/0.5/0.8，无额外解码干预
    python eval_sft.py --n 200 --temps 0.1,0.5 --rep_penalties 1.0,1.2 --digits 0,1
"""
import argparse
import os
import re
import time

import torch

import transformer_sft as t

HERE = os.path.dirname(os.path.abspath(__file__))
NUM = re.compile(r"-?\d+(?:\.\d+)?")


def nums(s):
    return NUM.findall(s)


def has_repeat(s, k=4):
    """是否出现重复的 k 字片段（复读的近似指标）。"""
    grams = [s[i:i + k] for i in range(len(s) - k + 1)]
    return len(grams) != len(set(grams))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(HERE, "sft_transformer.pt"))
    ap.add_argument("--val", default=os.path.join(HERE, "sft_val.jsonl"))
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--temps", default="0,0.1,0.5,0.8")
    ap.add_argument("--rep_penalties", default="1.0", help="逗号分隔的重复惩罚系数")
    ap.add_argument("--digits", default="0", help="逗号分隔：1 表示只允许数字+结束符")
    ap.add_argument("--max_new_chars", type=int, default=80)
    args = ap.parse_args()

    device = torch.device("cpu")
    ck = torch.load(args.ckpt, map_location=device)
    c = ck["args"]
    c2i, i2c = ck["char2idx"], ck["idx2char"]
    model = t.TransformerLM(len(c2i), c["d_model"], c["heads"], c["layers"],
                            c["seq_len"], 0.0)
    model.load_state_dict(ck["model_state"])
    model.to(device)
    model.eval()

    digit_ids = {i for ch, i in c2i.items() if ch in t.DIGIT_CHARS + t.END_TAG}

    rows = t.load_jsonl(args.val)[: args.n]
    n_num = sum(1 for r in rows if nums(r["output"]))
    print(f"ckpt={os.path.basename(args.ckpt)}  样本={len(rows)} 条"
          f"（含数字答案 {n_num} 条）  字表={len(c2i)}")

    header = (f"{'温度':>4} {'惩罚':>4} {'约束':>4} {'EM%':>6} {'首数字%':>8}"
              f" {'全数字%':>8} {'复读%':>7} {'平均长度':>8} {'耗时':>6}")
    print(header)
    print("-" * len(header))

    for temp in [float(x) for x in args.temps.split(",")]:
        for rep in [float(x) for x in args.rep_penalties.split(",")]:
            for digits in [int(x) for x in args.digits.split(",")]:
                t0 = time.time()
                em = first = allnum = nrepeat = 0
                lens = []
                for r in rows:
                    pred = t.answer_question(
                        model, c2i, i2c, r["instruction"], device, c["seq_len"],
                        max_new_chars=args.max_new_chars, temperature=temp,
                        repetition_penalty=rep,
                        allowed_ids=digit_ids if digits else None)
                    gold = r["output"]
                    lens.append(len(pred))
                    if pred.strip() == gold.strip():
                        em += 1
                    pg, gg = nums(pred), nums(gold)
                    if pg and gg:
                        if pg[0] == gg[0]:
                            first += 1
                        if all(g in pg for g in gg):
                            allnum += 1
                    if has_repeat(pred):
                        nrepeat += 1
                n = len(rows)
                dt = time.time() - t0
                print(f"{temp:>4.1f} {rep:>4.1f} {'数字' if digits else '-':>4}"
                      f" {100*em/n:>6.1f} {100*first/n_num:>8.1f}"
                      f" {100*allnum/n_num:>8.1f} {100*nrepeat/n:>7.1f}"
                      f" {sum(lens)/n:>8.1f} {dt:>5.0f}s")


if __name__ == "__main__":
    main()
