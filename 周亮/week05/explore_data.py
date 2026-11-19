"""explore_data.py — SFT 训练数据分析：长度分布 → 选 max_len；数据质量检查

输出：
  1) 数据质量检查：空 instruction / 空 output / 完全重复样本计数
  2) 长度分布：instruction / output / 总长（instruction+output 字符数）的分位数表
     —— 我们是字符级模型，字符数就是 token 数，P95 直接可用于选 max_len
  3) 总长的文本直方图（纯 print，不依赖 matplotlib）
  4) 结论：P95 + 3 个模板标记位 → 建议的 max_len

用法：
  python explore_data.py
"""
import json
import os
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(HERE, "data", "sft_train.jsonl")
VAL_PATH = os.path.join(HERE, "data", "sft_val.jsonl")

# 模板：<|user|> + instruction + <|assistant|> + output + <|end|>
# 特殊标记整串算 1 个 token，共 3 个额外位置
N_SPECIAL = 3


def load_rows(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def quantiles(xs):
    """分位数（经验近似：排序后按比例取下标）"""
    xs = sorted(xs)
    n = len(xs)
    return {
        "min": xs[0],
        "P50": xs[int(0.50 * n)],
        "P90": xs[int(0.90 * n)],
        "P95": xs[int(0.95 * n)],
        "P99": xs[min(n - 1, int(0.99 * n))],
        "max": xs[-1],
    }


def hist_text(xs, bucket=16, width=40):
    """终端文本直方图：按 bucket 字符一桶，# 长度与样本数成正比"""
    lo, hi = min(xs), max(xs)
    nb = (hi - lo) // bucket + 1
    counts = [0] * nb
    for x in xs:
        counts[(x - lo) // bucket] += 1
    cmax = max(counts)
    for i, c in enumerate(counts):
        bar = "#" * max(1, round(c / cmax * width))
        print(f"  {lo + i * bucket:>4}-{lo + i * bucket + bucket - 1:>4} 字 | {bar} {c}")


def analyze(path, name):
    rows = load_rows(path)
    print(f"\n{'=' * 56}")
    print(f"{name}：{len(rows)} 条")
    print("=" * 56)

    # ── 数据质量检查 ──
    empty_i = sum(1 for r in rows if not str(r.get("instruction", "")).strip())
    empty_o = sum(1 for r in rows if not str(r.get("output", "")).strip())
    pairs = Counter((r.get("instruction", ""), r.get("output", "")) for r in rows)
    dups = sum(c - 1 for c in pairs.values() if c > 1)
    print(f"[质量] 空 instruction: {empty_i} | 空 output: {empty_o} | 完全重复样本: {dups}")

    # ── 长度分布 ──
    lens_i = [len(r["instruction"]) for r in rows]
    lens_o = [len(r["output"]) for r in rows]
    lens_t = [a + b for a, b in zip(lens_i, lens_o)]
    for label, xs in [("instruction", lens_i), ("output", lens_o), ("总长(instr+output)", lens_t)]:
        qs = quantiles(xs)
        mean = sum(xs) / len(xs)
        detail = "  ".join(f"{k}={v}" for k, v in qs.items())
        print(f"[长度] {label:<20} mean={mean:6.1f}  {detail}")

    print("[直方图] 总长分布（每桶 16 字符）:")
    hist_text(lens_t)
    return rows, lens_t


def main():
    rows, lens_t = analyze(TRAIN_PATH, "训练集 sft_train.jsonl")
    analyze(VAL_PATH, "验证集 sft_val.jsonl")

    # ── max_len 结论 ──
    p95 = quantiles(lens_t)["P95"]
    need = p95 + N_SPECIAL
    max_len = 128
    cover = sum(1 for t in lens_t if t + N_SPECIAL <= max_len) / len(lens_t) * 100
    print(f"\n[结论] P95={p95} 字 + {N_SPECIAL} 个模板标记位 = {need} → "
          f"建议 max_len={max_len}（覆盖 {cover:.1f}% 样本，含全部截断保护）")


if __name__ == "__main__":
    main()
