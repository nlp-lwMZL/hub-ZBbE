# -*- coding: utf-8 -*-
"""
把本次重跑的数字与 `ppt准备材料/week7_extraction/_NUMBERS.md` 里的历史值逐项对比。

为什么需要这个脚本（而不是人肉对表）：
  重构 + 重跑之后，最需要回答的问题是「**数字有没有变**」。
  人肉对 20 多行 × 4 个指标非常容易看漏，所以把历史值固化成一个字典、逐项算差，
  并把超过阈值（默认 0.5pp）的项显式标红 —— 这正是「重构不能改变数值行为」的验收依据。

历史值的来源：`_NUMBERS.md` §主表 / §消融表（2026-11-16 那一轮实测）。
如果以后又更新了 `_NUMBERS.md`，把下面的 OLD 字典同步改掉即可。

用法：
  python eval/compare_to_numbers.py                # 打印对比表
  python eval/compare_to_numbers.py --threshold 0.5
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parents[1]
METRIC_DISC = ROOT / "outputs" / "metrics" / "disc"
METRIC_LLM = ROOT / "outputs" / "metrics" / "llm"

# 历史值（来自 _NUMBERS.md，2026-11-16 实测）：method -> (trigger, argument, event)
OLD = {
    "rule":                 (0.6168, 0.1740, 0.0161),
    "bilstm_crf":           (0.8165, 0.4634, 0.0885),
    "bert_linear_256":      (0.8304, 0.5625, 0.1564),
    "bert_crf_256":         (0.8320, 0.5815, 0.1697),
    "bert_linear_128":      (0.8209, 0.5254, 0.1453),
    "bert4L_linear_256":    (0.7793, 0.3495, 0.0535),
    "bert_crf_256_hard":    (0.8464, 0.6566, 0.2177),
    "pipeline":             (0.6769, 0.3690, 0.0957),
    "api_zero_n200":        (0.7789, 0.5246, 0.0402),
    "api_few_n200":         (0.8291, 0.5347, 0.0905),
    "sft_lora_n200":       (0.9100, 0.6966, 0.2050),   # LoRA SFT 的 n=200 结果文件就叫这个名字
}

# 历史值里这些是 n=200 口径；其余是 dev 930 全量口径
N200 = {"api_zero_n200", "api_few_n200", "sft_lora_n200"}


def load_pred(method: str):
    for d in (METRIC_DISC, METRIC_LLM):
        p = d / f"{method}.json"
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    # LoRA 评测脚本输出的 tag 可能带后缀
    for p in sorted(METRIC_LLM.glob("*.json")):
        if method in p.stem:
            return json.loads(p.read_text(encoding="utf-8"))
    return None


def main():
    ap = argparse.ArgumentParser(description="本次重跑 vs _NUMBERS.md 历史值")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="超过多少个百分点标 ⚠（默认 0.5pp）")
    args = ap.parse_args()

    print(f"{'方法':<22}{'口径':<8}{'trigger':>18}{'argument':>18}{'event':>18}")
    print("-" * 86)
    flagged = []
    for method, (ot, oa, oe) in OLD.items():
        r = load_pred(method)
        if r is None:
            print(f"{method:<22}{'(未跑)':<8}")
            continue
        scope = "n=200" if method in N200 else "dev930"
        cur = (r["trigger"]["f1"], r["argument"]["f1"], r["event_exact"]["f1"])
        cells = []
        for o, c in zip((ot, oa, oe), cur):
            d = (c - o) * 100
            mark = " ⚠" if abs(d) > args.threshold else ""
            cells.append(f"{o:.4f}→{c:.4f} ({d:+.2f}pp){mark}")
            if abs(d) > args.threshold:
                flagged.append((method, scope, d))
        print(f"{method:<22}{scope:<8}" + "".join(f"{c:>18}" for c in cells))

    print("-" * 86)
    if flagged:
        print(f"\n⚠ 超出 ±{args.threshold}pp 的项（需人工确认是否是 n=200 的采样噪声）：")
        for m, s, d in flagged:
            print(f"  - {m}（{s}）：{d:+.2f}pp")
    else:
        print(f"\n✅ 没有任何一项偏差超过 ±{args.threshold}pp —— 重构未改变数值行为。")
    print("\n注：n=200 口径下 1 条样本 ≈ 0.5pp，所以该口径的 ±0.5pp 波动属采样噪声。")


if __name__ == "__main__":
    main()
