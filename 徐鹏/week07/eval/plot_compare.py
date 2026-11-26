# -*- coding: utf-8 -*-
"""
把 LLM 链路各方案的指标画成一张对比柱状图（PPT 用）

为什么单独一个画图脚本：
  数字散落在 outputs/metrics/llm/*.json 里，人肉抄到 PPT 极易抄错、也容易"取两个口径不同的数"。
  这里只认统一评测脚本产出的 json（trigger/argument/event 三级 F1），
  按文件名里的方法名排序画图，并打印同样内容的 markdown 表（可粘进文档）。

用法：
  python eval/plot_compare.py                      # 自动扫描全部 LLM 实验
  python eval/plot_compare.py --n 200              # 只收 200 条那批（保证同批可比）
  python eval/plot_compare.py --extra_globs "outputs/metrics/*.json"   # 加上判别式结果

输出：
  outputs/llm/method_compare.png   （柱状图，英文标签，避免 matplotlib 中文方块）
  outputs/llm/method_compare.csv   （同一份数字的 CSV，方便贴 PPT/表格）
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")            # 无界面后端：只存图不弹窗（服务器/后台任务必需）
import matplotlib.pyplot as plt  # noqa: E402
import sys

ROOT = Path(__file__).resolve().parents[1]   # 项目根目录（方法目录的上一级）
sys.path.insert(0, str(ROOT / "common"))     # 公共模块：event_data / crf / eval_utils / models / log_utils

# 方法名 → 图例名（按"讲课顺序"排：zero-shot → few-shot → 本地 → SFT）
METHOD_ORDER = [
    ("api_zero",  "API zero-shot"),
    ("api_few",   "API few-shot"),
    ("local_zero", "Qwen3-0.6B zero-shot"),
    ("sft_lora",  "LoRA SFT (Qwen3-0.6B)"),
    ("sft_full",  "Full SFT (Qwen3-0.6B)"),
]

def display_name(tag: str) -> str:
    """把文件名 tag 映射成图上的大类名；未知的（其它方案）原样显示。"""
    for key, name in METHOD_ORDER:
        if tag.startswith(key):
            return name
    return tag

def order_key(tag: str):
    """按 METHOD_ORDER 排序，未知方案排在最后（保持稳定）。"""
    for i, (key, _) in enumerate(METHOD_ORDER):
        if tag.startswith(key):
            return (i, tag)
    return (len(METHOD_ORDER), tag)

def load_metrics(paths, n_filter: int = 0):
    """读统一评测产出的 json，返回 [(tag, trigger_f1, argument_f1, event_f1, n_samples)]。"""
    rows = []
    for path in paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                m = json.load(f)
            n_samples = m.get("num_samples", 0)
            if n_filter and n_samples != n_filter:
                continue
            rows.append((path.stem, m["trigger"]["f1"], m["argument"]["f1"],
                         m["event_exact"]["f1"], n_samples))
        except (json.JSONDecodeError, KeyError) as exc:
            print(f"[跳过] {path.name}：{exc}")
    rows.sort(key=lambda r: order_key(r[0]))
    return rows

def main():
    ap = argparse.ArgumentParser(description="LLM 事件抽取方法对比图")
    ap.add_argument("--metrics_dir", default=str(ROOT / "outputs" / "metrics" / "llm"))
    ap.add_argument("--extra_globs", default="", help="额外要纳入的 json（逗号分隔 glob）")
    ap.add_argument("--n", default=0, type=int, help="只收 num_samples==n 的实验（0=不过滤）")
    ap.add_argument("--skip", default="probe,smoke,mock,dry",
                    help="文件名含这些关键字的实验不纳入（冒烟/假数据）")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "llm" / "method_compare.png"))
    ap.add_argument("--title", default="Event extraction on DuEE-Fin dev (unified evaluation)")
    args = ap.parse_args()

    skip = [s for s in args.skip.split(",") if s]
    paths = [p for p in sorted(Path(args.metrics_dir).glob("*.json"))
             if not any(k in p.stem for k in skip)]
    for pattern in [g for g in args.extra_globs.split(",") if g]:
        paths += [p for p in sorted(ROOT.glob(pattern))
                  if not any(k in p.stem for k in skip) and p not in paths]

    rows = load_metrics(paths, args.n)
    if not rows:
        print("没有可画的实验（先跑 llm_extract.py / evaluate_sft.py 生成 metrics json）")
        return

    n_show = rows[0][4] if len({r[4] for r in rows}) == 1 else "mixed"
    labels = [display_name(r[0]) for r in rows]
    trig = [r[1] for r in rows]
    arg = [r[2] for r in rows]
    ev = [r[3] for r in rows]

    # ── 画图：每个方法 3 根柱（trigger / argument / event）────────────────────
    import numpy as np
    x = np.arange(len(labels))
    width = 0.26
    fig, ax = plt.subplots(figsize=(max(7.5, 2.1 * len(labels)), 4.4), dpi=150)
    b1 = ax.bar(x - width, trig, width, label="trigger F1", color="#4C72B0")
    b2 = ax.bar(x, arg, width, label="argument F1", color="#DD8452")
    b3 = ax.bar(x + width, ev, width, label="event F1", color="#55A868")
    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.3f}", (bar.get_x() + bar.get_width() / 2, h),
                        textcoords="offset points", xytext=(0, 2),
                        ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, rotation=12)
    ax.set_ylabel("F1 (higher is better)")
    ax.set_ylim(0, 1.0)
    ax.set_title(f"{args.title}\n(n={n_show}, seed=42, same sample set)", fontsize=10)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    print(f"图 → {out_path}")

    # ── CSV + markdown 表（同一份数字，贴 PPT / 文档用）──────────────────────
    csv_path = out_path.with_suffix(".csv")
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "trigger_f1", "argument_f1", "event_f1", "n_samples"])
        w.writerows(rows)
    print(f"CSV → {csv_path}\n")
    print(f"| 方法 | trigger F1 | argument F1 | event F1 | 样本数 |")
    print(f"|---|---|---|---|---|")
    for tag, t, a, e, n in rows:
        print(f"| {display_name(tag)} | {t:.4f} | {a:.4f} | {e:.4f} | {n} |")

if __name__ == "__main__":
    main()
