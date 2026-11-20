# -*- coding: utf-8 -*-
"""
把换基座的对比结果画成一张图（作业/汇报用）

只读 outputs/ 下两个 log.json，不训练、不占显存，几秒钟出图：
  outputs/compare_base_models.png

图分左右两栏：
  左：Macro-F1 / Accuracy 两个汇总指标（换基座到底涨了几个点）
  右：15 个类别各自的 F1 变化量（涨的集中在哪几类，掉的又是哪几类）

用法：python plot_compare.py
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402

# 中文字体：Windows 上优先微软雅黑，退到黑体；按本机实测，竖排/旋转的中文会叠字，
# 所以纵轴用「类别名横排 + 数值标注」的横向条形图，不做旋转标签
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 默认读「与本脚本同目录的 outputs/base_models」——与 train_base_model.py 的产物目录一致
OUT_DIR = Path(__file__).resolve().parent / "outputs" / "base_models"
LABELS = ["故事", "文化", "娱乐", "体育", "财经", "房产", "汽车", "教育",
          "科技", "军事", "旅游", "国际", "证券", "农业", "电竞"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="bert-base-chinese", help="基线产物前缀")
    ap.add_argument("--b", default="chinese-roberta-wwm-ext", help="对比基座产物前缀")
    ap.add_argument("--out_dir", default=str(OUT_DIR))
    args = ap.parse_args()

    out = Path(args.out_dir)
    la = json.loads((out / f"{args.a}_log.json").read_text(encoding="utf-8"))
    lb = json.loads((out / f"{args.b}_log.json").read_text(encoding="utf-8"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.6), gridspec_kw={"width_ratios": [1, 1.5]})

    # ── 左：两个汇总指标 ─────────────────────────────────────────────
    metrics = ["Macro-F1", "Accuracy"]
    va = [la["best_macro_f1"], la["best_acc"]]
    vb = [lb["best_macro_f1"], lb["best_acc"]]
    x = range(len(metrics))
    w = 0.36
    c_a, c_b = "#2E86AB", "#D4720A"
    bars_a = ax1.bar([i - w / 2 for i in x], va, w, label=args.a, color=c_a)
    bars_b = ax1.bar([i + w / 2 for i in x], vb, w, label=args.b, color=c_b)
    for bars, vals in ((bars_a, va), (bars_b, vb)):
        for r, v in zip(bars, vals):
            ax1.annotate(f"{v:.4f}", (r.get_x() + r.get_width() / 2, v),
                         textcoords="offset points", xytext=(0, 4), ha="center", fontsize=11)
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(metrics, fontsize=12)
    ax1.set_ylim(min(va + vb) - 0.08, max(va + vb) + 0.06)
    ax1.set_ylabel("分数", fontsize=12)
    ax1.legend(fontsize=10, frameon=False, loc="lower right")
    ax1.set_title("汇总指标（越高越好）", fontsize=13, fontweight="bold")
    ax1.grid(axis="y", alpha=0.25)

    # ── 右：每类 F1 变化量（横向条形，类别名横排不旋转）───────────────
    pa, pb = la["per_class_f1"], lb["per_class_f1"]
    deltas = [(name, pb[name] - pa[name]) for name in LABELS]
    deltas.sort(key=lambda t: t[1])                # 从掉得最多到涨得最多
    names = [d[0] for d in deltas]
    vals = [d[1] for d in deltas]
    colors = ["#C0392B" if v < 0 else c_b for v in vals]
    ax2.barh(names, vals, color=colors, alpha=0.85)
    ax2.axvline(0, color="#333", linewidth=0.8)
    for i, v in enumerate(vals):
        ax2.annotate(f"{v:+.3f}", (v, i), textcoords="offset points",
                     xytext=(6 if v >= 0 else -6, 0), va="center",
                     ha="left" if v >= 0 else "right", fontsize=9.5)
    ax2.set_xlabel(f"Δ F1（{args.b} − {args.a}）", fontsize=12)
    ax2.set_title("各类别 F1 变化（右=变好，左=变差）", fontsize=13, fontweight="bold")
    ax2.grid(axis="x", alpha=0.25)
    ax2.set_xlim(min(vals) + (min(vals) * 0.35 if min(vals) < 0 else -0.02),
                 max(vals) * 1.25 if max(vals) > 0 else 0.02)

    fig.suptitle(f"换基座实验：{args.a} → {args.b}（Macro-F1 {la['best_macro_f1']:.4f} → "
                 f"{lb['best_macro_f1']:.4f}，{lb['best_macro_f1'] - la['best_macro_f1']:+.4f}；"
                 f"Accuracy {la['best_acc']:.4f} → {lb['best_acc']:.4f}）\n"
                 f"同一批 {la['meta']['n_clean_val']:,} 条去重验证样本 · 同超参 · seed {la['meta']['seed']} · "
                 f"两个基座都是 {la['meta']['params_M']}M 参数",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    p = out / "compare_base_models.png"
    fig.savefig(p, dpi=160, bbox_inches="tight", facecolor="white")
    print(f"已保存 → {p}")


if __name__ == "__main__":
    main()
