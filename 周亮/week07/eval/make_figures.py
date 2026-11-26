#!/usr/bin/env python3
"""
生成 PPT 可直接贴的图（老师 P41/P42 那套可视化）。

从现有产物离线生成，不重跑模型：
  1. fig_entity_distribution.png  实体类别分布（train/val/test）
  2. fig_sent_length.png          句子长度分布 + 截断线
  3. fig_method_compare.png       六臂 F1 对比（整体 + type_macro）
  4. fig_per_type_f1.png          各类 F1 雷达/分组柱状（长尾定位）
  5. fig_error_breakdown.png      错误分解堆叠图（类型错/边界错/漏召/多预测）
  6. fig_training_curves.png      六臂训练曲线

用法：python eval/make_figures.py
产物：outputs/figures/*.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))
from common import paths

# 中文字体：macOS PingFang，其他系统退回默认（会缺字但不报错）
plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

METHODS = ["bilstm_linear", "bilstm_crf", "roberta_linear", "roberta_crf",
           "qwen_lora_sft", "qwen_full_sft"]
LABELS = {"bilstm_linear": "BiLSTM-Linear", "bilstm_crf": "BiLSTM+CRF",
          "roberta_linear": "RoBERTa-Linear", "roberta_crf": "RoBERTa+CRF",
          "qwen_lora_sft": "Qwen3-LoRA", "qwen_full_sft": "Qwen3-full"}
COLORS = {"bilstm_linear": "#B4B2A9", "bilstm_crf": "#888780",
          "roberta_linear": "#85B7EB", "roberta_crf": "#378ADD",
          "qwen_lora_sft": "#5DCAA5", "qwen_full_sft": "#1D9E75"}


def load_json(p: Path) -> dict | None:
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def main() -> None:
    out_dir = paths.get_out_root() / "outputs" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_dir = paths.eval_dir()
    data_stats = load_json(eval_dir / "data_stats.json")
    errors = load_json(eval_dir / "errors_stats.json")

    # 1) 实体类别分布
    if data_stats:
        types = list(data_stats["splits"]["train"]["entity_types"].keys())
        fig, ax = plt.subplots(figsize=(8, 4.5))
        x = np.arange(len(types))
        w = 0.25
        for i, (s, color) in enumerate(zip(("train", "validation", "test"),
                                           ("#378ADD", "#1D9E75", "#BA7517"))):
            vals = [data_stats["splits"][s]["entity_types"].get(t, 0) for t in types]
            bars = ax.bar(x + (i - 1) * w, vals, w, label=s, color=color)
            ax.bar_label(bars, fmt="%d", fontsize=9)
        ax.set_xticks(x); ax.set_xticklabels(types)
        ax.set_ylabel("实体数"); ax.set_title("实体类别分布（5 类，14.8× 不均衡）")
        ax.legend(); ax.grid(axis="y", alpha=0.3)
        fig.tight_layout(); fig.savefig(out_dir / "fig_entity_distribution.png", dpi=150); plt.close(fig)

        # 2) 句长分布 + 截断线
        fig, ax = plt.subplots(figsize=(8, 4.5))
        # 用分位数条形代替直方图（stats 里没存原始长度）
        for s, color in zip(("train", "validation", "test"), ("#378ADD", "#1D9E75", "#BA7517")):
            d = data_stats["splits"][s]["sent_len"]
            pts = [d["p50"], d["p90"], d["p95"], d["p99"], d["max"]]
            ax.plot(["p50", "p90", "p95", "p99", "max"], pts, marker="o", label=s, color=color)
        ax.axhline(256, color="#A32D2D", linestyle="--", label="max_len=256（截断线）")
        ax.set_yscale("log"); ax.set_ylabel("token 数（log）"); ax.set_title("句子长度分位数与截断线")
        ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(out_dir / "fig_sent_length.png", dpi=150); plt.close(fig)

    # 3) 六臂 F1 对比
    metrics = {}
    for m in METHODS:
        d = load_json(paths.eval_dir() / f"{m}_test.json")
        if d:
            metrics[m] = d["metrics"]
    if metrics:
        ms = [m for m in METHODS if m in metrics]
        f1 = [metrics[m]["f1"] for m in ms]
        macro = [metrics[m]["type_macro_f1"] for m in ms]
        fig, ax = plt.subplots(figsize=(9, 4.5))
        x = np.arange(len(ms)); w = 0.35
        b1 = ax.bar(x - w/2, f1, w, label="实体级 F1", color="#378ADD")
        b2 = ax.bar(x + w/2, macro, w, label="type_macro F1", color="#BA7517")
        ax.bar_label(b1, fmt="%.3f", fontsize=9); ax.bar_label(b2, fmt="%.3f", fontsize=9)
        ax.set_xticks(x); ax.set_xticklabels([LABELS[m] for m in ms], rotation=15)
        ax.set_ylim(0, 0.85); ax.set_title("六臂实体级 F1 对比（cdtier clean test 387 句，同一评测口径）")
        ax.legend(); ax.grid(axis="y", alpha=0.3)
        fig.tight_layout(); fig.savefig(out_dir / "fig_method_compare.png", dpi=150); plt.close(fig)

        # 4) per-type F1（取最优臂 full + 基线 crf）
        types = sorted(set().union(*[
            set(metrics.get(m, {}).get("per_type", {}).keys()) for m in ("roberta_crf", "qwen_full_sft")
        ]))
        if types:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            x = np.arange(len(types)); w = 0.35
            for i, (m, color) in enumerate((("roberta_crf", "#378ADD"),
                                            ("qwen_full_sft", "#1D9E75"))):
                vals = [metrics.get(m, {}).get("per_type", {}).get(t, {}).get("f1", 0) for t in types]
                bars = ax.bar(x + (i - 0.5) * w, vals, w, label=LABELS[m], color=color)
                ax.bar_label(bars, fmt="%.2f", fontsize=9)
            ax.set_xticks(x); ax.set_xticklabels(types)
            ax.set_ylabel("F1"); ax.set_title("分类型 F1（长尾定位：Campaign 样本最少）")
            ax.legend(); ax.grid(axis="y", alpha=0.3)
            fig.tight_layout(); fig.savefig(out_dir / "fig_per_type_f1.png", dpi=150); plt.close(fig)

    # 5) 错误分解堆叠图
    if errors:
        ms = [m for m in METHODS if m in errors["methods"]]
        fig, ax = plt.subplots(figsize=(9, 4.5))
        categories = [
            ("tp", "完全正确", "#1D9E75"),
            ("boundary_error", "边界错(类型对)", "#BA7517"),
            ("type_error", "类型错(边界对)", "#D4537E"),
            ("both_error", "都错", "#A32D2D"),
            ("missed", "漏召", "#888780"),
        ]
        bottom = np.zeros(len(ms))
        for key, name, color in categories:
            vals = np.array([errors["methods"][m]["counts"].get(key, 0) for m in ms], dtype=float)
            ax.bar([LABELS[m] for m in ms], vals, bottom=bottom, label=name, color=color)
            bottom += vals
        ax.set_ylabel("gold span 数"); ax.set_title("错误分解：判别式卡在边界，生成式卡在多预测")
        ax.legend(loc="upper left", fontsize=9); ax.grid(axis="y", alpha=0.3)
        plt.xticks(rotation=15)
        fig.tight_layout(); fig.savefig(out_dir / "fig_error_breakdown.png", dpi=150); plt.close(fig)

    # 6) 训练曲线（从 logs/train_*.jsonl）
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for m in METHODS:
        p = paths.log_dir() / f"train_{m}.jsonl"
        if not p.exists():
            continue
        recs = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        epochs = [r.get("epoch", i + 1) for i, r in enumerate(recs)]
        # 各脚本字段名不完全一致：val_entity_f1 或 avg_loss
        key = "val_entity_f1" if any("val_entity_f1" in r for r in recs) else "avg_loss"
        vals = [r.get(key, 0) for r in recs]
        ax.plot(epochs, vals, marker="o", label=f"{LABELS[m]} ({key})",
                color=COLORS[m], alpha=0.85)
    ax.set_xlabel("epoch"); ax.set_ylabel("值（F1 或 loss，量纲不同不可横比）")
    ax.set_title("六臂训练曲线（选模看 val F1，不看 loss）")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_dir / "fig_training_curves.png", dpi=150); plt.close(fig)

    print(f"图表已生成到 {out_dir}:")
    for p in sorted(out_dir.glob("fig_*.png")):
        print(f"  {p.name}  ({p.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
