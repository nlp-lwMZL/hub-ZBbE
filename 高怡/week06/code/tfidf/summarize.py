"""
汇总 tfidf_cls_* 各次实验的 test 指标，生成对比表 + 柱状图。
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

OUT_ROOT = Path(__file__).resolve().parent.parent / "output" / "tfidf"
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def collect() -> pd.DataFrame:
    rows = []
    for d in sorted(OUT_ROOT.glob("tfidf_cls_*")):
        config_path = d / "config.json"
        test_metrics_path = d / "test" / "metrics.json"
        if not config_path.exists() or not test_metrics_path.exists():
            continue
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        tm = json.loads(test_metrics_path.read_text(encoding="utf-8"))
        rows.append({
            "dir": d.name,
            "model": cfg["model"],
            "fusion": cfg.get("fusion", "_fusion" in cfg["model"]),
            "max_features": cfg["max_features"],
            "test_macro_f1": tm["macro_f1"],
            "test_weighted_f1": tm["weighted_f1"],
            "test_acc": tm["accuracy"],
            "fit_time_sec": cfg.get("fit_time_sec", 0),
        })
    return pd.DataFrame(rows).sort_values("test_macro_f1", ascending=False, ignore_index=True)


def main() -> None:
    df = collect()
    if df.empty:
        print("没有找到任何 tfidf_cls_* 实验结果")
        return

    summary_path = OUT_ROOT / "tfidf_summary.csv"
    df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"[summary] {summary_path}")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # 对比柱状图：test macro F1
    fig, ax = plt.subplots(figsize=(9, 4.5))
    labels = df["model"].tolist()
    x = range(len(labels))
    ax.bar(x, df["test_macro_f1"], color="#02743B")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Macro F1 (test)")
    ax.set_title("TF-IDF 多分类基线实验对比")
    for i, v in enumerate(df["test_macro_f1"]):
        ax.text(i, v + 0.0005, f"{v:.4f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(df["test_macro_f1"].min() - 0.01, 1.001)
    fig.tight_layout()
    fig.savefig(OUT_ROOT / "tfidf_summary_macro_f1.png", dpi=150)
    plt.close(fig)
    print(f"[plot] {OUT_ROOT / 'tfidf_summary_macro_f1.png'}")


if __name__ == "__main__":
    main()
