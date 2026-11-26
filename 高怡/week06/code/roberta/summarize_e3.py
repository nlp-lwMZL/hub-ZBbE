"""
汇总 4 层 3 epoch 消融实验结果：生成汇总 CSV + 对比图。

数据来源
--------
output/roberta/roberta_train_log_s42_*_cls4_e3.json   逐 epoch 训练日志
output/roberta/test_metrics_s42_*_cls4_e3.json        测试集指标

输出
----
output/roberta/ablation_3ep_summary.csv
output/roberta/ablation_3ep_curves.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "output" / "roberta"

NAMES = {
    "s42_ce_cls4_e3": "基线 CE",
    "s42_ce_ls01_cls4_e3": "LS 0.1",
    "s42_poolmax_cls4_e3": "pool=max",
    "s42_nocw_cls4_e3": "无类别加权",
    "s42_focal_cls4_e3": "Focal γ=2",
    "s42_fgm_cls4_e3": "FGM ε=1.0",
    "s42_fgm05_cls4_e3": "FGM ε=0.5",
}

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def main() -> None:
    rows = []
    for tag, name in NAMES.items():
        log_path = OUT_DIR / f"roberta_train_log_{tag}.json"
        test_path = OUT_DIR / f"test_metrics_{tag}.json"
        if not log_path.exists() or not test_path.exists():
            print(f"[skip] {tag} 缺文件")
            continue
        log = json.loads(log_path.read_text(encoding="utf-8"))
        test = json.loads(test_path.read_text(encoding="utf-8"))
        hist = log.get("history") or log.get("epochs") or []
        vals = [h["val_macro_f1"] for h in hist]
        trains = [h["train_macro_f1"] for h in hist]
        rows.append({
            "tag": tag,
            "实验": name,
            "ep1_train": trains[0], "ep1_val": vals[0],
            "ep2_train": trains[1], "ep2_val": vals[1],
            "ep3_train": trains[2], "ep3_val": vals[2],
            "best_val": max(vals),
            "test_acc": test["test_acc"],
            "test_macro_f1": test["test_macro_f1"],
            "test_err_cnt": round(2037 * (1 - test["test_acc"])),
        })

    df = pd.DataFrame(rows).sort_values("test_macro_f1", ascending=False)
    df.to_csv(OUT_DIR / "ablation_3ep_summary.csv", index=False, encoding="utf-8-sig")
    print(df[["实验", "best_val", "test_acc", "test_macro_f1", "test_err_cnt"]]
          .to_string(index=False))

    # 左图：逐 epoch val 曲线；右图：测试集 Macro F1 排名
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 4.8))

    ax = axes[0]
    for r in rows:
        ax.plot([1, 2, 3], [r["ep1_val"], r["ep2_val"], r["ep3_val"]],
                marker="o", markersize=5, linewidth=1.6, label=r["实验"])
    ax.set_xticks([1, 2, 3])
    ax.set_xlabel("epoch")
    ax.set_ylabel("Val Macro F1")
    ax.set_title("逐 epoch 验证集曲线（4 层 / seed=42）")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2, loc="lower right")

    ax = axes[1]
    d2 = df.sort_values("test_macro_f1")
    bars = ax.barh(d2["实验"], d2["test_macro_f1"], color="#4C78A8", height=0.62)
    lo = d2["test_macro_f1"].min()
    ax.set_xlim(lo - 0.0012, 1.0)
    for b, v, e in zip(bars, d2["test_macro_f1"], d2["test_err_cnt"]):
        ax.text(v + 0.00008, b.get_y() + b.get_height() / 2,
                f"{v:.4f}（错 {e} 条）", va="center", fontsize=8)
    ax.set_xlabel("Test Macro F1")
    ax.set_title("测试集 Macro F1 排名（横轴已放大）")
    ax.grid(alpha=0.25, axis="x")

    fig.tight_layout()
    out = OUT_DIR / "ablation_3ep_curves.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[图] {out}")

    plot_val_test(rows)


def plot_val_test(rows: list[dict]) -> None:
    """val 逐 epoch 曲线 + test 单点，并附 val→test 斜率图。

    测试集只在"最优 val 的 ckpt"上评估过一次，因此 test 只有一个点，
    画在该配置最优 val 所在的 epoch 位置上（星标）。
    """
    colors = plt.cm.tab10.colors

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2),
                             gridspec_kw={"width_ratios": [1.45, 1]})

    # ---- 左：val 曲线 + test 星标 ----
    ax = axes[0]
    for i, r in enumerate(rows):
        vals = [r["ep1_val"], r["ep2_val"], r["ep3_val"]]
        c = colors[i % 10]
        ax.plot([1, 2, 3], vals, marker="o", markersize=5, linewidth=1.7,
                color=c, label=r["实验"])
        best_ep = vals.index(max(vals)) + 1
        ax.scatter([best_ep], [r["test_macro_f1"]], marker="*", s=170,
                   color=c, edgecolor="black", linewidth=0.8, zorder=6)

    ax.set_xticks([1, 2, 3])
    ax.set_xlabel("epoch")
    ax.set_ylabel("Macro F1")
    ax.set_title("val 逐 epoch 曲线（实线圆点） + test 结果（星标）\n"
                 "星标横坐标为该配置最优 val 所在 epoch", fontsize=11)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2, loc="lower left")

    lo = min(min(r["ep1_val"], r["ep2_val"], r["ep3_val"]) for r in rows)
    hi = max(max(r["ep1_val"], r["ep2_val"], r["ep3_val"]) for r in rows)
    pad = (hi - lo) * 0.15 + 0.0003
    ax.set_ylim(lo - pad, hi + pad)

    # ---- 右：val → test 斜率图 ----
    ax = axes[1]
    for i, r in enumerate(rows):
        c = colors[i % 10]
        ax.plot([0, 1], [r["best_val"], r["test_macro_f1"]],
                marker="o", markersize=6, linewidth=1.7, color=c, label=r["实验"])
        ax.annotate(f"{r['best_val']:.4f}", xy=(0, r["best_val"]),
                    xytext=(-8, 0), textcoords="offset points",
                    ha="right", va="center", fontsize=8, color=c)
        ax.annotate(f"{r['test_macro_f1']:.4f}", xy=(1, r["test_macro_f1"]),
                    xytext=(8, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=8, color=c)

    ax.set_xlim(-0.42, 1.42)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["最优 val", "test"])
    ax.set_ylabel("Macro F1")
    ax.set_title("验证集 → 测试集 的排序变化\n（线条交叉即说明 val 排序不可靠）", fontsize=11)
    ax.grid(alpha=0.25, axis="y")
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    out = OUT_DIR / "ablation_3ep_val_vs_test.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[图] {out}")


if __name__ == "__main__":
    main()
