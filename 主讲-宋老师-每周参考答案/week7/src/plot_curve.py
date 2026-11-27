# -*- coding: utf-8 -*-
"""
画训练曲线：直接读 outputs/logs/<run_name>_steps.jsonl 和 <run_name>_train.json。

为什么要单独画：
  steps.jsonl 是给人看的中间产物，但几十行数字不如一张图直观。
  学生交作业时附上 loss 曲线，能一眼看出「有没有收敛、有没有 NaN、第几个 epoch 最好」。
  图中标签用英文，避免 Windows 上 matplotlib 默认字体没有中文字形、显示成方块。

运行：
  python src/plot_curve.py --run_name bert_bio_128
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json

from paths import FIG_DIR, LOG_DIR


def main():
    args = parse_args()
    steps_path = LOG_DIR / f"{args.run_name}_steps.jsonl"
    train_path = LOG_DIR / f"{args.run_name}_train.json"

    if not steps_path.exists():
        raise FileNotFoundError(f"找不到训练曲线日志：{steps_path}（先跑 train.py）")

    steps = []
    with open(steps_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                steps.append(json.loads(line))

    with open(train_path, "r", encoding="utf-8") as f:
        train_doc = json.load(f)

    import matplotlib
    matplotlib.use("Agg")          # 无界面后端：服务器/Windows 后台都能存图
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIG_DIR / f"{args.run_name}_loss_curve.png"

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    # 左图：逐步 loss。x 轴用「全局 step」，epoch 边界画竖虚线，方便看每轮从哪开始
    ax = axes[0]
    xs = list(range(len(steps)))
    losses = [s["loss"] for s in steps]
    ax.plot(xs, losses, color="#2f6fed", linewidth=1.6, label="train loss (logged)")
    ax.set_xlabel("logged step index")
    ax.set_ylabel("loss")
    ax.set_title(f"Training loss - {args.run_name}")
    ax.grid(alpha=0.3)
    ax.legend()

    # 右图：每个 epoch 的验证 F1。训练日志里 epochs[i]["val"]["f1"]
    ax = axes[1]
    epochs = [e["epoch"] for e in train_doc.get("epochs", []) if "val" in e]
    val_f1 = [e["val"]["f1"] for e in train_doc.get("epochs", []) if "val" in e]
    train_loss = [e["train_loss"] for e in train_doc.get("epochs", []) if "val" in e]
    ax.plot(epochs, val_f1, marker="o", color="#e0582b", linewidth=2, label="val entity F1")
    ax.set_xlabel("epoch")
    ax.set_ylabel("entity-level F1")
    ax.set_title("Validation F1 by epoch")
    ax.set_ylim(0, 1.0)
    ax.grid(alpha=0.3)
    for e, f1 in zip(epochs, val_f1):
        ax.annotate(f"{f1:.4f}", (e, f1), textcoords="offset points", xytext=(0, 7),
                    ha="center", fontsize=9)

    # 用 train loss 的坐标轴画第二条线会量纲不一致，这里只打印到标题附近
    if train_loss:
        fig.suptitle(f"final train loss: {train_loss[-1]:.4f}", fontsize=10, y=0.98)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] 训练曲线已保存 → {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="画 BERT NER 训练曲线")
    parser.add_argument("--run_name", type=str, default="bert_bio_128")
    return parser.parse_args()


if __name__ == "__main__":
    main()
