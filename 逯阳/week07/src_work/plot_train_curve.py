"""
训练曲线（人民日报 NER，src_work/ 专用）

读 outputs_work/logs/work_train_{linear,crf}.json，画 5 个 epoch 的
  val_entity_f1 / train_loss / val_loss 三条曲线，
并在第 3 轮处画一条竖线，标出「前 3 轮从头训练」与「后 2 轮续训」的分界，
方便一眼看出多训 2 轮到底还有没有收益。

输出：outputs_work/figures/work_train_curve.png

使用方式：
  python src_work/plot_train_curve.py
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 中文字体（和主项目其它图保持一致）
matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
matplotlib.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).parent.parent
LOG_DIR = ROOT / "outputs_work" / "logs"
FIG_DIR = ROOT / "outputs_work" / "figures"

RUNS = [
    {"tag": "linear", "label": "BERT + Linear", "color": "#4C72B0"},
    {"tag": "crf",    "label": "BERT + CRF",    "color": "#DD8452"},
]
RESUME_EPOCH = 3   # 前 3 轮从头训练，第 4、5 轮是续训


def main():
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))

    for run in RUNS:
        path = LOG_DIR / f"work_train_{run['tag']}.json"
        if not path.exists():
            print(f"[跳过] 找不到 {path}")
            continue
        recs = json.loads(path.read_text(encoding="utf-8"))
        epochs = [r["epoch"] for r in recs]

        axes[0].plot(epochs, [r["val_entity_f1"] for r in recs], "o-",
                     color=run["color"], label=run["label"], linewidth=2, markersize=6)
        axes[1].plot(epochs, [r["train_loss"] for r in recs], "o-",
                     color=run["color"], label=f"{run['label']} train", linewidth=2, markersize=6)
        axes[1].plot(epochs, [r["val_loss"] for r in recs], "s--",
                     color=run["color"], alpha=0.55,
                     label=f"{run['label']} val", linewidth=1.6, markersize=5)

        # 末轮数值直接标在点上，省得看图估
        last = recs[-1]
        axes[0].annotate(f"{last['val_entity_f1']:.4f}",
                         (last["epoch"], last["val_entity_f1"]),
                         textcoords="offset points", xytext=(6, -12),
                         fontsize=9, color=run["color"])

    # 分界线：左边是从头训练的 3 轮，右边是续训的 2 轮
    for ax in axes:
        ax.axvline(RESUME_EPOCH + 0.5, color="gray", linestyle=":", linewidth=1.4)
        # 用轴坐标（0~1）定位，不要用数据坐标：log 轴的 ylim 下界可能是 1e-5 量级，
        # 文字放那里会被 bbox_inches="tight" 算进画布，图片高度直接爆到十几万像素
        ax.text(RESUME_EPOCH + 0.58, 0.93, "续训起点", transform=ax.get_xaxis_transform(),
                fontsize=8.5, color="gray", va="top")
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.25)
        ax.set_axisbelow(True)
        ax.set_xticks(range(1, 6))

    axes[0].set_ylabel("验证集 Entity-level F1")
    axes[0].set_title("验证集 F1 随 epoch 变化")
    axes[0].legend(loc="lower right")
    axes[0].set_ylim(0.92, 0.965)

    axes[1].set_ylabel("Loss")
    axes[1].set_title("训练 / 验证 Loss 随 epoch 变化")
    axes[1].legend(fontsize=8.5)
    # CRF 的 loss 是负对数似然（含转移分数），量级和线性头的交叉熵不可比，
    # 所以纵轴用对数刻度，否则线性头那条会被压成贴地的一条线
    axes[1].set_yscale("log")
    axes[1].set_ylim(0.002, 20)

    fig.suptitle("人民日报 NER 训练曲线：前 3 轮从头训练，第 4~5 轮为续训", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "work_train_curve.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"训练曲线 → {out}")

    # 终端顺带把表打出来
    print(f"\n{'模型':<14}{'epoch':>7}{'train_loss':>12}{'val_loss':>10}{'val_f1':>9}{'耗时s':>8}")
    print("-" * 62)
    for run in RUNS:
        path = LOG_DIR / f"work_train_{run['tag']}.json"
        if not path.exists():
            continue
        for r in json.loads(path.read_text(encoding="utf-8")):
            print(f"{run['label']:<14}{r['epoch']:>7}{r['train_loss']:>12.4f}"
                  f"{r['val_loss']:>10.4f}{r['val_entity_f1']:>9.4f}{r['elapsed_s']:>8.0f}")
        print("-" * 62)


if __name__ == "__main__":
    main()
