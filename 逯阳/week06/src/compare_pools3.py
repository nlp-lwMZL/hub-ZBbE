"""
三种池化策略同框对比（cls / max / mean，均为不加权）

和另外两个对比脚本的分工：
  compare_class_weight.py —— 固定一个 pool，比较「普通 loss vs 加权 loss」
  compare_pool.py         —— 两个 pool 两两对比（出 A vs B 的图）
  compare_pools3.py       —— 三个 pool 一起画在一张图上

使用方式：
  python src/compare_pools3.py                          # 默认 cls, max, mean
  python src/compare_pools3.py --pools cls,mean,max     # 换顺序
  python src/compare_pools3.py --pools cls,mean         # 两个也行

输出：
  终端：整体指标表 / 逐类 F1 表（标出每类最强的 pool）
  图片：outputs/figures/compare_pools_{顺序}_unweighted.png
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

# 推理函数直接复用 compare_pool.py（同为 MAX_LENGTH=128 / BATCH_SIZE=32），避免重复实现
from compare_pool import get_predictions

matplotlib.rcParams["axes.unicode_minus"] = False


def _find_chinese_font():
    candidates = ["SimHei", "Microsoft YaHei", "PingFang SC",
                  "Noto Sans CJK SC", "WenQuanYi Micro Hei"]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            return name
    return None


_CN_FONT = _find_chinese_font()
if _CN_FONT:
    plt.rcParams["font.family"] = _CN_FONT
    # 换字体后重新关掉 Unicode 负号，SimHei 没有 U+2212 字形
    plt.rcParams["axes.unicode_minus"] = False

ROOT      = Path(__file__).parent.parent
DATA_DIR  = ROOT / "data"
BERT_PATH = ROOT.parent.parent / "pretrain_models" / "bert-base-chinese"
CKPT_DIR  = ROOT / "outputs" / "checkpoints"
FIG_DIR   = ROOT / "outputs" / "figures"

# 三种池化固定用这三个颜色，cls 蓝 / max 橙 / mean 绿
COLORS = {"cls": "#4C72B0", "max": "#DD8452", "mean": "#55A868"}


def main():
    parser = argparse.ArgumentParser(description="三种池化策略同框对比")
    parser.add_argument("--pools",      default="cls,max,mean",
                        help="逗号分隔，可选 cls / mean / max")
    parser.add_argument("--bert_path",  default=str(BERT_PATH))
    parser.add_argument("--data_dir",   default=str(DATA_DIR))
    args = parser.parse_args()

    pools = [p.strip() for p in args.pools.split(",") if p.strip()]
    if len(pools) < 2:
        print("[错误] 至少需要两个 pool")
        return

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data_dir)

    # ── 检查权重是否齐全 ────────────────────────────────────────────────────
    missing = [p for p in pools if not (CKPT_DIR / f"best_{p}.pt").exists()]
    if missing:
        for p in missing:
            print(f"[缺少] {CKPT_DIR / f'best_{p}.pt'}")
            print(f"       请先训练：python src/train.py --pool {p}")
        return

    with open(data_dir / "label_map.json", encoding="utf-8") as f:
        id2name = {int(k): v for k, v in json.load(f)["id2name"].items()}
    label_ids   = sorted(id2name.keys())
    class_names = [id2name[i] for i in label_ids]

    # ── 逐个推理 ────────────────────────────────────────────────────────────
    preds, metas = {}, {}
    for p in pools:
        ckpt_path = CKPT_DIR / f"best_{p}.pt"
        print(f"\n加载 {p:<4}：{ckpt_path.name}")
        pr, labels, meta = get_predictions(ckpt_path, data_dir, args.bert_path, device)
        preds[p] = pr
        metas[p] = meta
        print(f"  → epoch={meta['epoch']}, val_acc={meta['val_acc']:.4f}, "
              f"use_class_weight={meta['use_class_weight']}")

    if any(m["use_class_weight"] for m in metas.values()):
        print("\n[注意] 有模型用了加权 loss，本次对比混入了 loss 的影响，不纯是池化差异")

    # ── 指标 ────────────────────────────────────────────────────────────────
    acc  = {p: accuracy_score(labels, preds[p])                       for p in pools}
    mf1  = {p: f1_score(labels, preds[p], average="macro",    zero_division=0) for p in pools}
    wf1  = {p: f1_score(labels, preds[p], average="weighted", zero_division=0) for p in pools}
    f1c  = {p: precision_recall_fscore_support(
                labels, preds[p], labels=label_ids, zero_division=0)[2] for p in pools}
    # 样本数只跟真标签有关，和哪个 pool 无关，算一次就够
    support = precision_recall_fscore_support(
        labels, preds[pools[0]], labels=label_ids, zero_division=0)[3]

    print(f"\n{'='*64}")
    print(f"{'整体指标（val 10000，均不加权）':^58}")
    print(f"{'='*64}")
    header = f"{'指标':<14}" + "".join(f"{p:>12}" for p in pools)
    print(header)
    print("-" * len(header))
    for name, d in [("Accuracy", acc), ("Macro F1", mf1), ("Weighted F1", wf1)]:
        best = max(d, key=d.get)
        row = f"  {name:<12}"
        for p in pools:
            row += f"{d[p]:>11.4f}" + ("*" if p == best else " ")
        print(row)
    print("  （* 表示该指标最优）")

    # ── 逐类 F1 ────────────────────────────────────────────────────────────
    print(f"\n{'逐类 F1（★ 标每类最强）':^58}")
    head2 = f"{'类别':<5}{'样本':>6}" + "".join(f"{p:>9}" for p in pools)
    print(head2)
    print("-" * len(head2))
    best_count = {p: 0 for p in pools}
    for i, name in enumerate(class_names):
        vals = {p: f1c[p][i] for p in pools}
        best = max(vals, key=vals.get)
        best_count[best] += 1
        row = f"  {name:<4}{int(support[i]):>6}"
        for p in pools:
            row += f"{vals[p]:>8.3f}" + ("★" if p == best else " ")
        print(row)
    print("\n  各类夺冠次数：" + "，".join(f"{p} {best_count[p]} 类" for p in pools))

    # ── 画图 ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(17, 5.5),
                             gridspec_kw={"width_ratios": [2.4, 1]})

    # 左：各类别 F1，三色并排
    x = np.arange(len(class_names))
    w = 0.8 / len(pools)
    for k, p in enumerate(pools):
        offset = (k - (len(pools) - 1) / 2) * w
        axes[0].bar(x + offset, f1c[p], w, label=p,
                    color=COLORS.get(p, "#888888"), alpha=0.88)
    axes[0].set_title("各类别 F1 对比")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(class_names, rotation=40, ha="right")
    axes[0].set_ylabel("F1")
    axes[0].set_ylim(0, 0.85)
    axes[0].axhline(0.5, color="gray", linestyle="--", alpha=0.4)
    axes[0].legend(title="池化")

    # 右：整体指标，三项 × 三池化
    metrics = [("Accuracy", acc), ("Macro F1", mf1), ("Weighted F1", wf1)]
    xm = np.arange(len(metrics))
    for k, p in enumerate(pools):
        offset = (k - (len(pools) - 1) / 2) * w
        vals = [d[p] for _, d in metrics]
        bars = axes[1].bar(xm + offset, vals, w, label=p,
                           color=COLORS.get(p, "#888888"), alpha=0.88)
        for b, v in zip(bars, vals):
            axes[1].text(b.get_x() + b.get_width() / 2, v + 0.004,
                         f"{v:.4f}", ha="center", va="bottom", fontsize=7, rotation=90)
    axes[1].set_title("整体指标")
    axes[1].set_xticks(xm)
    axes[1].set_xticklabels([n for n, _ in metrics])
    axes[1].set_ylim(0, 0.70)
    axes[1].grid(axis="y", alpha=0.25)
    # 右栏不放图例：颜色和左栏一致，放了会压住柱子
    axes[1].set_axisbelow(True)

    plt.suptitle("池化策略对比：" + " / ".join(pools) + "（均不加权，val set）", fontsize=13)
    plt.tight_layout()

    fig_name = f"compare_pools_{'_'.join(pools)}_unweighted.png"
    save_path = FIG_DIR / fig_name
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\n对比图已保存 → {save_path}")


if __name__ == "__main__":
    main()
