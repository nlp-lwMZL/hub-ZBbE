"""
池化策略横向对比（同一训练条件下，比较 cls / mean / max）

分工说明（两个脚本都只做推理，都不训练）：
  compare_class_weight.py —— 固定同一个 pool，比较「普通 loss vs 加权 loss」
  compare_pool.py         —— 固定同一个 loss 条件，比较「不同 pool 之间」

使用方式：
  python src/compare_pool.py                            # 默认 cls vs mean（都不加权）
  python src/compare_pool.py --pool_a mean --pool_b cls # 只换顺序
  python src/compare_pool.py --pool_a cls --pool_b max  # 前提是 best_max.pt 存在

输出：
  终端：整体指标表 / 15 类 P-R-F1 对照表 / 逐类 Δ F1
  图片：outputs/figures/compare_pool_{a}_vs_{b}_unweighted.png
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_fscore_support,
)

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
    # 字体换过之后重新关掉 Unicode 负号，否则 SimHei 缺 U+2212 字形会告警
    plt.rcParams["axes.unicode_minus"] = False

ROOT      = Path(__file__).parent.parent
DATA_DIR  = ROOT / "data"
BERT_PATH = ROOT.parent.parent / "pretrain_models" / "bert-base-chinese"
CKPT_DIR  = ROOT / "outputs" / "checkpoints"
FIG_DIR   = ROOT / "outputs" / "figures"

# 推理参数与 evaluate.py / compare_class_weight.py 保持一致，方便横向对照
MAX_LENGTH = 128
BATCH_SIZE = 32     # 比 compare_class_weight 的 64 小，训练占着 GPU 时也能跑（显存约 700MB）


def get_predictions(ckpt_path, data_dir, bert_path, device):
    """加载一个 checkpoint，在验证集上跑一遍，返回预测和真标签。"""
    from model import build_model
    from dataset import build_dataloaders
    from transformers import BertTokenizer

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    pool = ckpt["pool"]

    model = build_model(bert_path, num_labels=15, pool=pool)
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device).eval()

    tokenizer = BertTokenizer.from_pretrained(bert_path)
    _, val_loader, _ = build_dataloaders(
        data_dir, tokenizer, max_length=MAX_LENGTH, batch_size=BATCH_SIZE
    )

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            logits = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["token_type_ids"].to(device),
            )
            all_preds.extend(logits.argmax(-1).cpu().numpy())
            all_labels.extend(batch["label"].numpy())

    return np.array(all_preds), np.array(all_labels), ckpt


def _weight_tag(ck_a, ck_b):
    """根据两个 ckpt 真实的 use_class_weight 决定文件名后缀，避免名字撒谎。"""
    wa, wb = ck_a["use_class_weight"], ck_b["use_class_weight"]
    if not wa and not wb:
        return "unweighted"
    if wa and wb:
        return "weighted"
    return "mixed_weight"      # 一个加权一个不加权，不是本脚本的典型用法


def compare_and_plot(
    preds_a, preds_b, labels, id2name,
    tag_a, tag_b, pool_a, pool_b,
    suffix="unweighted",
    output_dir: Path = FIG_DIR,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    label_ids   = sorted(id2name.keys())
    class_names = [id2name[i] for i in label_ids]

    # ── 整体指标 ────────────────────────────────────────────────────────────
    acc_a = accuracy_score(labels, preds_a)
    acc_b = accuracy_score(labels, preds_b)
    mf1_a = f1_score(labels, preds_a, average="macro",    zero_division=0)
    mf1_b = f1_score(labels, preds_b, average="macro",    zero_division=0)
    wf1_a = f1_score(labels, preds_a, average="weighted", zero_division=0)
    wf1_b = f1_score(labels, preds_b, average="weighted", zero_division=0)

    print(f"\n{'='*66}")
    print(f"{'整体指标对比（' + pool_a + ' vs ' + pool_b + '）':^60}")
    print(f"{'='*66}")
    print(f"{'指标':<16} {tag_a:>16}  {tag_b:>16}  {'差值':>8}")
    print(f"{'-'*66}")
    for name, va, vb in [("Accuracy", acc_a, acc_b),
                         ("Macro F1",  mf1_a, mf1_b),
                         ("Weighted F1", wf1_a, wf1_b)]:
        sign = "+" if vb - va >= 0 else ""
        print(f"  {name:<14} {va:>16.4f}  {vb:>16.4f}  {sign}{vb-va:>7.4f}")

    # ── 逐类 P / R / F1 ────────────────────────────────────────────────────
    p_a, r_a, f_a, sup = precision_recall_fscore_support(
        labels, preds_a, labels=label_ids, zero_division=0)
    p_b, r_b, f_b, _   = precision_recall_fscore_support(
        labels, preds_b, labels=label_ids, zero_division=0)

    print(f"\n{'逐类指标对照（A=' + pool_a + '，B=' + pool_b + '）':^66}")
    print(f"{'类别':<5}{'样本':>6}   {'A-P':>5}{'A-R':>6}{'A-F1':>6}"
          f"   {'B-P':>5}{'B-R':>6}{'B-F1':>6}   {'ΔF1':>7}")
    print(f"{'-'*66}")
    delta_f1 = f_b - f_a
    for i, name in enumerate(class_names):
        dr = delta_f1[i]
        marker = "  ◀ B 更好" if dr == delta_f1.max() and dr > 0 else \
                 "  ◀ A 更好" if dr == delta_f1.min() and dr < 0 else ""
        print(f"  {name:<4}{sup[i]:>6}   {p_a[i]:>5.2f}{r_a[i]:>6.2f}{f_a[i]:>6.2f}"
              f"   {p_b[i]:>5.2f}{r_b[i]:>6.2f}{f_b[i]:>6.2f}"
              f"   {dr:>+7.3f}{marker}")

    # ── 可视化 ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # 左图：各类别 F1 并排柱（F1 同时含精确率和召回率，比单看 Recall 更适合比池化）
    x = np.arange(len(class_names))
    w = 0.35
    axes[0].bar(x - w/2, f_a, w, label=tag_a, color="#4C72B0", alpha=0.85)
    axes[0].bar(x + w/2, f_b, w, label=tag_b, color="#DD8452", alpha=0.85)
    axes[0].set_title(f"各类别 F1 对比（{pool_a} vs {pool_b}，{suffix}）")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(class_names, rotation=40, ha="right")
    axes[0].set_ylabel("F1")
    axes[0].set_ylim(0, 0.85)
    axes[0].legend()
    axes[0].axhline(0.5, color="gray", linestyle="--", alpha=0.4)

    # 右图：F1 变化量（正 = B 更好）
    colors = ["#2ca02c" if d >= 0 else "#d62728" for d in delta_f1]
    axes[1].bar(class_names, delta_f1, color=colors, alpha=0.85)
    axes[1].axhline(0, color="black", linewidth=0.8)
    # 注意：这里必须用 ASCII 的 "-"，SimHei 没有 U+2212（数学减号）字形，会显示成方框
    axes[1].set_title(f"F1 变化量（{pool_b} - {pool_a}）")
    axes[1].set_ylabel("Δ F1")
    axes[1].tick_params(axis="x", rotation=40)
    for i, d in enumerate(delta_f1):
        axes[1].text(i, d + (0.002 if d >= 0 else -0.004),
                     f"{d:+.3f}", ha="center",
                     va="bottom" if d >= 0 else "top", fontsize=7)

    plt.suptitle(f"池化策略对比：{pool_a} vs {pool_b}（{suffix}，val set）", fontsize=13)
    plt.tight_layout()

    fig_name = f"compare_pool_{pool_a}_vs_{pool_b}_{suffix}.png"
    save_path = output_dir / fig_name
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\n对比图已保存 → {save_path}")

    # ── 一句话结论 ─────────────────────────────────────────────────────────
    print(f"\n整体：{pool_b} 的 Macro F1 {mf1_b:.4f} vs {pool_a} {mf1_a:.4f}"
          f"（{mf1_b - mf1_a:+.4f}），"
          f"Accuracy {acc_b:.4f} vs {acc_a:.4f}（{acc_b - acc_a:+.4f}）")
    win_b = int((delta_f1 > 0).sum())
    win_a = int((delta_f1 < 0).sum())
    print(f"逐类：{pool_b} 更好的有 {win_b} 类，{pool_a} 更好的有 {win_a} 类，"
          f"持平 {15 - win_b - win_a} 类")


def main():
    parser = argparse.ArgumentParser(description="比较两种池化策略")
    parser.add_argument("--pool_a",    default="cls", choices=["cls", "mean", "max"])
    parser.add_argument("--pool_b",    default="mean", choices=["cls", "mean", "max"])
    parser.add_argument("--bert_path", default=str(BERT_PATH))
    parser.add_argument("--data_dir",  default=str(DATA_DIR))
    args = parser.parse_args()

    if args.pool_a == args.pool_b:
        print(f"[错误] --pool_a 和 --pool_b 都是 {args.pool_a}，没有可比的")
        return

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data_dir)

    ckpt_a = CKPT_DIR / f"best_{args.pool_a}.pt"
    ckpt_b = CKPT_DIR / f"best_{args.pool_b}.pt"
    for p in [ckpt_a, ckpt_b]:
        if not p.exists():
            print(f"[缺少] {p}")
            print("请先训练对应池化策略，例如：")
            print(f"  python src/train.py --pool {p.stem.replace('best_', '')}")
            return

    with open(data_dir / "label_map.json", encoding="utf-8") as f:
        id2name = {int(k): v for k, v in json.load(f)["id2name"].items()}

    print(f"加载 {args.pool_a:<4}：{ckpt_a.name}")
    preds_a, labels, meta_a = get_predictions(ckpt_a, data_dir, args.bert_path, device)
    print(f"加载 {args.pool_b:<4}：{ckpt_b.name}")
    preds_b, _,      meta_b = get_predictions(ckpt_b, data_dir, args.bert_path, device)

    print(f"\n{args.pool_a:<4} → epoch={meta_a['epoch']}, "
          f"val_acc={meta_a['val_acc']:.4f}, "
          f"use_class_weight={meta_a['use_class_weight']}")
    print(f"{args.pool_b:<4} → epoch={meta_b['epoch']}, "
          f"val_acc={meta_b['val_acc']:.4f}, "
          f"use_class_weight={meta_b['use_class_weight']}")

    suffix = _weight_tag(meta_a, meta_b)
    if suffix == "mixed_weight":
        print("\n[注意] 两个模型一个加权一个不加权，对比会混入 loss 的影响，"
              "不纯是池化差异")

    compare_and_plot(
        preds_a, preds_b, labels, id2name,
        tag_a=f"{args.pool_a} (epoch={meta_a['epoch']})",
        tag_b=f"{args.pool_b} (epoch={meta_b['epoch']})",
        pool_a=args.pool_a, pool_b=args.pool_b,
        suffix=suffix,
    )


if __name__ == "__main__":
    main()
