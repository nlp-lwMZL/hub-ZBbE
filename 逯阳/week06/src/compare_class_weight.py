"""
加权 Loss vs 普通 Loss 效果对比

教学重点：
  1. 类别不均衡时 accuracy 会"说谎"——整体 acc 差不多，但少数类 Recall 差距很大
  2. 加权 loss 的代价：提升少数类 Recall 的同时，可能略微降低多数类的 Precision
  3. macro F1 比 weighted F1 更能反映加权 loss 的收益

使用方式：
  python compare_class_weight.py
  python compare_class_weight.py --pool cls --epochs 1
"""

import argparse
import json
from pathlib import Path

import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from sklearn.metrics import classification_report, f1_score, accuracy_score

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

ROOT      = Path(__file__).parent.parent
DATA_DIR  = ROOT / "data"
BERT_PATH = ROOT.parent.parent / "pretrain_models" / "bert-base-chinese"
CKPT_DIR  = ROOT / "outputs" / "checkpoints"
FIG_DIR   = ROOT / "outputs" / "figures"


def get_predictions(ckpt_path, data_dir, bert_path, device):
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
        data_dir, tokenizer, max_length=128, batch_size=64
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


def compare_and_plot(
    preds_a, preds_b, labels, id2name,
    tag_a="普通 Loss", tag_b="加权 Loss",
    pool: str = "cls",
    output_dir: Path = FIG_DIR,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    label_ids   = sorted(id2name.keys())
    class_names = [id2name[i] for i in label_ids]

    # ── 整体指标 ────────────────────────────────────────────────────────────────
    acc_a  = accuracy_score(labels, preds_a)
    acc_b  = accuracy_score(labels, preds_b)
    mf1_a  = f1_score(labels, preds_a, average="macro",    zero_division=0)
    mf1_b  = f1_score(labels, preds_b, average="macro",    zero_division=0)
    wf1_a  = f1_score(labels, preds_a, average="weighted", zero_division=0)
    wf1_b  = f1_score(labels, preds_b, average="weighted", zero_division=0)

    print(f"\n{'='*60}")
    print(f"{'整体指标对比':^60}")
    print(f"{'='*60}")
    print(f"{'指标':<20} {tag_a:>12}  {tag_b:>12}  {'差值':>8}")
    print(f"{'-'*60}")
    for name, va, vb in [("Accuracy", acc_a, acc_b),
                          ("Macro F1",  mf1_a, mf1_b),
                          ("Weighted F1", wf1_a, wf1_b)]:
        sign = "+" if vb - va >= 0 else ""
        print(f"  {name:<18} {va:>12.4f}  {vb:>12.4f}  {sign}{vb-va:>7.4f}")

    # ── per-class Recall 对比 ────────────────────────────────────────────────
    from sklearn.metrics import recall_score, precision_score
    recall_a = recall_score(labels, preds_a, labels=label_ids,
                            average=None, zero_division=0)
    recall_b = recall_score(labels, preds_b, labels=label_ids,
                            average=None, zero_division=0)
    prec_a   = precision_score(labels, preds_a, labels=label_ids,
                               average=None, zero_division=0)
    prec_b   = precision_score(labels, preds_b, labels=label_ids,
                               average=None, zero_division=0)

    print(f"\n{'各类别 Recall 对比':^60}")
    print(f"{'类别':<6} {tag_a:>10}  {tag_b:>10}  {'Δ Recall':>10}")
    print(f"{'-'*45}")
    delta_recall = recall_b - recall_a
    for i, (name, ra, rb, dr) in enumerate(
            zip(class_names, recall_a, recall_b, delta_recall)):
        marker = " ◀ 最大提升" if dr == delta_recall.max() and dr > 0 else \
                 " ◀ 最大下降" if dr == delta_recall.min() and dr < 0 else ""
        sign = "+" if dr >= 0 else ""
        print(f"  {name:<4} {ra:>10.3f}  {rb:>10.3f}  {sign}{dr:>9.3f}{marker}")

    # ── 可视化：Recall 变化量 ────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # 左图：各类别 Recall 并排柱状图
    x = np.arange(len(class_names))
    w = 0.35
    axes[0].bar(x - w/2, recall_a, w, label=tag_a, color="#4C72B0", alpha=0.8)
    axes[0].bar(x + w/2, recall_b, w, label=tag_b, color="#C44E52", alpha=0.8)
    axes[0].set_title("各类别 Recall 对比")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(class_names, rotation=40, ha="right")
    axes[0].set_ylabel("Recall")
    axes[0].legend()
    axes[0].axhline(0.5, color="gray", linestyle="--", alpha=0.4)

    # 右图：Recall 变化量（正 = 加权后提升）
    colors = ["#2ca02c" if d >= 0 else "#d62728" for d in delta_recall]
    axes[1].bar(class_names, delta_recall, color=colors, alpha=0.85)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_title(f"Recall 变化量（加权 − 普通）")
    axes[1].set_ylabel("Δ Recall")
    axes[1].tick_params(axis="x", rotation=40)
    # 标注数值
    for i, d in enumerate(delta_recall):
        axes[1].text(i, d + (0.005 if d >= 0 else -0.012),
                     f"{d:+.3f}", ha="center", va="bottom" if d >= 0 else "top",
                     fontsize=7)

    plt.suptitle(f"类别加权 Loss 效果对比（val set, 1 epoch）", fontsize=13)
    plt.tight_layout()
    # 文件名带上 pool 类型，cls / mean / max 三套对比图不会互相覆盖
    save_path = output_dir / f"compare_class_weight_{pool}.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\n对比图已保存 → {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool",      default="cls")
    parser.add_argument("--bert_path", default=str(BERT_PATH))
    parser.add_argument("--data_dir",  default=str(DATA_DIR))
    args = parser.parse_args()

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data_dir)

    ckpt_plain    = CKPT_DIR / f"best_{args.pool}.pt"
    ckpt_weighted = CKPT_DIR / f"best_{args.pool}_weighted.pt"

    for p in [ckpt_plain, ckpt_weighted]:
        if not p.exists():
            print(f"[缺少] {p}")
            print("请先分别运行：")
            print(f"  python train.py --pool {args.pool} --epochs 1")
            print(f"  python train.py --pool {args.pool} --epochs 1 --use_class_weight")
            return

    with open(data_dir / "label_map.json", encoding="utf-8") as f:
        id2name = {int(k): v for k, v in json.load(f)["id2name"].items()}

    print(f"加载普通 Loss 模型：{ckpt_plain.name}")
    preds_a, labels, ckpt_a = get_predictions(ckpt_plain, data_dir, args.bert_path, device)
    print(f"加载加权 Loss 模型：{ckpt_weighted.name}")
    preds_b, _,      ckpt_b = get_predictions(ckpt_weighted, data_dir, args.bert_path, device)

    print(f"\n普通 Loss  → epoch={ckpt_a['epoch']}, val_acc={ckpt_a['val_acc']:.4f}")
    print(f"加权 Loss  → epoch={ckpt_b['epoch']}, val_acc={ckpt_b['val_acc']:.4f}")

    compare_and_plot(preds_a, preds_b, labels, id2name,
                     tag_a=f"普通 Loss (epoch={ckpt_a['epoch']})",
                     tag_b=f"加权 Loss (epoch={ckpt_b['epoch']})",
                     pool=args.pool)


if __name__ == "__main__":
    main()
