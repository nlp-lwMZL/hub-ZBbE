"""
模型评估脚本

教学重点：
  1. accuracy 和 macro F1 的区别：类别不均衡时 macro F1 更能反映真实性能
  2. 混淆矩阵：快速定位模型在哪些类别之间容易混淆
  3. 每类别 Precision / Recall / F1：发现长尾类别（如"证券"）的实际表现

使用方式：
  # 作为脚本直接运行（加载 checkpoint 评估）
  python evaluate.py --pool cls
  python evaluate.py --pool mean --ckpt_path ../outputs/checkpoints/best_mean.pt

  # 作为模块调用（在 train.py 中使用）
  from evaluate import evaluate_model
  metrics = evaluate_model(model, val_loader, device, id2name)

依赖：
  pip install torch transformers scikit-learn seaborn matplotlib
"""

import argparse
import json
from pathlib import Path

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
)
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

matplotlib.rcParams["axes.unicode_minus"] = False

# ── 中文字体 ──────────────────────────────────────────────────────────────────
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

ROOT     = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
BERT_PATH = ROOT.parent.parent / "pretrain_models" / "bert-base-chinese"
CKPT_DIR = ROOT / "outputs" / "checkpoints"
FIG_DIR  = ROOT / "outputs" / "figures"


def evaluate_model(
    model,
    loader,
    device: torch.device,
    id2name: dict,
    print_report: bool = True,
) -> dict:
    """
    在给定 DataLoader 上评估模型，返回指标字典。
    可在 train.py 的每个 epoch 末调用。
    """
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels         = batch["label"]

            logits = model(input_ids, attention_mask, token_type_ids)
            preds  = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    # 过滤 test 集的 -1 标签（test 集无标签）
    valid_mask = all_labels != -1
    all_preds  = all_preds[valid_mask]
    all_labels = all_labels[valid_mask]

    acc      = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    if print_report:
        label_ids   = sorted(id2name.keys())
        target_names = [id2name[i] for i in label_ids]
        print("\n分类报告：")
        print(classification_report(
            all_labels, all_preds,
            labels=label_ids,
            target_names=target_names,
            zero_division=0,
        ))

    return {
        "accuracy":  acc,
        "macro_f1":  macro_f1,
        "preds":     all_preds,
        "labels":    all_labels,
    }


def plot_confusion_matrix(preds, labels, id2name, save_path: Path):
    label_ids    = sorted(id2name.keys())
    class_names  = [id2name[i] for i in label_ids]
    cm = confusion_matrix(labels, preds, labels=label_ids)

    # 归一化（按行，即真实类别）
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # 原始计数
    sns.heatmap(cm, ax=axes[0], annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={"size": 7})
    axes[0].set_title("混淆矩阵（绝对计数）")
    axes[0].set_xlabel("预测类别")
    axes[0].set_ylabel("真实类别")
    axes[0].tick_params(axis="x", rotation=40)
    axes[0].tick_params(axis="y", rotation=0)

    # 归一化
    sns.heatmap(cm_norm, ax=axes[1], annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={"size": 7}, vmin=0, vmax=1)
    axes[1].set_title("混淆矩阵（按行归一化，对角线 = Recall）")
    axes[1].set_xlabel("预测类别")
    axes[1].set_ylabel("真实类别")
    axes[1].tick_params(axis="x", rotation=40)
    axes[1].tick_params(axis="y", rotation=0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"混淆矩阵已保存 → {save_path}")


def main():
    parser = argparse.ArgumentParser(description="加载 checkpoint 并评估")
    parser.add_argument("--pool",      default="cls", choices=["cls", "mean", "max"])
    parser.add_argument("--ckpt_path", default=None, type=str,
                        help="默认使用 outputs/checkpoints/best_{pool}.pt")
    parser.add_argument("--bert_path", default=str(BERT_PATH), type=str)
    parser.add_argument("--data_dir",  default=str(DATA_DIR),  type=str)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--max_length", default=128, type=int)
    args = parser.parse_args()

    data_dir  = Path(args.data_dir)
    ckpt_path = Path(args.ckpt_path) if args.ckpt_path \
                else CKPT_DIR / f"best_{args.pool}.pt"
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    with open(data_dir / "label_map.json", encoding="utf-8") as f:
        label_map = json.load(f)
    num_labels = label_map["num_labels"]
    id2name    = {int(k): v for k, v in label_map["id2name"].items()}

    from transformers import BertTokenizer
    from dataset import build_dataloaders
    from model import build_model

    tokenizer = BertTokenizer.from_pretrained(args.bert_path)
    _, val_loader, _ = build_dataloaders(
        data_dir, tokenizer, max_length=args.max_length, batch_size=args.batch_size
    )

    model = build_model(args.bert_path, num_labels, pool=args.pool)
    # weights_only=False：checkpoint 含 args dict（包含 Python 基础类型），
    # PyTorch 2.6 默认改为 True，本地可信文件需显式设为 False
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device)
    print(f"Checkpoint 加载成功：{ckpt_path}")
    print(f"  训练 epoch={ckpt['epoch']}, val_acc={ckpt['val_acc']:.4f}")

    metrics = evaluate_model(model, val_loader, device, id2name, print_report=True)
    print(f"\nval accuracy : {metrics['accuracy']:.4f}")
    print(f"val macro F1 : {metrics['macro_f1']:.4f}")

    cm_path = FIG_DIR / f"confusion_matrix_{args.pool}.png"
    plot_confusion_matrix(metrics["preds"], metrics["labels"], id2name, cm_path)


if __name__ == "__main__":
    main()
