"""
BERT 文本分类训练

教学重点：
  1. fine-tuning 的学习率设置：BERT 层用较小 lr（1e-5 ~ 3e-5），分类头可稍大
  2. 类别不均衡的处理：class_weight → 加权 CrossEntropyLoss
  3. GPU/CPU 自动兼容：torch.device 的标准用法
  4. 训练 checkpoint 保存策略：只保留验证集最优的模型
  5. 梯度累积（可选）：显存不足时等效扩大 batch size

使用方式：
  # 默认参数（CLS 池化，不加权 loss）
  python train.py

  # 使用均值池化 + 加权 loss（处理类别不均衡）
  python train.py --pool mean --use_class_weight

  # 自定义参数
  python train.py --pool max --epochs 5 --batch_size 16 --lr 2e-5

依赖：
  pip install torch transformers scikit-learn tqdm
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from tqdm import tqdm

from dataset import build_dataloaders
from model import build_model
from evaluate import evaluate_model

# ─────────────────── 默认路径（相对于 src/ 目录）────────────────────────────
ROOT          = Path(__file__).parent.parent
DATA_DIR      = ROOT / "data"
BERT_PATH     = ROOT.parent.parent / "pretrain_models" / "bert-base-chinese"
OUTPUT_DIR    = ROOT / "outputs"
CKPT_DIR      = OUTPUT_DIR / "checkpoints"


def parse_args():
    parser = argparse.ArgumentParser(description="BERT 文本分类训练")
    parser.add_argument("--bert_path",      default=str(BERT_PATH), type=str)
    parser.add_argument("--data_dir",       default=str(DATA_DIR),  type=str)
    parser.add_argument("--output_dir",     default=str(OUTPUT_DIR), type=str)
    parser.add_argument("--pool",           default="mean",
                        choices=["cls", "mean", "max"],
                        help="向量提取策略：cls / mean / max")
    parser.add_argument("--epochs",         default=3,   type=int)
    parser.add_argument("--batch_size",     default=32,  type=int)
    parser.add_argument("--max_length",     default=64, type=int)
    parser.add_argument("--lr",             default=2e-5, type=float,
                        help="BERT 层学习率")
    parser.add_argument("--head_lr_mult",   default=5.0,  type=float,
                        help="分类头学习率倍数（head_lr = lr * head_lr_mult）")
    parser.add_argument("--dropout",        default=0.1,  type=float)
    parser.add_argument("--warmup_ratio",   default=0.1,  type=float,
                        help="warmup 步数占总步数的比例")
    parser.add_argument("--grad_accum",     default=1,    type=int,
                        help="梯度累积步数，显存不足时设为 2/4")
    parser.add_argument("--use_class_weight", action="store_true",
                        help="使用加权 CrossEntropyLoss 处理类别不均衡")
    return parser.parse_args()


def compute_loss_weights(data_dir: Path, num_labels: int, device: torch.device):
    """根据训练集类别频次计算 inverse-frequency 权重。"""
    with open(data_dir / "train.json", encoding="utf-8") as f:
        train_data = json.load(f)
    labels = np.array([item["label"] for item in train_data])
    classes = np.arange(num_labels)
    weights = compute_class_weight("balanced", classes=classes, y=labels)
    print("类别权重（用于加权 loss）：")
    with open(data_dir / "label_map.json", encoding="utf-8") as f:
        id2name = {int(k): v for k, v in json.load(f)["id2name"].items()}
    for i, w in enumerate(weights):
        print(f"  {i:2d} {id2name[i]:4s}: {w:.3f}")
    return torch.tensor(weights, dtype=torch.float).to(device)


def train_one_epoch(
    model, loader, optimizer, scheduler, criterion,
    device, epoch, total_epochs, grad_accum
):
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    optimizer.zero_grad()

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [Train]", leave=False)
    for step, batch in enumerate(pbar):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels         = batch["label"].to(device)

        logits = model(input_ids, attention_mask, token_type_ids)  # [B, C]
        loss   = criterion(logits, labels)

        # 梯度累积：loss 除以累积步数，等效于更大 batch
        (loss / grad_accum).backward()

        if (step + 1) % grad_accum == 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        preds = logits.argmax(dim=-1)
        total_loss    += loss.item() * labels.size(0)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)
        pbar.set_postfix(loss=f"{total_loss/total_samples:.4f}",
                         acc=f"{total_correct/total_samples:.4f}")

    avg_loss = total_loss / total_samples
    avg_acc  = total_correct / total_samples
    return avg_loss, avg_acc


def main():
    args = parse_args()
    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    ckpt_dir   = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # ── 加载 label_map ───────────────────────────────────────────────────────
    with open(data_dir / "label_map.json", encoding="utf-8") as f:
        label_map = json.load(f)
    num_labels = label_map["num_labels"]
    id2name    = {int(k): v for k, v in label_map["id2name"].items()}
    print(f"类别数: {num_labels}")

    # ── Tokenizer & DataLoader ───────────────────────────────────────────────
    tokenizer = BertTokenizer.from_pretrained(args.bert_path)
    train_loader, val_loader, _ = build_dataloaders(
        data_dir, tokenizer,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )

    # ── 模型 ────────────────────────────────────────────────────────────────
    model = build_model(args.bert_path, num_labels, pool=args.pool)
    model = model.to(device)

    # ── Loss ────────────────────────────────────────────────────────────────
    if args.use_class_weight:
        weights = compute_loss_weights(data_dir, num_labels, device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        print("使用加权 CrossEntropyLoss")
    else:
        criterion = nn.CrossEntropyLoss()
        print("使用普通 CrossEntropyLoss")

    # ── 优化器：BERT 层和分类头用不同学习率 ─────────────────────────────────
    bert_params = list(model.bert.parameters())
    head_params = list(model.classifier.parameters()) + list(model.dropout.parameters())
    optimizer = AdamW([
        {"params": bert_params, "lr": args.lr},
        {"params": head_params, "lr": args.lr * args.head_lr_mult},
    ], weight_decay=0.01)

    total_steps  = len(train_loader) * args.epochs // args.grad_accum
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    print(f"总训练步数: {total_steps}, warmup: {warmup_steps}")

    # ── 训练循环 ─────────────────────────────────────────────────────────────
    best_val_acc = 0.0
    log_records  = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion,
            device, epoch, args.epochs, args.grad_accum
        )
        val_metrics = evaluate_model(model, val_loader, device, id2name,
                                     print_report=(epoch == args.epochs))
        elapsed = time.time() - t0

        val_acc = val_metrics["accuracy"]
        val_f1  = val_metrics["macro_f1"]
        print(f"Epoch {epoch}/{args.epochs} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_acc={val_acc:.4f} val_macro_f1={val_f1:.4f} | "
              f"{elapsed:.0f}s")

        log_records.append({
            "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
            "val_acc": val_acc, "val_macro_f1": val_f1, "elapsed_s": elapsed,
        })

        # 只保存验证集最优的 checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            run_tag  = f"{args.pool}_weighted" if args.use_class_weight else args.pool
            ckpt_path = ckpt_dir / f"best_{run_tag}.pt"
            torch.save({
                "epoch":           epoch,
                "pool":            args.pool,
                "use_class_weight": args.use_class_weight,
                "state_dict":      model.state_dict(),
                "val_acc":         val_acc,
                "val_macro_f1":    val_f1,
                "args":            vars(args),
            }, ckpt_path)
            print(f"  ✓ 新最优模型已保存 → {ckpt_path}  (val_acc={val_acc:.4f})")

    # ── 保存训练日志 ─────────────────────────────────────────────────────────
    run_tag  = f"{args.pool}_weighted" if args.use_class_weight else args.pool
    log_path = output_dir / f"train_log_{run_tag}.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_records, f, ensure_ascii=False, indent=2)
    print(f"\n训练完成。最优 val_acc={best_val_acc:.4f}")
    print(f"训练日志 → {log_path}")


if __name__ == "__main__":
    main()
