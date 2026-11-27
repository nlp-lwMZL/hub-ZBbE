# -*- coding: utf-8 -*-
"""
BERT + 线性分类头 训练脚本：把 BIO 标签当成「逐 token 的 7 分类」来做。

为什么用 BERT+Linear 而不是一上来就 CRF：
  本作业的重点是**数据格式处理**，不是解码器。线性头最直接地对应 BIO 的语义：
  每个字独立预测一个标签（O / B-X / I-X）。先把这条链路跑通、把日志留全，
  再考虑 CRF 这种「加约束」的进阶手段，学习路径更清楚。
  （课上 week7 的 BERT+CRF 是事件抽取主线；这里换成 NER + Linear，
    正好让学生看到「同一套 BERT 微调，换数据格式/换任务头」的差别。）

训练日志（学生课后复看的关键，和课件 week7 同一套三件套）：
  outputs/logs/<run_name>_train.json    每个 epoch 的 loss/验证指标/耗时 + 最终总结
  outputs/logs/<run_name>_steps.jsonl   每 log_every 步一行，画 loss 曲线用
  outputs/logs/<run_name>_console.log   本次运行的完整 stdout

运行：
  python src/train.py --bert_path <bert-base-chinese 目录>
  python src/train.py --bert_path ... --epochs 1 --limit_train 200 --limit_eval 100 --no_save_ckpt  # 冒烟
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from seqeval.metrics import f1_score, precision_score, recall_score

from bio_labels import build_label_maps, check_bio, load_label_names
from dataset import build_dataloaders
from log_utils import TrainLogger, peak_gpu_mb
from paths import (
    CKPT_DIR,
    PROJECT_ROOT,
    ensure_output_dirs,
    resolve_bert_path,
    resolve_data_dir,
)


def set_seed(seed: int) -> None:
    """固定随机种子。注意：同 seed 不保证逐位可复现（cuDNN/并行归约），但能大幅降低波动。"""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model(bert_path: Path, num_labels: int):
    """
    加载 BERT + 一个线性分类头。

    AutoModelForTokenClassification 会在 BERT 输出上接 `nn.Linear(hidden, num_labels)`，
    对每个 token 位置独立分类 —— 这正是 BIO 逐 token 标签的建模方式。
    """
    model = AutoModelForTokenClassification.from_pretrained(
        str(bert_path),
        num_labels=num_labels,
    )
    return model


@torch.no_grad()
def evaluate_epoch(model, loader, id2label: dict, device: torch.device) -> dict:
    """
    在验证集上算 loss + seqeval entity-level P/R/F1 + 非法 BIO 统计。

    为什么用 entity-level 而不是 token-level accuracy：
      本数据集 88.8% 的 token 是 O，全预测 O 也能有 ~89% 的 token accuracy，
      但这个模型完全没有用。entity-level 要求「类型 + 边界」整个 span 都对，
      才是 NER 真正关心的指标。
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_golds: list[list[str]] = []
    all_preds: list[list[str]] = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
            return_dict=True,   # transformers 5.x 默认返回 tuple，必须显式要 dict
        )
        total_loss += outputs.loss.item()
        n_batches += 1

        pred_ids = outputs.logits.argmax(dim=-1).cpu().tolist()
        label_ids = labels.cpu().tolist()

        for i in range(len(input_ids)):
            gold_seq, pred_seq = [], []
            for j, gold_id in enumerate(label_ids[i]):
                if gold_id == -100:      # 特殊 token / 非首子词 / padding：跳过
                    continue
                gold_seq.append(id2label[gold_id])
                pred_seq.append(id2label[pred_ids[i][j]])
            all_golds.append(gold_seq)
            all_preds.append(pred_seq)

    p = precision_score(all_golds, all_preds)
    r = recall_score(all_golds, all_preds)
    f1 = f1_score(all_golds, all_preds)

    illegal_seqs = 0
    illegal_transitions = 0
    for seq in all_preds:
        bad = check_bio(seq)
        illegal_seqs += bad["n_sequences_with_illegal"]
        illegal_transitions += bad["illegal_start"] + bad["illegal_transition"]

    return {
        "loss": round(total_loss / max(n_batches, 1), 6),
        "precision": round(float(p), 6),
        "recall": round(float(r), 6),
        "f1": round(float(f1), 6),
        "illegal_sequences": illegal_seqs,
        "illegal_transitions": illegal_transitions,
    }


def train_one_epoch(model, loader, optimizer, scheduler, device, epoch, total_epochs,
                    logger: TrainLogger, log_every: int) -> float:
    """训练一个 epoch，返回平均 train loss。"""
    model.train()
    total_loss = 0.0
    n_batches = 0
    t_epoch = time.perf_counter()

    for step, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
            return_dict=True,   # transformers 5.x 默认返回 tuple，必须显式要 dict
        )
        loss = outputs.loss

        loss.backward()
        # 梯度裁剪：BERT 微调初期偶尔有大梯度，裁到 1.0 更稳
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        n_batches += 1

        if step % log_every == 0 or step == len(loader) - 1:
            lr = scheduler.get_last_lr()[0]
            # 最后一步也写进 steps.jsonl（force=True），否则曲线缺 epoch 收尾
            logger.log_step(epoch=epoch, step=step, loss=loss.item(), lr=lr,
                            force=(step == len(loader) - 1))
            print(f"Epoch {epoch}/{total_epochs} | step {step:>4}/{len(loader) - 1} | "
                  f"loss={loss.item():.4f} | lr={lr:.2e} | "
                  f"elapsed={time.perf_counter() - t_epoch:.0f}s")

    return total_loss / max(n_batches, 1)


def count_parameters(model) -> int:
    """可训练参数量（用于日志/报告里写清楚模型规模）。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    args = parse_args()
    set_seed(args.seed)
    ensure_output_dirs()

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("指定了 --device cuda，但 torch.cuda.is_available() 为 False")
        device = torch.device("cuda")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:  # auto：有卡用卡，没卡回退 CPU（冒烟测试常用）
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = resolve_data_dir(args.data_dir)
    bert_path = resolve_bert_path(args.bert_path)
    label_names = load_label_names(data_dir)
    label2id, id2label = build_label_maps(label_names)

    tokenizer = AutoTokenizer.from_pretrained(str(bert_path), use_fast=True)
    if not tokenizer.is_fast:
        raise RuntimeError("需要 fast tokenizer（word_ids 对齐依赖它）；请确认 transformers 正常安装")

    train_loader, val_loader, _ = build_dataloaders(
        data_dir=data_dir,
        tokenizer=tokenizer,
        label2id=label2id,
        batch_size=args.batch_size,
        max_length=args.max_length,
        num_workers=args.num_workers,
        limit_train=args.limit_train,
        limit_eval=args.limit_eval,
    )

    model = build_model(bert_path, num_labels=len(label_names)).to(device)
    n_params = count_parameters(model)

    # 分层学习率：BERT 主体用 2e-5，随机初始化的分类头用 5 倍 lr 加快收敛。
    # 分类头如果也用 2e-5，前几个 epoch 基本在「学一个很差的随机头」，浪费算力。
    bert_params, head_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (bert_params if name.startswith("bert.") else head_params).append(param)

    optimizer = AdamW(
        [
            {"params": bert_params, "lr": args.lr},
            {"params": head_params, "lr": args.lr * args.head_lr_mult},
        ],
        weight_decay=args.weight_decay,
    )

    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # 先装好日志器，再打印训练头信息：这样「设备/模型/超参」也会进 console.log，
    # 学生只拿到日志文件时，能一眼看出这次跑的是什么配置。
    logger = TrainLogger(
        method=args.run_name,
        args=args,
        root=PROJECT_ROOT,
        params=n_params,
        log_every=args.log_every,
        notes="BERT+Linear on peoples_daily BIO NER；重点在 BIO 数据处理与完整日志，不在刷 SOTA",
    )

    print("=" * 78)
    print("BERT + Linear | 人民日报 NER（BIO）")
    print("=" * 78)
    print(f"设备：{device}")
    print(f"模型：{bert_path}")
    print(f"标签（{len(label_names)}）：{label_names}")
    print(f"可训练参数：{n_params:,}（{n_params / 1e6:.1f}M）")
    print(f"超参：epochs={args.epochs} batch={args.batch_size} max_length={args.max_length} "
          f"lr={args.lr} head_lr_mult={args.head_lr_mult} warmup={args.warmup_ratio:.0%} "
          f"weight_decay={args.weight_decay} seed={args.seed}")
    print(f"训练步数：{total_steps}（warmup {warmup_steps}）")
    print(f"训练样本：{len(train_loader.dataset)}，验证样本：{len(val_loader.dataset)}")

    best_f1 = -1.0
    best_epoch = 0
    t_start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device,
            epoch, args.epochs, logger, args.log_every,
        )
        val = evaluate_epoch(model, val_loader, id2label, device)
        elapsed = time.perf_counter() - t0

        print(f"\nEpoch {epoch}/{args.epochs} 完成 | train_loss={train_loss:.4f} | "
              f"val_loss={val['loss']:.4f} | val_P={val['precision']:.4f} | "
              f"val_R={val['recall']:.4f} | val_F1={val['f1']:.4f} | "
              f"非法序列={val['illegal_sequences']} | 用时={elapsed:.0f}s\n")

        logger.log_epoch(epoch=epoch, train_loss=train_loss, elapsed=elapsed, val_metrics=val)

        if val["f1"] > best_f1:
            best_f1 = val["f1"]
            best_epoch = epoch
            print(f"  ★ 新最优 val_F1={best_f1:.4f}（epoch {best_epoch}）")
            if args.save_ckpt:
                ckpt_path = CKPT_DIR / f"{args.run_name}.pt"
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "epoch": epoch,
                        "val_f1": best_f1,
                        "label_names": label_names,
                        "max_length": args.max_length,
                        "args": vars(args),
                    },
                    ckpt_path,
                )
                print(f"  已保存 checkpoint → {ckpt_path}（约 {ckpt_path.stat().st_size / 1024**2:.0f} MB）")

    total_s = time.perf_counter() - t_start

    # 先打印最终汇总（会被 console.log 捕获），再 finalize 收尾日志文件。
    # 反过来的话，finalize 已经恢复 stdout，汇总行只出现在屏幕、不在日志里。
    print("\n" + "=" * 78)
    print(f"训练完成：best epoch={best_epoch}  best val_F1={best_f1:.4f}  "
          f"总耗时={total_s / 60:.1f} min  峰值显存={peak_gpu_mb():.0f} MB")
    if args.save_ckpt:
        print(f"Checkpoint：{CKPT_DIR / (args.run_name + '.pt')}")
        print(f"下一步：python src/evaluate.py --bert_path {bert_path} --run_name {args.run_name}")
    else:
        print("本次未保存 checkpoint（--no_save_ckpt）；模型只用于冒烟/看日志。")

    logger.finalize(
        total_s=total_s,
        best_epoch=best_epoch,
        best_val_f1=best_f1,
        peak_gpu_mb=peak_gpu_mb(),
        n_params=n_params,
        train_size=len(train_loader.dataset),
        val_size=len(val_loader.dataset),
    )


def parse_args():
    parser = argparse.ArgumentParser(description="BERT+Linear 训练人民日报 NER（BIO）")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--bert_path", type=str, default=None)
    parser.add_argument("--run_name", type=str, default="bert_bio_128")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--head_lr_mult", type=float, default=5.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--limit_train", type=int, default=0, help="只取前 N 条训练（0=全量，冒烟用）")
    parser.add_argument("--limit_eval", type=int, default=0, help="只取前 N 条验证（0=全量，冒烟用）")
    parser.add_argument("--save_ckpt", action="store_true", default=True)
    parser.add_argument("--no_save_ckpt", action="store_false", dest="save_ckpt")
    return parser.parse_args()


if __name__ == "__main__":
    main()
