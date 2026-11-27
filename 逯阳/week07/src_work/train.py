"""
人民日报 NER 训练脚本（作业用，与 ../src/train.py 的 cluener 版本完全隔离）

和 cluener 版的差异只有三处，其余训练逻辑一模一样：
  1. 数据目录：data/peoples_daily（7 个标签，3 类实体）
  2. 输出目录：outputs_work/{checkpoints,logs,figures}
  3. 日志/权重文件名带 work_ 前缀，避免和主项目的 best_linear.pt / train_linear.json 撞名

教学重点（与主项目一致）：
  1. --use_crf 一套脚本切换两种模型
  2. 分层学习率：BERT 2e-5，分类头 1e-4（head_lr_mult=5）
  3. 线性 warmup 10%，之后线性衰减到 0
  4. seqeval entity-level F1（不是 token accuracy）

使用方式：
  python src_work/train.py                 # BERT + Linear
  python src_work/train.py --use_crf       # BERT + CRF
  python src_work/train.py --epochs 1 --batch_size 16   # 快速冒烟

断点续训（在已有权重基础上再多跑几轮）：
  python src_work/train.py --resume outputs_work/checkpoints/work_best_linear.pt --epochs 2
  # 注意：--resume 模式下 --epochs 表示【额外】训练的轮数，不是总轮数
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import time
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

from dataset import build_label_schema, build_dataloaders, entity_types
from model import build_model

ROOT = Path(__file__).parent.parent
# 注意：ROOT = 序列标注项目/，所以是 ROOT.parent（= D:/project），不是 parent.parent
BERT_PATH = ROOT.parent / "pretrain_models" / "bert-base-chinese"
DATA_DIR = ROOT / "data" / "peoples_daily"

# 作业专用输出目录，和主项目的 outputs/ 完全分开
WORK_DIR = ROOT / "outputs_work"
CKPT_DIR = WORK_DIR / "checkpoints"
LOG_DIR = WORK_DIR / "logs"


def evaluate_epoch(
    model: nn.Module,
    loader,
    id2label: dict,
    device: torch.device,
    use_crf: bool,
) -> tuple[float, float]:
    """在 loader 上评估，返回 (avg_loss, entity_f1)。"""
    from seqeval.metrics import f1_score as seqeval_f1

    model.eval()
    total_loss = 0.0
    all_preds: list[list[str]] = []
    all_golds: list[list[str]] = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["labels"].to(device)

            if use_crf:
                _, loss = model(input_ids, attention_mask, token_type_ids, labels)
                pred_ids_list = model.decode(input_ids, attention_mask, token_type_ids)
            else:
                logits, loss = model(input_ids, attention_mask, token_type_ids, labels)
                pred_ids_list = logits.argmax(dim=-1).tolist()

            total_loss += loss.item()

            labels_np = labels.cpu().tolist()
            for i in range(len(input_ids)):
                gold_seq, pred_seq = [], []
                pred_ids = pred_ids_list[i]

                for j, gold_id in enumerate(labels_np[i]):
                    if gold_id == -100:
                        continue
                    gold_seq.append(id2label[gold_id])
                    # CRF 的 pred 长度 = 实际 token 数，可能短于 padding 后的长度，要防越界
                    pred_seq.append(
                        id2label.get(pred_ids[j], "O") if j < len(pred_ids) else "O"
                    )

                all_golds.append(gold_seq)
                all_preds.append(pred_seq)

    avg_loss = total_loss / len(loader)
    return avg_loss, seqeval_f1(all_golds, all_preds)


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer,
    scheduler,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    grad_accum: int,
) -> float:
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [Train]", leave=False)
    for step, batch in enumerate(pbar):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels = batch["labels"].to(device)

        _, loss = model(input_ids, attention_mask, token_type_ids, labels)

        (loss / grad_accum).backward()
        total_loss += loss.item()

        if (step + 1) % grad_accum == 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    # 最后不足 grad_accum 的批次也要更新一次，否则尾部梯度白算
    if len(loader) % grad_accum != 0:
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return total_loss / len(loader)


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备：{device}")

    labels, label2id, id2label = build_label_schema(DATA_DIR)
    num_labels = len(labels)
    print(f"BIO 标签数：{num_labels}  →  {labels}")
    print(f"实体类型：{entity_types(labels)}")

    tokenizer = BertTokenizer.from_pretrained(str(args.bert_path))

    train_loader, val_loader, _ = build_dataloaders(
        tokenizer=tokenizer,
        label2id=label2id,
        batch_size=args.batch_size,
        max_length=args.max_length,
        data_dir=DATA_DIR,
    )

    model = build_model(
        use_crf=args.use_crf,
        bert_path=str(args.bert_path),
        num_labels=num_labels,
        dropout=args.dropout,
    ).to(device)

    run_tag = "crf" if args.use_crf else "linear"
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CKPT_DIR / f"work_best_{run_tag}.pt"
    log_path = LOG_DIR / f"work_train_{run_tag}.json"

    # ── 断点续训：把已有权重读进来，在它基础上再跑 --epochs 轮 ──────────────
    # 两处和从头训练不同：
    #   1. 学习率砍半 —— 模型已经收敛到 0.95 附近，还用原来的 2e-5 会把它推离最优
    #   2. 不做 warmup —— 预热的意义是让随机初始化的分类头先稳一下，续训时分类头
    #      早就学好了，重新把 LR 拉回高位再降下来纯粹是扰动
    start_epoch = 0
    best_f1 = 0.0
    log_records: list[dict] = []
    lr_bert = args.lr
    warmup_ratio = args.warmup_ratio

    if args.resume:
        if not Path(args.resume).exists():
            raise SystemExit(f"续训失败：找不到权重 {args.resume}")
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        start_epoch = int(ckpt["epoch"])
        best_f1 = float(ckpt["val_entity_f1"])
        # 接着上次的日志往后写，不要把前 3 轮的记录冲掉
        if log_path.exists():
            log_records = json.loads(log_path.read_text(encoding="utf-8"))
        lr_bert = args.lr * 0.5
        warmup_ratio = 0.0
        print(f"\n续训模式：从 {Path(args.resume).name} 恢复")
        print(f"  已训练 {start_epoch} 轮，历史最优 val_f1={best_f1:.4f}")
        print(f"  本次再跑 {args.epochs} 轮，学习率 {lr_bert:.2e}（原 {args.lr:.2e} 的一半），不 warmup")

    # 分层学习率：BERT 小步微调，分类头（含 CRF 转移矩阵）大步收敛
    bert_params = list(model.bert.parameters())
    head_params = list(model.classifier.parameters())
    if args.use_crf:
        head_params += list(model.crf.parameters())

    optimizer = AdamW(
        [
            {"params": bert_params, "lr": lr_bert},
            {"params": head_params, "lr": lr_bert * args.head_lr_mult},
        ],
        weight_decay=0.01,
    )

    total_steps = len(train_loader) * args.epochs // args.grad_accum
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    print(f"\n训练步数：{total_steps}，预热步数：{warmup_steps}")

    end_epoch = start_epoch + args.epochs   # 续训时用于显示 "Epoch 4/5"

    print(f"\n开始训练（{'BERT+CRF' if args.use_crf else 'BERT+Linear'}）...")
    for epoch in range(start_epoch + 1, end_epoch + 1):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device,
            epoch, end_epoch, args.grad_accum,
        )
        val_loss, val_f1 = evaluate_epoch(model, val_loader, id2label, device, args.use_crf)
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch}/{end_epoch} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_entity_f1={val_f1:.4f} | "
            f"time={elapsed:.0f}s"
        )

        log_records.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "val_entity_f1": round(val_f1, 6),
            "elapsed_s": round(elapsed, 1),
        })

        # 每个 epoch 都落盘一次日志，中途被打断也能看到已有进度
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(log_records, f, ensure_ascii=False, indent=2)

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(
                {
                    "epoch": epoch,
                    "use_crf": args.use_crf,
                    "state_dict": model.state_dict(),
                    "val_entity_f1": val_f1,
                    "label2id": label2id,
                    "id2label": id2label,
                    "args": vars(args),
                },
                ckpt_path,
            )
            print(f"  ★ 新最优 F1={val_f1:.4f}，已保存 → {ckpt_path}")

    print(f"\n训练完成！最优 val_entity_f1={best_f1:.4f}")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  训练日志:   {log_path}")
    print(f"\n下一步：python src_work/evaluate.py {'--use_crf' if args.use_crf else ''}")


def parse_args():
    parser = argparse.ArgumentParser(description="训练 BERT NER（人民日报数据集）")
    parser.add_argument("--use_crf", action="store_true", help="使用 CRF 层（否则线性头）")
    parser.add_argument("--bert_path", type=Path, default=BERT_PATH)
    parser.add_argument("--epochs", type=int, default=3,
                        help="训练轮数；配合 --resume 时表示【额外】训练的轮数")
    parser.add_argument("--resume", type=Path, default=None,
                        help="从该 checkpoint 继续训练（学习率自动减半、不做 warmup）")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-5, help="BERT 层学习率")
    parser.add_argument("--head_lr_mult", type=float, default=5.0, help="分类头学习率倍数")
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    return parser.parse_args()


if __name__ == "__main__":
    main()
