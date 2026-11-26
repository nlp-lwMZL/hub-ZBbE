"""
L2 训练：RoBERTa-wwm-ext + Linear / CRF（NER，cdtier/clean）。

用法（Colab）：
  python bert/train.py --decoder linear --run_name roberta_linear
  python bert/train.py --decoder crf    --run_name roberta_crf
  python bert/train.py --decoder crf --max_steps 20 --eval_max_batches 2 --run_name smoke  # 冒烟
本机冒烟（无 RoBERTa 权重，用本地 bert）：
  python bert/train.py --bert_path /本机/bert-base-chinese --tokenizer /本机/bert-base-chinese \
      --decoder linear --max_steps 20 --eval_max_batches 2 --run_name smoke

【纪律】
1. 用 train 切 10% 内层验证集选最佳 epoch，dev/test 只在 evaluate 阶段用一次；
2. 每类方法的预测都写成 common/metrics 的 pred_spans JSONL，三类同口径；
3. loss 量纲 CE vs CRF NLL 不同，选模看实体级 F1，不看 loss；
4. 冒烟产物写 outputs/smoke，不污染正式结果。
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Subset

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR / "bert"))

from common import paths
from common.ner_data import (IGNORE_INDEX, DataConfig, DataLoaderFactory, LabelScheme,
                            CharDataset, load_records, decode_bio_ids)
from common.metrics import write_predictions, evaluate
from model import build_model, resolve_bert_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="L2 RoBERTa NER 训练")
    p.add_argument("--decoder", choices=["linear", "crf"], default="linear")
    p.add_argument("--bert_path", default=None, help="本地底座；不给则用 RoBERTa-wwm-ext")
    p.add_argument("--tokenizer", default=None, help="tokenizer；默认同 bert_path")
    p.add_argument("--run_name", default=None, help="不填自动按方法+时间生成")
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--eval_batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--head_lr_mult", type=float, default=5.0)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--inner_val", type=float, default=0.1, help="从 train 切多少做内层验证")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--max_steps", type=int, default=0, help="每 epoch 最多训练多少 batch；0=全量")
    p.add_argument("--eval_max_batches", type=int, default=0, help="验证最多跑多少 batch；0=全量")
    p.add_argument("--num_train", type=int, default=0, help="只用前 N 条训练；0=全量（消融用）")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            try:
                torch.ones(1, device="mps"); return torch.device("mps")
            except (RuntimeError, AssertionError):
                pass
        return torch.device("cpu")
    return torch.device(name)


def to_device(batch: dict, device: torch.device) -> dict:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def make_scheduler(optimizer, total_steps: int, warmup_ratio: float):
    warm = max(1, int(total_steps * warmup_ratio))
    decay = max(1, total_steps - warm)
    def lr_lambda(s: int) -> float:
        if s < warm:
            return float(s + 1) / warm
        return max(0.0, 1.0 - float(s - warm) / decay)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda), warm


def collect_sequences(labels: torch.Tensor, pred: torch.Tensor,
                      id2label: dict[int, str]) -> tuple[list[list[str]], list[list[str]]]:
    """按 labels != -100 同一口径过滤，转 BIO 字符串（gold/pred 同规则）。"""
    golds, preds = [], []
    lc, pc = labels.cpu(), pred.cpu()
    for b in range(lc.shape[0]):
        g, p = [], []
        for j in range(lc.shape[1]):
            gid = int(lc[b, j])
            if gid == IGNORE_INDEX:
                continue
            g.append(id2label[gid]); p.append(id2label[int(pc[b, j])])
        golds.append(g); preds.append(p)
    return golds, preds


def batch_to_pred_spans(batch: dict, pred: torch.Tensor, id2label: dict[int, str],
                        records: list[dict]) -> dict[str, list[dict]]:
    """把 decode 的 BIO id 转 span，写字间距（model 输出按 token 位，此处 tokens==字符）。

    records 必须是**当前 batch 对应的切片**，不能传全量数据集：zip 以短的为准，
    传全量会导致只有第一个 batch 的 id 被写入。
    """
    out: dict[str, list[dict]] = {}
    golds, preds = collect_sequences(batch["labels"], pred, id2label)
    assert len(records) == len(preds), "records 必须是当前 batch 的切片，长度应与 preds 一致"
    for rec, tags in zip(records, preds):
        spans = []
        for sp in _bio_tags_to_spans(tags):
            s, e = sp["start"], sp["end"]
            text = "".join(rec["tokens"][s:e])
            spans.append({"type": sp["type"], "start": s, "end": e, "text": text})
        out[rec["id"]] = spans
    return out


def _bio_tags_to_spans(tags: list[str]) -> list[dict]:
    spans: list[dict] = []
    cur_type: str | None = None
    cur_start = 0
    def close(end: int) -> None:
        nonlocal cur_type
        if cur_type is not None:
            spans.append({"type": cur_type, "start": cur_start, "end": end})
            cur_type = None
    for i, t in enumerate(tags):
        if t == "O":
            close(i); continue
        pre, _, et = t.partition("-")
        if pre == "B":
            close(i); cur_type, cur_start = et, i
        else:
            if cur_type != et:
                close(i); cur_type, cur_start = et, i
    close(len(tags))
    return spans


@torch.no_grad()
def evaluate_loss_f1(model, loader: DataLoader, id2label: dict[int, str],
                     device: torch.device, max_batches: int):
    from seqeval.metrics import f1_score
    model.eval()
    total_loss, n = 0.0, 0
    golds_all, preds_all = [], []
    for i, batch in enumerate(loader):
        if max_batches > 0 and i >= max_batches:
            break
        batch = to_device(batch, device)
        _, loss = model(**batch)
        pred = model.decode(**batch)
        g, p = collect_sequences(batch["labels"], pred, id2label)
        golds_all.extend(g); preds_all.extend(p)
        total_loss += float(loss.item()); n += 1
    f1 = f1_score(golds_all, preds_all) if golds_all else 0.0
    return total_loss / max(n, 1), float(f1)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    smoke = args.max_steps > 0 or args.eval_max_batches > 0 or args.num_train > 0
    tag = args.run_name or f"roberta_{args.decoder}_{datetime.now():%m%d_%H%M%S}"
    out_root = paths.get_out_root() / ("smoke" if smoke else "")
    log_dir = out_root / "logs"; log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_root / "checkpoints" / f"{tag}.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"train_{tag}.jsonl"
    pred_path = paths.pred_dir() / f"{tag}_val.jsonl" if not smoke else log_dir / f"{tag}_val.jsonl"

    data_dir = paths.get_data_dir()
    scheme = LabelScheme.from_json(data_dir / "label_names.json")
    bert_path = resolve_bert_path(args.bert_path)
    tok_path = args.tokenizer or bert_path
    print("=" * 72)
    print(f"L2 RoBERTa NER | decoder={args.decoder} | device={device} | run={tag}")
    print(f"  bert_path={bert_path}\n  tokenizer={tok_path}\n  data_dir={data_dir}")
    print(f"  epochs={args.epochs} bs={args.batch_size} lr={args.lr} "
          f"head_lr_mult={args.head_lr_mult} max_len={args.max_length}")
    if smoke:
        print("  [冒烟/小规模] 产物落 smoke 目录")
    print("=" * 72)

    # 数据：train 切内层验证；dev/test 只留给 evaluate.py
    records = load_records("train", data_dir)
    if args.num_train > 0:
        records = records[: args.num_train]
    rng = random.Random(args.seed)
    idx = list(range(len(records))); rng.shuffle(idx)
    n_val = max(1, int(len(records) * args.inner_val))
    val_idx = set(idx[:n_val])
    train_rows = [r for i, r in enumerate(records) if i not in val_idx]
    val_rows = [r for i, r in enumerate(records) if i in val_idx]
    print(f"  训练 {len(train_rows)} / 内层验证 {len(val_rows)} 句（dev/test 未参与）")

    cfg = DataConfig(data_dir=data_dir, tokenizer_name_or_path=tok_path,
                     max_length=args.max_length, batch_size=args.batch_size,
                     eval_batch_size=args.eval_batch_size)
    factory = DataLoaderFactory(tok_path, scheme, cfg)
    # 复用 factory 的 tokenizer，对 train/val 子集单独编码
    from common.ner_data import NERDataset
    train_ds = NERDataset(train_rows, factory.tokenizer, scheme, cfg)
    val_ds = NERDataset(val_rows, factory.tokenizer, scheme, cfg, verbose=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=factory.collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.eval_batch_size, shuffle=False,
                            collate_fn=factory.collate_fn)
    print(f"  训练截断率 {train_ds.stats['truncated_samples']}/{len(train_ds)} "
          f"({train_ds.stats['truncated_samples']/max(1,len(train_ds))*100:.2f}%)")

    model = build_model(args.decoder == "crf", bert_path, scheme.num_labels, args.dropout).to(device)
    optimizer = torch.optim.AdamW(
        model.parameter_groups(args.lr, args.lr * args.head_lr_mult),
        weight_decay=args.weight_decay)
    steps_per_ep = args.max_steps if args.max_steps > 0 else len(train_loader)
    scheduler, warm = make_scheduler(optimizer, steps_per_ep * args.epochs, args.warmup_ratio)
    print(f"  优化器 AdamW total_steps={steps_per_ep*args.epochs} warmup={warm}\n")

    best_f1, best_epoch = -1.0, 0
    history: list[dict] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum, n_steps = 0.0, 0
        for step, batch in enumerate(train_loader):
            if args.max_steps > 0 and step >= args.max_steps:
                break
            batch = to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            _, loss = model(**batch)
            loss.backward()
            clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step(); scheduler.step()
            loss_sum += float(loss.item()); n_steps += 1
            if step % 50 == 0:
                print(f"    ep{epoch} step{step:4d} loss={loss.item():.4f} "
                      f"lr={scheduler.get_last_lr()[0]:.2e}", flush=True)
        train_loss = loss_sum / max(1, n_steps)
        val_loss, val_f1 = evaluate_loss_f1(model, val_loader, scheme.id2label,
                                            device, args.eval_max_batches)
        rec = {"epoch": epoch, "train_loss": round(train_loss, 6),
               "val_loss": round(val_loss, 6), "val_entity_f1": round(val_f1, 6),
               "timestamp": datetime.now().isoformat(timespec="seconds")}
        history.append(rec)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"  [Epoch {epoch}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"val_entity_f1={val_f1:.4f}")
        if val_f1 > best_f1:
            best_f1, best_epoch = val_f1, epoch
            torch.save({"model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        "decoder": args.decoder, "use_crf": args.decoder == "crf",
                        "num_labels": scheme.num_labels, "label2id": scheme.label2id,
                        "id2label": scheme.id2label, "max_length": args.max_length,
                        "bert_path": bert_path, "tokenizer_path": tok_path,
                        "epoch": epoch, "seed": args.seed, "val_entity_f1": val_f1,
                        "args": vars(args)}, ckpt_path)
            print(f"    ★ 新最优 F1={val_f1:.4f} 已保存")

    # 在 val 上产出统一格式预测（供 eval/compare 用；test 只在 evaluate.py 跑一次）
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=False)["model_state_dict"])
    model.to(device); model.eval()
    pred_by_id: dict[str, list[dict]] = {}
    offset = 0
    with torch.no_grad():
        for batch in val_loader:
            batch_dev = to_device(batch, device)
            pred = model.decode(**batch_dev)
            # 按当前 batch 的 records 切片配对，不能传 val_ds.records 全量
            cur = val_ds.records[offset:offset + int(batch["labels"].shape[0])]
            pred_by_id.update(batch_to_pred_spans(batch, pred, scheme.id2label, cur))
            offset += int(batch["labels"].shape[0])
    write_predictions(pred_path, pred_by_id)
    val_metrics = evaluate(val_rows, pred_by_id, scheme)
    print(f"\n  内层验证最终实体 F1={val_metrics['f1']} | 最佳 epoch={best_epoch} "
          f"(F1={best_f1:.4f})")
    print(f"  预测已写 {pred_path}")

    summary = {"run_name": tag, "method": f"RoBERTa+{args.decoder}", "best_epoch": best_epoch,
               "best_val_entity_f1": best_f1, "val_metrics": val_metrics,
               "history": history, "checkpoint": str(ckpt_path), "pred_path": str(pred_path),
               "args": vars(args), "finished": datetime.now().isoformat(timespec="seconds")}
    out_path = log_dir / f"summary_{tag}.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  汇总已写 {out_path}")


if __name__ == "__main__":
    main()
