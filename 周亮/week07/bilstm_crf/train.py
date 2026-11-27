"""
L1 训练：字符级 BiLSTM + Linear / CRF（NER，cdtier/clean，无预训练）。

用法（Colab / 本机 CPU 均可）：
  python bilstm_crf/train.py --decoder crf    --run_name bilstm_crf
  python bilstm_crf/train.py --decoder linear --run_name bilstm_linear
  python bilstm_crf/train.py --decoder crf --max_steps 20 --eval_max_batches 2 --num_train 40 --run_name smoke

【与 L2 的可比性纪律】
  - 标签体系、内层验证切分（seed=42, 10%）、统一评测函数与 L2 完全一致；
  - 字表只用训练集字符构建（用 val/test 建表 = 泄漏）；
  - 选模看实体级 F1，不看 loss（CE vs CRF NLL 量纲不同）；
  - 用 char collate 补到 batch 内最长，attention_mask = 连续前缀。
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
from torch.utils.data import DataLoader

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR / "bilstm_crf"))

from common import paths
from common.ner_data import IGNORE_INDEX, LabelScheme, CharDataset, make_char_vocab, load_records, collate_char
from common.metrics import evaluate, write_predictions
from model import build_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="L1 字符级 BiLSTM NER 训练")
    p.add_argument("--decoder", choices=["linear", "crf"], default="crf")
    p.add_argument("--run_name", default=None)
    p.add_argument("--emb_dim", type=int, default=128)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--num_layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--eval_batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--max_grad_norm", type=float, default=5.0)
    p.add_argument("--inner_val", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--max_steps", type=int, default=0)
    p.add_argument("--eval_max_batches", type=int, default=0)
    p.add_argument("--num_train", type=int, default=0)
    return p.parse_args()


def set_seed(s: int) -> None:
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        try:
            torch.ones(1, device="mps"); return torch.device("mps")
        except (RuntimeError, AssertionError):
            pass
    return torch.device("cpu")


def to_device(batch: dict, device: torch.device) -> dict:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def collect_sequences(labels: torch.Tensor, pred: torch.Tensor, id2label: dict[int, str]):
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


def bio_to_spans(tags: list[str]) -> list[dict]:
    spans, cur_type, cur_start = [], None, 0
    def close(end):
        nonlocal cur_type
        if cur_type is not None:
            spans.append({"type": cur_type, "start": cur_start, "end": end}); cur_type = None
    for i, t in enumerate(tags):
        if t == "O":
            close(i); continue
        pre, _, et = t.partition("-")
        if pre == "B":
            close(i); cur_type, cur_start = et, i
        elif cur_type != et:
            close(i); cur_type, cur_start = et, i
    close(len(tags))
    return spans


@torch.no_grad()
def eval_loss_f1(model, loader, id2label, device, max_batches):
    from seqeval.metrics import f1_score
    model.eval()
    total, n = 0.0, 0
    gs_all, ps_all = [], []
    for i, batch in enumerate(loader):
        if max_batches > 0 and i >= max_batches:
            break
        bdev = to_device(batch, device)
        _, loss = model(bdev["input_ids"], bdev["attention_mask"], bdev["labels"])
        pred = model.decode(bdev["input_ids"], bdev["attention_mask"], bdev["labels"])
        g, p = collect_sequences(batch["labels"], pred, id2label)
        gs_all.extend(g); ps_all.extend(p)
        total += float(loss.item()); n += 1
    return total / max(n, 1), float(f1_score(gs_all, ps_all)) if gs_all else 0.0


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    smoke = args.max_steps > 0 or args.eval_max_batches > 0 or args.num_train > 0
    tag = args.run_name or f"bilstm_{args.decoder}_{datetime.now():%m%d_%H%M%S}"
    out_root = paths.get_out_root() / ("smoke" if smoke else "")
    log_dir = out_root / "logs"; log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_root / "checkpoints" / f"{tag}.pt"; ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"train_{tag}.jsonl"

    data_dir = paths.get_data_dir()
    scheme = LabelScheme.from_json(data_dir / "label_names.json")
    records = load_records("train", data_dir)
    if args.num_train > 0:
        records = records[:args.num_train]
    rng = random.Random(args.seed)
    idx = list(range(len(records))); rng.shuffle(idx)
    n_val = max(1, int(len(records) * args.inner_val))
    vset = set(idx[:n_val])
    train_rows = [r for i, r in enumerate(records) if i not in vset]
    val_rows = [r for i, r in enumerate(records) if i in vset]
    print("=" * 72)
    print(f"L1 字符级 BiLSTM+{args.decoder} | device={device} | run={tag}")
    print(f"  训练 {len(train_rows)} / 内层验证 {len(val_rows)} 句（dev/test 未参与）")
    print("=" * 72)

    char2id = make_char_vocab(train_rows)     # 只用训练字符建表
    train_ds = CharDataset(train_rows, char2id, scheme, args.max_length)
    val_ds = CharDataset(val_rows, char2id, scheme, args.max_length, verbose=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_char)
    val_loader = DataLoader(val_ds, batch_size=args.eval_batch_size, shuffle=False,
                            collate_fn=collate_char)

    model = build_model(args.decoder == "crf", len(char2id), scheme.num_labels,
                        emb_dim=args.emb_dim, hidden=args.hidden,
                        num_layers=args.num_layers, dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameter_groups(args.lr), weight_decay=args.weight_decay)
    steps_per_ep = args.max_steps if args.max_steps > 0 else len(train_loader)
    total_steps = steps_per_ep * args.epochs
    warm = max(1, int(total_steps * 0.1))
    def lr_lambda(s):
        return float(s + 1) / warm if s < warm else max(0.0, 1.0 - (s - warm) / max(1, total_steps - warm))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_f1, best_epoch = -1.0, 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train(); loss_sum, n = 0.0, 0
        for step, batch in enumerate(train_loader):
            if args.max_steps > 0 and step >= args.max_steps:
                break
            bdev = to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            _, loss = model(bdev["input_ids"], bdev["attention_mask"], bdev["labels"])
            loss.backward()
            clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step(); scheduler.step()
            loss_sum += float(loss.item()); n += 1
        train_loss = loss_sum / max(1, n)
        val_loss, val_f1 = eval_loss_f1(model, val_loader, scheme.id2label, device, args.eval_max_batches)
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
                        "id2label": scheme.id2label, "char2id": char2id,
                        "emb_dim": args.emb_dim, "hidden": args.hidden,
                        "num_layers": args.num_layers, "dropout": args.dropout,
                        "max_length": args.max_length, "epoch": epoch, "seed": args.seed,
                        "val_entity_f1": val_f1, "args": vars(args)}, ckpt_path)
            print(f"    ★ 新最优 F1={val_f1:.4f} 已保存")

    # val 预测 -> 统一格式
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model_state_dict"])
    model.to(device); model.eval()
    pred_by_id: dict[str, list[dict]] = {}
    offset = 0
    with torch.no_grad():
        for batch in val_loader:
            bdev = to_device(batch, device)
            pred = model.decode(bdev["input_ids"], bdev["attention_mask"], bdev["labels"])
            _, preds = collect_sequences(batch["labels"], pred, scheme.id2label)
            # 关键：必须把「当前 batch 对应的 records 切片」和 preds 配对。
            # 不能用 val_ds.records（全量）zip preds（单 batch），否则 zip 以短的为准，
            # 只有第一个 batch 的 id 被写进 pred_by_id → 其余 gold 全部「缺预测」。
            cur = val_ds.records[offset:offset + len(preds)]
            assert len(cur) == len(preds), "batch 预测与 records 切片长度不一致"
            for rec, tags in zip(cur, preds):
                pred_by_id[rec["id"]] = [
                    {"type": s["type"], "start": s["start"], "end": s["end"],
                     "text": "".join(rec["tokens"][s["start"]:s["end"]])}
                    for s in bio_to_spans(tags)
                ]
            offset += len(preds)
    # 冒烟产物统一落 smoke 子目录，和 ckpt/summary 一致，不污染正式的 logs/、outputs/preds/
    pred_path = (out_root / "logs" if smoke else paths.pred_dir()) / f"{tag}_val.jsonl"
    write_predictions(pred_path, pred_by_id)
    val_metrics = evaluate(val_rows, pred_by_id, scheme)
    print(f"\n  内层验证实体 F1={val_metrics['f1']} | 最佳 epoch={best_epoch} (F1={best_f1:.4f})")
    summary = {"run_name": tag, "method": f"BiLSTM+{args.decoder}", "best_epoch": best_epoch,
               "best_val_entity_f1": best_f1, "val_metrics": val_metrics, "history": history,
               "checkpoint": str(ckpt_path), "pred_path": str(pred_path),
               "args": vars(args), "finished": datetime.now().isoformat(timespec="seconds")}
    sp = log_dir / f"summary_{tag}.json"
    sp.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  汇总 {sp}")


if __name__ == "__main__":
    main()
