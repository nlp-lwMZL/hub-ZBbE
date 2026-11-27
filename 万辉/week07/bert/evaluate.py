"""
从 checkpoint 单独跑一遍 dev 推理 + 统一评测（不重新训练）

为什么要这个脚本：
  训练脚本在结尾会把「训练集预测」也算一遍（用于对照非法 BIO 数），
  对 CRF 来说这一步很贵：路径分数要对每条序列额外跑一遍前向算法，
  5173 句能吃掉十几分钟 GPU。时间预算紧张时，更划算的做法是
  **只对 dev 出正式指标**，训练集对照改用抽样（默认 800 句）近似。
  另外它也是「换 checkpoint / 复评」的标准入口，学生改完模型可以直接复评不用重训。

使用方式：
  python bert/evaluate.py --ckpt outputs/checkpoints/bert_crf_256.pt \
                              --run_name bert_crf_256
  python bert/evaluate.py --ckpt ... --train_sample 800   # 另算训练集抽样上的非法率

产物：outputs/disc/pred_<run_name>.jsonl、outputs/metrics/disc/<run_name>.json
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import random
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]   # 项目根目录（方法目录的上一级）
sys.path.insert(0, str(ROOT / "common"))     # 公共模块：event_data / crf / eval_utils / models / log_utils

from eval_utils import CKPT_DIR, Timer, evaluate_predictions
from log_utils import EvalLogger
from event_data import (BioDataset, DATA_DIR, ROOT, collate_bio, decode_bio_spans,
                        illegal_bio_stats, load_enums, load_extract_split,
                        make_pred_row, spans_to_event)
from model_paths import BERT_MODEL_DIR, resolve_model_path

def main():
    ap = argparse.ArgumentParser(description="从 checkpoint 复评（dev 推理 + 统一评测）")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--run_name", default="")
    ap.add_argument("--data_dir", default="")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--train_sample", type=int, default=0,
                    help="训练集抽样多少条来估非法率（0=不算）")
    ap.add_argument("--device", default="")
    args = ap.parse_args()

    from transformers import BertTokenizerFast
    from models import build_model, predict_char_tags

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt_path = Path(args.ckpt) if args.ckpt else CKPT_DIR / "bert_crf_256.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    label_scheme = ckpt["label_scheme"]
    id2label = ckpt["id2label"]
    train_args = ckpt["args"]

    class _A:
        pass

    a = _A()
    for k, v in train_args.items():
        setattr(a, k, v)
    a.hard_constraint = False       # 复评用软约束（和训练时一致）
    a.bert_path = train_args.get("bert_path") or str(resolve_model_path(BERT_MODEL_DIR))

    model = build_model(a, label_scheme).to(device)
    model.load_state_dict(ckpt["state_dict"])
    eval_logger = EvalLogger(method=args.run_name, args=args, root=ROOT,
                             notes="纯评测：从 checkpoint 复评 dev（不训练）")
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"加载 {ckpt_path.name}（epoch {ckpt.get('epoch')}，"
          f"内层验证 spanF1={ckpt.get('inner_val_span_f1', 0):.4f}，"
          f"解码器={train_args.get('decoder')}）")

    data_dir = Path(args.data_dir) if args.data_dir else DATA_DIR
    tokenizer = BertTokenizerFast.from_pretrained(a.bert_path)
    rows = load_extract_split(args.split, data_dir)
    enums = load_enums(data_dir / "event_schema.json")
    ds = BioDataset(rows, tokenizer, label_scheme, train_args.get("max_len", 256))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0,
                        collate_fn=collate_bio)

    t0 = time.perf_counter()
    with Timer() as timer:
        tags, _ = predict_char_tags(model, loader, id2label, device)
    illegal = illegal_bio_stats(tags)

    preds = []
    for row, tg in zip(rows, tags):
        ev = spans_to_event(decode_bio_spans(tg), row["text"], enums=enums)
        preds.append(make_pred_row(row, [ev] if ev else []))
    lat = timer.summary(len(rows))

    meta = {k: v for k, v in train_args.items()
            if k in ("decoder", "bert_layers", "max_len", "epochs", "lr", "batch_size",
                     "grad_accum", "head_lr_mult", "hard_constraint", "seed")}
    meta.update({
        "display_name": f"BERT-{train_args.get('bert_layers') or 12}L+"
                        f"{str(train_args.get('decoder', 'crf')).upper()}",
        "train_s": 0,      # 训练已在别处完成，这里只复评
        "params_m": round(n_params / 1e6, 1),
        "ms_per_sentence": lat["ms_per_sentence"],
        "best_epoch": ckpt.get("epoch"),
        "inner_val_span_f1": round(float(ckpt.get("inner_val_span_f1", 0)), 4),
        "illegal_dev": illegal,
        "n_truncated": ds.n_truncated,
        "note": "由 eval_bio_ckpt.py 从 checkpoint 复评（训练已在 train_bio.py 完成）",
    })

    # 训练集抽样上的非法率（便宜版对照，代替对全训练集做一次慢推理）
    if args.train_sample > 0:
        tr_rows = load_extract_split("train", data_dir)
        random.Random(train_args.get("seed", 42)).shuffle(tr_rows)
        sub = tr_rows[:args.train_sample]
        tr_ds = BioDataset(sub, tokenizer, label_scheme, train_args.get("max_len", 256))
        tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=False,
                               num_workers=0, collate_fn=collate_bio)
        tr_tags, _ = predict_char_tags(model, tr_loader, id2label, device)
        meta["illegal_train_pred_sample"] = illegal_bio_stats(tr_tags)
        meta["illegal_train_pred_sample_n"] = len(sub)
        print(f"训练集抽样 {len(sub)} 句的非法率："
              f"{meta['illegal_train_pred_sample']['illegal_seq_rate']:.1%}")

    name = args.run_name or ckpt_path.stem
    res = evaluate_predictions(name, preds, meta=meta,
                              gold_path=data_dir / f"{args.split}_extract.jsonl")
    print(f"\n非法统计（dev）：含非法序列 {illegal['illegal_seqs']} 条"
          f"（{illegal['illegal_seq_rate']:.1%}），"
          f"非法转移 {illegal['illegal_transition']} 处 / 非法开头 {illegal['illegal_start']} 处")
    eval_logger.finalize(metrics={k: res[k]["f1"] for k in ("trigger", "argument", "event_exact")},
                        type_macro_f1=res.get("type_macro_f1"), illegal_dev=illegal,
                        inference_latency_ms=lat["ms_per_sentence"], ckpt=str(ckpt_path))

    print(f"推理耗时 {time.perf_counter() - t0:.0f}s（{lat['ms_per_sentence']} ms/句）")

if __name__ == "__main__":
    main()
