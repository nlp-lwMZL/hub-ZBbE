"""
L1 评测：从 checkpoint 在指定 split（默认 test）复评，统一格式预测 + 统一评测。

用法：python bilstm_crf/evaluate.py --checkpoint <ckpt.pt> [--split test]
      python bilstm_crf/evaluate.py --run_name bilstm_crf
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR / "bilstm_crf"))

from common import paths
from common.ner_data import IGNORE_INDEX, LabelScheme, CharDataset, load_records, collate_char
from common.metrics import evaluate, write_predictions
from model import build_model


def parse_args():
    p = argparse.ArgumentParser(description="L1 BiLSTM NER 评测")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--run_name", default=None)
    p.add_argument("--split", default="test", choices=["train", "validation", "test"])
    p.add_argument("--eval_batch_size", type=int, default=64)
    p.add_argument("--device", default="auto")
    p.add_argument("--num_train", type=int, default=0, help="冒烟时只评前 N 条")
    p.add_argument("--sliding_window", action="store_true",
                   help="对超长样本用带重叠的滑动窗口推理")
    p.add_argument("--window_overlap", type=int, default=32, help="滑窗重叠字符数")
    p.add_argument("--dump_raw_tags", action="store_true",
                   help="导出模型原始 BIO 序列到 preds/<run>_<split>_rawtags.jsonl，"
                        "用于精确统计非法 BIO 转移（Linear vs CRF 的核心对比点）。")
    return p.parse_args()


def resolve_device(name):
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


def collect_sequences(labels, pred, id2label):
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


def bio_to_spans(tags):
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
def predict_long_record_sliding(rec: dict, model, char2id: dict, scheme, id2label, device,
                                max_length: int, overlap: int) -> list[dict]:
    """对一条超长样本做滑窗推理，返回合并后的全局字符坐标 span。CharDataset 直接吃字符切片。"""
    from torch.utils.data import DataLoader
    from common.sliding_window import make_windows, merge_spans

    tokens = rec["tokens"]
    win_size = max_length - 2
    windows = make_windows(len(tokens), win_size, overlap)
    spans_all: list[dict] = []
    for (s, e) in windows:
        sub = {"id": rec["id"], "tokens": tokens[s:e],
               "ner_tags": ["O"] * (e - s), "text": "".join(tokens[s:e])}
        ds_win = CharDataset([sub], char2id, scheme, max_length, verbose=False)
        loader_win = DataLoader(ds_win, batch_size=1, shuffle=False, collate_fn=collate_char)
        batch = next(iter(loader_win))
        bdev = {k: v.to(device) for k, v in batch.items()}
        pred = model.decode(bdev["input_ids"], bdev["attention_mask"], bdev["labels"])
        _, preds = collect_sequences(batch["labels"], pred, id2label)
        for sp in bio_to_spans(preds[0]):
            gs, ge = s + sp["start"], s + sp["end"]
            spans_all.append({"type": sp["type"], "start": gs, "end": ge,
                              "text": "".join(tokens[gs:ge])})
    return merge_spans(spans_all)


def main():
    args = parse_args()
    device = resolve_device(args.device)
    ckpt_path = Path(args.checkpoint) if args.checkpoint else (
        paths.get_out_root() / "checkpoints" / f"{args.run_name}.pt")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"找不到 checkpoint：{ckpt_path}")
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    tag = ck.get("run_name") or ckpt_path.stem
    data_dir = paths.get_data_dir()
    scheme = LabelScheme.from_json(data_dir / "label_names.json")
    if scheme.label2id != ck["label2id"]:
        raise ValueError("标签 id 顺序与 checkpoint 不一致")
    records = load_records(args.split, data_dir)
    if args.num_train > 0:
        records = records[:args.num_train]
    ds = CharDataset(records, ck["char2id"], scheme, int(ck["max_length"]), verbose=False)
    loader = DataLoader(ds, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate_char)

    model = build_model(ck["decoder"] == "crf", len(ck["char2id"]), int(ck["num_labels"]),
                        emb_dim=int(ck["emb_dim"]), hidden=int(ck["hidden"]),
                        num_layers=int(ck["num_layers"]), dropout=float(ck["dropout"]))
    model.load_state_dict(ck["model_state_dict"])
    model.to(device); model.eval()

    pred_by_id = {}
    raw_tags_by_id = {}
    offset = 0
    with torch.no_grad():
        for batch in loader:
            bdev = {k: v.to(device) for k, v in batch.items()}
            pred = model.decode(bdev["input_ids"], bdev["attention_mask"], bdev["labels"])
            _, preds = collect_sequences(batch["labels"], pred, scheme.id2label)
            # 必须按当前 batch 的 records 切片配对，否则只有第一个 batch 的 id 被写入
            cur = ds.records[offset:offset + len(preds)]
            assert len(cur) == len(preds), "batch 预测与 records 切片长度不一致"
            for rec, tags in zip(cur, preds):
                if args.dump_raw_tags:
                    raw_tags_by_id[rec["id"]] = list(tags)
                pred_by_id[rec["id"]] = [
                    {"type": s["type"], "start": s["start"], "end": s["end"],
                     "text": "".join(rec["tokens"][s["start"]:s["end"]])}
                    for s in bio_to_spans(tags)
                ]
            offset += len(preds)

    # 长句滑窗覆盖
    if args.sliding_window:
        max_len = int(ck["max_length"])
        long_recs = [r for r in records if len(r["tokens"]) > max_len - 2]
        print(f"  [滑窗] 检测到 {len(long_recs)} 条超长样本（> {max_len-2} 字符），逐条滑窗重推 ...")
        for j, rec in enumerate(long_recs):
            pred_by_id[rec["id"]] = predict_long_record_sliding(
                rec, model, ck["char2id"], scheme, scheme.id2label,
                device, max_len, args.window_overlap)
            if (j + 1) % 20 == 0:
                print(f"    滑窗 {j+1}/{len(long_recs)}", flush=True)

    metrics = evaluate(records, pred_by_id, scheme)
    pred_path = paths.pred_dir() / f"{tag}_{args.split}.jsonl"
    write_predictions(pred_path, pred_by_id)
    if args.dump_raw_tags:
        raw_path = paths.pred_dir() / f"{tag}_{args.split}_rawtags.jsonl"
        with raw_path.open("w", encoding="utf-8") as f:
            for rid, tags in raw_tags_by_id.items():
                f.write(json.dumps({"id": rid, "bio": tags}, ensure_ascii=False) + "\n")
        print(f"  [原始BIO] 已导出 {len(raw_tags_by_id)} 条 → {raw_path}")
    res = {"run_name": tag, "method": f"BiLSTM+{ck['decoder']}", "split": args.split,
           "num_samples": len(records), "metrics": metrics, "checkpoint": str(ckpt_path),
           "pred_path": str(pred_path), "timestamp": datetime.now().isoformat(timespec="seconds")}
    out = paths.eval_dir() / f"{tag}_{args.split}.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"实体 P={metrics['precision']} R={metrics['recall']} F1={metrics['f1']} "
          f"| type_macro={metrics['type_macro_f1']}")
    print(f"  预测 {pred_path}\n  指标 {out}")


if __name__ == "__main__":
    main()
