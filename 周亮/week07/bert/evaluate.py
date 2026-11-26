"""
L2 评测：从 checkpoint 在指定 split（默认 test）上复评，输出统一格式预测。

【口径纪律】
1. checkpoint 元信息（decoder / 底座 / 标签 / max_length）全部从 ckpt 读，不靠命令行猜；
2. test 只在最终评测跑一次；调参用 validation；
3. CRF 走 model.decode() 维特比，不对 emissions 直接 argmax；
4. pred/gold 都按 labels != -100 同一口径过滤；预测写 common/metrics 的 pred_spans JSONL；
5. loss 量纲 CE vs CRF NLL 不同，跨模型只看实体 F1。

用法：
  python bert/evaluate.py --checkpoint <ckpt.pt> --split test
  python bert/evaluate.py --run_name roberta_crf        # 自动找 checkpoints/<run_name>.pt
本机冒烟：
  python bert/evaluate.py --checkpoint <smoke ckpt> --split validation --max_batches 2
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR / "bert"))

from common import paths
from common.ner_data import IGNORE_INDEX, DataConfig, LabelScheme, NERDataset
from common.metrics import evaluate, write_predictions
from model import build_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="L2 RoBERTa NER 评测")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--run_name", default=None, help="不给 checkpoint 时用 run_name 定位")
    p.add_argument("--split", default="test", choices=["train", "validation", "test"])
    p.add_argument("--eval_batch_size", type=int, default=32)
    p.add_argument("--max_batches", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--sliding_window", action="store_true",
                   help="对超长样本用带重叠的滑动窗口推理（解决长句尾部实体被截断丢失）")
    p.add_argument("--window_overlap", type=int, default=32, help="滑窗重叠字符数")
    p.add_argument("--dump_raw_tags", action="store_true",
                   help="导出模型原始 BIO 序列到 preds/<run>_<split>_rawtags.jsonl，"
                        "用于精确统计非法 BIO 转移（Linear vs CRF 的核心对比点）。"
                        "注意：必须用 decode 前的原始输出，不能用兜底后的 span 反推。")
    return p.parse_args()


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
def predict_long_record_sliding(rec: dict, model, tok, scheme, id2label, device,
                                max_length: int, overlap: int) -> list[dict]:
    """对一条超长样本做滑窗推理，返回合并后的**全局字符坐标** span 列表。"""
    from torch.utils.data import DataLoader
    from functools import partial
    from common.ner_data import NERDataset, pad_collate
    from common.sliding_window import make_windows, merge_spans

    tokens = rec["tokens"]
    win_size = max_length - 2                      # 给 CLS/SEP 留位
    windows = make_windows(len(tokens), win_size, overlap)
    cfg = DataConfig(tokenizer_name_or_path=tok.name_or_path, max_length=max_length)
    spans_all: list[dict] = []
    for (s, e) in windows:
        sub = {"id": rec["id"], "tokens": tokens[s:e],
               "ner_tags": ["O"] * (e - s), "text": "".join(tokens[s:e])}
        ds_win = NERDataset([sub], tok, scheme, cfg, verbose=False)
        loader_win = DataLoader(ds_win, batch_size=1, shuffle=False,
                                collate_fn=partial(pad_collate, pad_token_id=int(tok.pad_token_id),
                                                   label_pad_id=IGNORE_INDEX,
                                                   return_token_type_ids=False))
        batch = next(iter(loader_win))
        bdev = to_device(batch, device)
        pred = model.decode(**bdev)
        _, preds = collect_sequences(batch["labels"], pred, id2label)
        for sp in bio_to_spans(preds[0]):
            gs, ge = s + sp["start"], s + sp["end"]
            spans_all.append({"type": sp["type"], "start": gs, "end": ge,
                              "text": "".join(tokens[gs:ge])})
    return merge_spans(spans_all)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    # 1) 定位 checkpoint
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    elif args.run_name:
        ckpt_path = paths.get_out_root() / "checkpoints" / f"{args.run_name}.pt"
    else:
        raise ValueError("需要 --checkpoint 或 --run_name")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"找不到 checkpoint：{ckpt_path}")

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    required = {"model_state_dict", "decoder", "num_labels", "label2id", "id2label",
                "max_length", "bert_path"}
    missing = required - set(ck.keys())
    if missing:
        raise KeyError(f"checkpoint 缺元信息：{sorted(missing)}")
    tag = ck.get("run_name") or ckpt_path.stem
    print("=" * 72)
    print(f"L2 评测 | decoder={ck['decoder']} | split={args.split} | device={device}")
    print(f"  ckpt={ckpt_path} (epoch={ck.get('epoch')}, val_F1={ck.get('val_entity_f1')})")
    if args.split == "test":
        print("  [提醒] test 只在最终评测跑一次")
    print("=" * 72)

    # 2) 数据 + tokenizer（严格用 ckpt 元信息）
    data_dir = paths.get_data_dir()
    scheme = LabelScheme.from_json(data_dir / "label_names.json")
    if scheme.label2id != ck["label2id"]:
        raise ValueError("label_names.json 与 checkpoint 的 label2id 不一致，id 顺序变了评估会失真")
    id2label = {int(k): v for k, v in ck["id2label"].items()}
    records = json.loads((data_dir / f"{args.split}.json").read_text(encoding="utf-8"))
    cfg = DataConfig(data_dir=data_dir, tokenizer_name_or_path=ck.get("tokenizer_path") or ck["bert_path"],
                     max_length=int(ck["max_length"]))
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.tokenizer_name_or_path, use_fast=True)
    ds = NERDataset(records, tok, scheme, cfg, verbose=False)
    from torch.utils.data import DataLoader
    from common.ner_data import pad_collate
    from functools import partial
    loader = DataLoader(ds, batch_size=args.eval_batch_size, shuffle=False,
                        collate_fn=partial(pad_collate, pad_token_id=int(tok.pad_token_id),
                                           label_pad_id=IGNORE_INDEX,
                                           return_token_type_ids=False))

    # 3) 重建模型 + 加载权重
    model = build_model(ck["decoder"] == "crf", ck["bert_path"], int(ck["num_labels"]),
                        float(ck.get("args", {}).get("dropout", 0.1)))
    model.load_state_dict(ck["model_state_dict"])
    model.to(device); model.eval()

    # 4) 推理 + 统一评测
    pred_by_id: dict[str, list[dict]] = {}
    raw_tags_by_id: dict[str, list[str]] = {}   # 原始 BIO（兜底前），仅 --dump_raw_tags 时写盘
    total_loss, n_batches = 0.0, 0
    offset = 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if args.max_batches > 0 and i >= args.max_batches:
                break
            bdev = to_device(batch, device)
            _, loss = model(**bdev)
            pred = model.decode(**bdev)
            golds, preds = collect_sequences(batch["labels"], pred, id2label)
            # 必须按当前 batch 的 records 切片配对，不能 zip 全量 ds.records
            cur = ds.records[offset:offset + len(preds)]
            assert len(cur) == len(preds), "batch 预测与 records 切片长度不一致"
            for rec, tags in zip(cur, preds):
                if args.dump_raw_tags:
                    raw_tags_by_id[rec["id"]] = list(tags)
                spans = []
                for sp in bio_to_spans(tags):
                    s, e = sp["start"], sp["end"]
                    spans.append({"type": sp["type"], "start": s, "end": e,
                                  "text": "".join(rec["tokens"][s:e])})
                pred_by_id[rec["id"]] = spans
            offset += len(preds)
            total_loss += float(loss.item()); n_batches += 1

    # 冒烟只跑了前 max_batches 个 batch → gold 也截到同样范围，否则缺预测会被
    # metrics.evaluate 的防御检查拦下（这正是「删行让分母变小、F1 虚高」的那道防线）。
    eval_records = records
    if args.max_batches > 0:
        eval_records = records[:len(pred_by_id)]

    # 长句滑窗：对超长样本重新推理（常规路径里它们被截断了），覆盖掉截断版的预测
    if args.sliding_window:
        max_len = int(ck["max_length"])
        long_recs = [r for r in eval_records if len(r["tokens"]) > max_len - 2]
        print(f"  [滑窗] 检测到 {len(long_recs)} 条超长样本（> {max_len-2} 字符），逐条滑窗重推 ...")
        for j, rec in enumerate(long_recs):
            pred_by_id[rec["id"]] = predict_long_record_sliding(
                rec, model, tok, scheme, id2label, device, max_len, args.window_overlap)
            if (j + 1) % 20 == 0:
                print(f"    滑窗 {j+1}/{len(long_recs)}", flush=True)

    metrics = evaluate(eval_records, pred_by_id, scheme)
    out_dir = paths.eval_dir() if args.max_batches == 0 else paths.log_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = paths.pred_dir() / f"{tag}_{args.split}.jsonl"
    write_predictions(pred_path, pred_by_id)
    if args.dump_raw_tags:
        raw_path = paths.pred_dir() / f"{tag}_{args.split}_rawtags.jsonl"
        with raw_path.open("w", encoding="utf-8") as f:
            for rid, tags in raw_tags_by_id.items():
                f.write(json.dumps({"id": rid, "bio": tags}, ensure_ascii=False) + "\n")
        print(f"  [原始BIO] 已导出 {len(raw_tags_by_id)} 条 → {raw_path}")
    result = {"run_name": tag, "method": f"RoBERTa+{ck['decoder']}", "split": args.split,
              "num_samples": len(eval_records), "avg_loss": round(total_loss / max(1, n_batches), 6),
              "metrics": metrics, "checkpoint": str(ckpt_path), "pred_path": str(pred_path),
              "timestamp": datetime.now().isoformat(timespec="seconds")}
    res_path = out_dir / f"{tag}_{args.split}.json"
    res_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  实体 P={metrics['precision']} R={metrics['recall']} F1={metrics['f1']} "
          f"| type_macro_F1={metrics['type_macro_f1']}")
    print(f"  预测: {pred_path}\n  指标: {res_path}")


if __name__ == "__main__":
    main()
