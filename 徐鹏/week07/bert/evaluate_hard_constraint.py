"""
CRF 硬约束的「解码期」验证：在训练好的 BERT+CRF 上，把非法转移的分数直接置 -inf 再解码

为什么用「解码期」而不是重训一遍：
  问题是「硬约束能不能让非法序列归零」。重训一遍要 40+ 分钟（CRF 的 O(L·C²) 前向很慢），
  而**转移矩阵是模型的参数，解码时才用**——把训练好的转移矩阵里的非法项 mask 成 -inf，
  重新维特比解码，就能直接回答这个问题，成本只要一次 dev 推理（约 1 分钟）。
  这是标准的「post-hoc 约束 / 受约束解码」做法（生产里也这么做：模型不动，解码加约束）。

  需要说清的边界：这不是「带约束训练」的复现。带约束训练会改变学到的转移分布，
  F1 可能略高（模型不再浪费容量去学非法转移）也可能略低（少了一个正则项）。
  本脚本给的数字只回答「非法序列能否严格归零」以及「F1 会不会因此明显变差」。

对比口径：软约束（训练时原样） vs 硬约束（解码期 mask）——**同一个 checkpoint、同一份 dev**，
所以两组数字唯一的差别就是解码约束，归因很干净。

使用方式：
  python bert/evaluate_hard_constraint.py                      # 默认用 bert_crf_256
  python bert/evaluate_hard_constraint.py --ckpt outputs/checkpoints/bert_crf_256.pt

产物：
  outputs/metrics/disc/bert_crf_256_hard.json    （解码期硬约束的指标，与软约束同口径）
  outputs/disc/pred_bert_crf_256_hard.jsonl
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]   # 项目根目录（方法目录的上一级）
sys.path.insert(0, str(ROOT / "common"))     # 公共模块：event_data / crf / eval_utils / models / log_utils

from eval_utils import CKPT_DIR, Timer, evaluate_predictions
from log_utils import EvalLogger
from event_data import (BioDataset, DATA_DIR, build_bio_constraint_mask,
                        build_start_constraint_mask, collate_bio, decode_bio_spans,
                        illegal_bio_stats, load_enums, load_extract_split, load_schema,
                        make_pred_row, spans_to_event)
from model_paths import BERT_MODEL_DIR, resolve_model_path

def main():
    ap = argparse.ArgumentParser(description="CRF 硬约束（解码期）对比")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--run_name", default="bert_crf_256_hard")
    ap.add_argument("--soft_json", default="bert_crf_256",
                    help="软约束那一组的结果名，用于自动生成对照说明")
    ap.add_argument("--device", default="")
    args = ap.parse_args()

    from transformers import BertTokenizer
    from models import build_model, predict_char_tags

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt_path = Path(args.ckpt) if args.ckpt else CKPT_DIR / "bert_crf_256.pt"
    if not ckpt_path.exists():
        print(f"找不到 checkpoint：{ckpt_path}（请先跑 python bert/train.py --decoder crf）")
        return

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    train_args = ckpt["args"]

    class _A:   # 把 checkpoint 里的超参还原成 build_model 需要的参数对象
        pass

    a = _A()
    for k, v in train_args.items():
        setattr(a, k, v)
    a.bert_layers = train_args.get("bert_layers", 0)
    a.hard_constraint = True          # ← 唯一的区别：解码时施加硬约束
    a.decoder = "crf"
    a.bert_path = train_args.get("bert_path") or str(resolve_model_path(BERT_MODEL_DIR))
    a.dropout = train_args.get("dropout", 0.1)

    label_scheme = ckpt["label_scheme"]
    id2label = ckpt["id2label"]
    model = build_model(a, label_scheme).to(device)
    model.load_state_dict(ckpt["state_dict"])
    # ⚠ 关键坑：`constraint` / `start_constraint` 是 **buffer（persistent=True）**，
    # 会跟着 checkpoint 一起存。checkpoint 是「软约束」训出来的（两个 buffer 全是 True），
    # 所以 `load_state_dict` 会把我们刚在构造时装的硬约束**覆盖回全 True** ——
    # 表面看代码加了约束，实际一个都没生效（本项目实测踩到：日志打「禁止 0 个非法转移对」）。
    # 修法：load 完之后把约束 buffer 再装一遍。
    model.crf.constraint = build_bio_constraint_mask(label_scheme).to(device)
    model.crf.start_constraint = build_start_constraint_mask(label_scheme).to(device)
    eval_logger = EvalLogger(method=args.run_name, args=args, root=ROOT,
                             notes="纯评测：从 checkpoint 复评 dev（不训练）")
    model.eval()

    # 防御性检查：确认 constraint buffer 真的生效了（避免「以为加了约束其实没加」）
    n_forbidden = int((~model.crf.constraint).sum())
    n_bad_start = int((~model.crf.start_constraint).sum())
    print(f"硬约束已装载：禁止 {n_forbidden} 个非法转移对、{n_bad_start} 个非法起始标签")
    if n_forbidden == 0:
        raise RuntimeError("硬约束没有生效（禁止 0 个转移对），结果会和软约束完全一样，"
                           "请检查 constraint buffer 是否被 load_state_dict 覆盖")

    data_dir = DATA_DIR
    tokenizer = BertTokenizer.from_pretrained(a.bert_path)
    dev_rows = load_extract_split("dev", data_dir)
    enums = load_enums()
    dev_ds = BioDataset(dev_rows, tokenizer, label_scheme, train_args.get("max_len", 256))
    dev_loader = DataLoader(dev_ds, batch_size=32, shuffle=False, num_workers=0,
                            collate_fn=collate_bio)

    with Timer() as timer:
        tags, _ = predict_char_tags(model, dev_loader, id2label, device)
    illegal = illegal_bio_stats(tags)
    print(f"硬约束解码后非法统计：非法开头 {illegal['illegal_start']} 处 / "
          f"非法转移 {illegal['illegal_transition']} 处（含非法序列 {illegal['illegal_seqs']} 句）")

    preds = []
    for row, tg in zip(dev_rows, tags):
        ev = spans_to_event(decode_bio_spans(tg), row["text"], enums=enums)
        preds.append(make_pred_row(row, [ev] if ev else []))
    lat = timer.summary(len(dev_rows))

    meta = {
        "display_name": "BERT+CRF（解码期硬约束）",
        "train_s": 0,     # 复用已有 checkpoint，无额外训练
        "ms_per_sentence": lat["ms_per_sentence"],
        "decoder": "crf-hard",
        "max_len": train_args.get("max_len", 256),
        "illegal_dev": illegal,
        "base_ckpt": str(ckpt_path),
        "constraint_note": ("解码期硬约束：训练好的转移矩阵里非法项 mask 成 -1e4 后重新维特比解码；"
                            "不是带约束重新训练"),
        "note": "与 bert_crf_256（软约束）同 checkpoint、同 dev，唯一差别是解码是否硬约束",
    }
    res = evaluate_predictions(args.run_name, preds, meta=meta,
                              gold_path=data_dir / "dev_extract.jsonl")

    # 和软约束那组自动对照（读不到就跳过，不硬编码数字）
    import json
    soft_path = ROOT / "outputs" / "metrics" / "disc" / f"{args.soft_json}.json"
    if soft_path.exists():
        soft = json.loads(soft_path.read_text(encoding="utf-8"))
        si = (soft.get("meta") or {}).get("illegal_dev") or {}
        print("\n【软约束 vs 硬约束（同 checkpoint / 同 dev）】")
        print(f"  非法序列：{si.get('illegal_seqs', '?')} 句 → {illegal['illegal_seqs']} 句")
        print(f"  非法转移：{si.get('illegal_transition', '?')} 处 → {illegal['illegal_transition']} 处")
        print(f"  论元 F1 ：{soft['argument']['f1']:.4f} → {res['argument']['f1']:.4f}")
        eval_logger.finalize(metrics={k: res[k]["f1"] for k in ("trigger", "argument", "event_exact")},
                        illegal_dev=illegal, ckpt=str(ckpt_path),
                        constraint_note="解码期硬约束（token 级非法归零）")

    print(f"  触发词 F1：{soft['trigger']['f1']:.4f} → {res['trigger']['f1']:.4f}")

if __name__ == "__main__":
    main()
