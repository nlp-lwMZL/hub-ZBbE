# -*- coding: utf-8 -*-
"""
评估脚本：加载训练好的 checkpoint，在验证集/测试集上算 entity-level 指标 + 错误分析。

产出：
  outputs/logs/<run_name>_eval.json        两个 split 的 P/R/F1、逐类型指标、非法 BIO、截断统计、错误分类
  outputs/logs/<run_name>_eval_console.log 完整 stdout
  outputs/predictions/<split>_predictions.jsonl  逐条预测（文本 + 金标 span + 预测 span）

为什么评估要单独写一个脚本：
  训练时每个 epoch 也要评估（用于选最优 checkpoint），但那只算一个 F1；
  真正写报告需要逐类型指标、错误类型分布、bad case，这些放在独立脚本里更清楚，
  也方便学生只重跑评估、不重训。

运行：
  python src/evaluate.py --bert_path <bert-base-chinese 目录> --run_name bert_bio_128
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import time
from collections import Counter

import torch
from seqeval.metrics import classification_report as seqeval_report
from seqeval.metrics import f1_score, precision_score, recall_score
from transformers import AutoModelForTokenClassification, AutoTokenizer

from bio_labels import (
    bio_to_spans,
    build_label_maps,
    check_bio,
    error_breakdown,
    load_label_names,
    spans_exact_equal,
)
from dataset import build_dataloaders
from log_utils import EvalLogger
from paths import (
    CKPT_DIR,
    PRED_DIR,
    PROJECT_ROOT,
    ensure_output_dirs,
    resolve_bert_path,
    resolve_data_dir,
)


@torch.no_grad()
def run_inference(model, loader, id2label: dict, device: torch.device) -> list[dict]:
    """
    在 loader 上逐条推理，返回「句子级」结果。

    每条结果同时保留两种视图：
      - token_gold / token_pred：只含参与 loss 的 token（子词对齐后的有效标签），
        用于 seqeval 计算 entity-level P/R/F1（和训练时验证口径完全一致）。
      - gold_tags / pred_tags：映射回原句每个字符的 BIO 标签，
        用于还原 span、写预测文件、做 bad case 展示。

    为什么两种都要：
      token 视图是「模型真正看到/预测的」；字符视图是「人能读的」。
      只留一种，要么指标口径对不上训练，要么 bad case 还原不回去。
    """
    model.eval()
    results: list[dict] = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,   # transformers 5.x 默认返回 tuple，必须显式要 dict
        )
        pred_ids = outputs.logits.argmax(dim=-1).cpu().tolist()
        label_ids = labels.cpu().tolist()
        word_ids = batch["word_ids"].tolist()
        char_lens = batch["char_len"].tolist()
        kept_chars = batch["kept_chars"].tolist()
        truncated = batch["truncated"].tolist()
        indices = batch["index"].tolist()

        for i in range(len(input_ids)):
            idx = int(indices[i])
            row = loader.dataset.records[idx]
            chars = row["tokens"]
            char_len = int(char_lens[i])
            kept = int(kept_chars[i])

            gold_tags = ["O"] * char_len
            pred_tags = ["O"] * char_len
            token_gold: list[str] = []
            token_pred: list[str] = []

            for wid, gold_id, pred_id in zip(word_ids[i], label_ids[i], pred_ids[i]):
                if gold_id == -100:
                    continue
                gold_tag = id2label[gold_id]
                pred_tag = id2label[pred_id]
                token_gold.append(gold_tag)
                token_pred.append(pred_tag)
                if wid >= 0:
                    gold_tags[wid] = gold_tag
                    pred_tags[wid] = pred_tag

            gold_spans = bio_to_spans(chars, gold_tags)
            pred_spans = bio_to_spans(chars, pred_tags)

            # 截断损失：用「完整金标」统计有多少实体完全掉尾 / 跨边界，
            # 不要用 gold_tags（它已经被截断成 O，统计不出来）
            full_gold_spans = bio_to_spans(chars, row["ner_tags"])
            dropped = sum(1 for s in full_gold_spans if s["start"] >= kept)
            partial = sum(1 for s in full_gold_spans if s["start"] < kept < s["end"])

            results.append({
                "index": idx,
                "text": "".join(chars),
                "char_len": char_len,
                "kept_chars": kept,
                "truncated": bool(truncated[i]),
                "dropped_entities": dropped,
                "partial_entities": partial,
                "token_gold": token_gold,
                "token_pred": token_pred,
                "gold_tags": gold_tags,
                "pred_tags": pred_tags,
                "gold_spans": gold_spans,
                "pred_spans": pred_spans,
            })

    return results


def summarize(results: list[dict]) -> dict:
    """把逐条结果聚合成一份可直接写进报告的指标字典。"""
    all_golds = [r["token_gold"] for r in results]
    all_preds = [r["token_pred"] for r in results]

    p = float(precision_score(all_golds, all_preds))
    r = float(recall_score(all_golds, all_preds))
    f1 = float(f1_score(all_golds, all_preds))
    per_type = seqeval_report(all_golds, all_preds, digits=4, output_dict=True)

    illegal_seqs = 0
    illegal_transitions = 0
    for seq in all_preds:
        bad = check_bio(seq)
        illegal_seqs += bad["n_sequences_with_illegal"]
        illegal_transitions += bad["illegal_start"] + bad["illegal_transition"]

    n_truncated = sum(1 for x in results if x["truncated"])
    dropped_entities = sum(x["dropped_entities"] for x in results)
    partial_entities = sum(x["partial_entities"] for x in results)

    # 句子级 exact match 只在「未被截断」的句子上算：
    # 截断句的尾巴实体根本不在输入里，算进去只会把指标压低，且与模型能力无关。
    complete = [x for x in results if not x["truncated"]]
    exact = sum(1 for x in complete if spans_exact_equal(x["gold_spans"], x["pred_spans"]))

    err_counts = Counter()
    err_examples = []
    for x in complete:
        if spans_exact_equal(x["gold_spans"], x["pred_spans"]):
            continue
        br = error_breakdown(x["gold_spans"], x["pred_spans"])
        err_counts.update(br["counts"])
        for e in br["examples"]:
            if len(err_examples) < 30:
                err_examples.append({
                    "text": x["text"],
                    "kind": e["kind"],
                    "gold": e["gold"],
                    "pred": e["pred"],
                })

    return {
        "n_sentences": len(results),
        "precision": round(p, 6),
        "recall": round(r, 6),
        "f1": round(f1, 6),
        "per_type": per_type,
        "illegal": {
            "sequences": illegal_seqs,
            "transitions": illegal_transitions,
            "rate": round(illegal_seqs / max(len(results), 1), 4),
        },
        "truncation": {
            "n_truncated": n_truncated,
            "ratio": round(n_truncated / max(len(results), 1), 4),
            "dropped_entities": dropped_entities,
            "partial_entities": partial_entities,
        },
        "sentence_exact_match": {
            "n_complete_sentences": len(complete),
            "n_exact": exact,
            "rate": round(exact / max(len(complete), 1), 4),
        },
        "error_breakdown": {
            "counts": dict(err_counts),
            "examples": err_examples,
        },
    }


def print_report(split: str, metrics: dict, n_badcases: int, results: list[dict]) -> None:
    """把指标 + bad case 打到控制台（同时被 EvalLogger 写进 console.log）。"""
    print("\n" + "=" * 78)
    print(f"[{split}] 评估结果")
    print("=" * 78)
    print(f"句子数：{metrics['n_sentences']}")
    print(f"Entity-level Precision：{metrics['precision']:.4f}")
    print(f"Entity-level Recall：   {metrics['recall']:.4f}")
    print(f"Entity-level F1：       {metrics['f1']:.4f}")
    print(f"句子级 exact match：{metrics['sentence_exact_match']['n_exact']} / "
          f"{metrics['sentence_exact_match']['n_complete_sentences']} "
          f"= {metrics['sentence_exact_match']['rate']:.2%}（仅未截断句）")
    print(f"截断：{metrics['truncation']['n_truncated']} 条 "
          f"({metrics['truncation']['ratio']:.2%})，丢 {metrics['truncation']['dropped_entities']} 个实体，"
          f"{metrics['truncation']['partial_entities']} 个跨边界")
    print(f"非法 BIO：{metrics['illegal']['sequences']} 条序列 "
          f"({metrics['illegal']['rate']:.2%})，{metrics['illegal']['transitions']} 次非法转移")

    print("\n【逐类型指标】")
    for name, m in metrics["per_type"].items():
        if name in {"micro avg", "macro avg", "weighted avg"}:
            continue
        if isinstance(m, dict):
            print(f"  {name:<6} P={m.get('precision', 0):.4f}  R={m.get('recall', 0):.4f}  "
                  f"F1={m.get('f1-score', 0):.4f}  support={m.get('support', 0)}")

    print("\n【错误类型分布】（仅未截断句）")
    print("  " + json.dumps(metrics["error_breakdown"]["counts"], ensure_ascii=False))

    bad = [x for x in results if not x["truncated"]
           and not spans_exact_equal(x["gold_spans"], x["pred_spans"])]
    print(f"\n【Bad cases】共 {len(bad)} 条，展示前 {min(n_badcases, len(bad))} 条：")
    for x in bad[:n_badcases]:
        print("-" * 78)
        print("原文：", x["text"][:160] + ("..." if len(x["text"]) > 160 else ""))
        print("金标：", [(s["type"], s["start"], s["end"], s["text"]) for s in x["gold_spans"]])
        print("预测：", [(s["type"], s["start"], s["end"], s["text"]) for s in x["pred_spans"]])
        br = error_breakdown(x["gold_spans"], x["pred_spans"])
        print("错误：", br["counts"])


def dump_predictions(split: str, results: list[dict]) -> str:
    """写逐条预测 JSONL，方便学生自己 grep / 统计 / 做可视化。"""
    path = PRED_DIR / f"{split}_predictions.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for x in results:
            f.write(json.dumps({
                "index": x["index"],
                "text": x["text"],
                "truncated": x["truncated"],
                "char_len": x["char_len"],
                "gold_spans": x["gold_spans"],
                "pred_spans": x["pred_spans"],
            }, ensure_ascii=False) + "\n")
    return str(path)


def main():
    args = parse_args()
    ensure_output_dirs()
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("指定了 --device cuda，但当前没有可用 GPU")
        device = torch.device("cuda")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 先装日志器再干活：这样「数据集规模 / checkpoint 信息 / 评估结果」都会进 console.log。
    # 如果等评估完再装，日志会缺前半段，学生拿到文件看不出用的是哪个 checkpoint。
    logger = EvalLogger(
        method=args.run_name,
        args=args,
        root=PROJECT_ROOT,
        console_name=f"{args.run_name}_eval",
        notes="evaluate.py：验证集 + 测试集，含逐类型指标/非法 BIO/截断/错误分类",
    )

    data_dir = resolve_data_dir(args.data_dir)
    bert_path = resolve_bert_path(args.bert_path)
    label_names = load_label_names(data_dir)
    label2id, id2label = build_label_maps(label_names)

    ckpt_path = CKPT_DIR / f"{args.run_name}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"找不到 checkpoint：{ckpt_path}\n"
            f"请先训练：python src/train.py --bert_path {bert_path} --run_name {args.run_name}"
        )

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if ckpt.get("label_names") and ckpt["label_names"] != label_names:
        raise ValueError(
            f"checkpoint 的标签体系与当前数据不一致：\n"
            f"  ckpt: {ckpt['label_names']}\n  data: {label_names}"
        )
    max_length = args.max_length or ckpt.get("max_length", 128)

    tokenizer = AutoTokenizer.from_pretrained(str(bert_path), use_fast=True)
    _, val_loader, test_loader = build_dataloaders(
        data_dir=data_dir,
        tokenizer=tokenizer,
        label2id=label2id,
        batch_size=args.batch_size,
        max_length=max_length,
        num_workers=args.num_workers,
        limit_eval=args.limit_eval,
    )

    model = AutoModelForTokenClassification.from_pretrained(str(bert_path), num_labels=len(label_names))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    print(f"加载 checkpoint：{ckpt_path}")
    print(f"  epoch={ckpt.get('epoch')}  best_val_F1={ckpt.get('val_f1'):.4f}  max_length={max_length}")

    splits = []
    if args.split in {"both", "validation"}:
        splits.append(("validation", val_loader))
    if args.split in {"both", "test"}:
        splits.append(("test", test_loader))

    t0 = time.perf_counter()
    all_metrics = {}
    for split, loader in splits:
        results = run_inference(model, loader, id2label, device)
        metrics = summarize(results)
        all_metrics[split] = metrics
        print_report(split, metrics, args.num_badcases, results)
        pred_path = dump_predictions(split, results)
        print(f"[predictions] {pred_path}")

    logger.finalize(metrics=all_metrics, runtime_s=time.perf_counter() - t0)


def parse_args():
    parser = argparse.ArgumentParser(description="评估 BERT+Linear NER（BIO）")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--bert_path", type=str, default=None)
    parser.add_argument("--run_name", type=str, default="bert_bio_128")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=0, help="0=沿用 checkpoint 里的 max_length")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--split", choices=["both", "validation", "test"], default="both")
    parser.add_argument("--limit_eval", type=int, default=0, help="只评前 N 条（冒烟用）")
    parser.add_argument("--num_badcases", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    main()
