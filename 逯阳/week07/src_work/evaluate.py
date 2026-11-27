"""
人民日报 NER 评估脚本（作业用）

和 cluener 版的差异：
  1. 数据换成 data/peoples_daily，7 个标签
  2. 输出到 outputs_work/logs/work_eval_{linear|crf}_{split}.json
  3. test 集**带标签**，所以 --split test 是有意义的（cluener 的 test 是空的）
  4. 结果 JSON 里额外存了逐类型 P/R/F1（classification_report 的 dict 形式），
     方便后面 compare_results.py 直接读，不用再解析文本表格

评估口径：
  - seqeval entity-level（类型 + 边界都完全一致才算 TP）
  - 非法 BIO 序列统计：illegal_start（I-X 开头）+ illegal_transition（B-X/I-X → I-Y, X≠Y）

使用方式：
  python src_work/evaluate.py                      # BERT+Linear，验证集
  python src_work/evaluate.py --use_crf            # BERT+CRF，验证集
  python src_work/evaluate.py --split test         # 在测试集上评估
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import argparse
from pathlib import Path

import torch
from transformers import BertTokenizer
from seqeval.metrics import (
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)

from dataset import build_label_schema, build_dataloaders, entity_types
from model import build_model

ROOT = Path(__file__).parent.parent
# 注意：ROOT = 序列标注项目/，所以是 ROOT.parent（= D:/project），不是 parent.parent
BERT_PATH = ROOT.parent / "pretrain_models" / "bert-base-chinese"
DATA_DIR = ROOT / "data" / "peoples_daily"

WORK_DIR = ROOT / "outputs_work"
CKPT_DIR = WORK_DIR / "checkpoints"
LOG_DIR = WORK_DIR / "logs"


def count_illegal_sequences(pred_seqs: list[list[str]]) -> dict:
    """统计非法 BIO 序列。

    非法类型：
      - illegal_start：序列以 I-X 开头
      - illegal_transition：I-Y 前面是 O，或前面是 B-X/I-X 但 X≠Y
    """
    stats = {"illegal_start": 0, "illegal_transition": 0, "total_seqs": len(pred_seqs)}
    for seq in pred_seqs:
        if not seq:
            continue
        if seq[0].startswith("I-"):
            stats["illegal_start"] += 1
        for i in range(1, len(seq)):
            prev, curr = seq[i - 1], seq[i]
            if curr.startswith("I-"):
                if prev == "O":
                    stats["illegal_transition"] += 1
                elif prev[2:] != curr[2:]:
                    stats["illegal_transition"] += 1

    stats["total_illegal"] = stats["illegal_start"] + stats["illegal_transition"]
    return stats


def run_inference(
    model,
    loader,
    id2label: dict,
    device: torch.device,
    use_crf: bool,
) -> tuple[list[list[str]], list[list[str]]]:
    """在 loader 上推理，返回 (all_preds, all_golds)，元素是字符串标签列表。"""
    model.eval()
    all_preds, all_golds = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["labels"].to(device)

            if use_crf:
                pred_ids_list = model.decode(input_ids, attention_mask, token_type_ids)
            else:
                logits, _ = model(input_ids, attention_mask, token_type_ids)
                pred_ids_list = logits.argmax(dim=-1).tolist()

            labels_list = labels.cpu().tolist()

            for i in range(len(input_ids)):
                gold_seq, pred_seq = [], []
                pred_ids = pred_ids_list[i]
                for j, gold_id in enumerate(labels_list[i]):
                    if gold_id == -100:
                        continue
                    gold_seq.append(id2label[gold_id])
                    pred_seq.append(
                        id2label.get(pred_ids[j], "O") if j < len(pred_ids) else "O"
                    )
                all_golds.append(gold_seq)
                all_preds.append(pred_seq)

    return all_preds, all_golds


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_tag = "crf" if args.use_crf else "linear"
    ckpt_path = CKPT_DIR / f"work_best_{run_tag}.pt"

    if not ckpt_path.exists():
        print(f"找不到 checkpoint：{ckpt_path}")
        print(f"请先运行：python src_work/train.py {'--use_crf' if args.use_crf else ''}")
        return

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    labels, label2id, id2label = build_label_schema(DATA_DIR)

    model = build_model(
        use_crf=args.use_crf,
        bert_path=str(args.bert_path),
        num_labels=len(labels),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    print(f"加载 checkpoint（epoch={ckpt['epoch']}，val_f1={ckpt['val_entity_f1']:.4f}）")

    tokenizer = BertTokenizer.from_pretrained(str(args.bert_path))
    max_length = args.max_length or ckpt["args"].get("max_length", 128)
    if max_length != ckpt["args"].get("max_length", 128):
        print(f"[注意] 评估用 max_length={max_length}，训练时是 "
              f"{ckpt['args'].get('max_length', 128)} —— 这是有意的口径对齐，不是笔误")
    _, val_loader, test_loader = build_dataloaders(
        tokenizer=tokenizer,
        label2id=label2id,
        batch_size=args.batch_size,
        max_length=max_length,
        data_dir=DATA_DIR,
    )
    loader = val_loader if args.split == "validation" else test_loader

    print(f"\n正在 [{args.split}] 集上推理...")
    all_preds, all_golds = run_inference(model, loader, id2label, device, args.use_crf)

    p = precision_score(all_golds, all_preds)
    r = recall_score(all_golds, all_preds)
    f1 = f1_score(all_golds, all_preds)
    report_dict = classification_report(
        all_golds, all_preds, digits=4, zero_division=0, output_dict=True
    )

    print("\n" + "=" * 70)
    print(f"模型：{'BERT + CRF' if args.use_crf else 'BERT + Linear'}  |  评估集：{args.split}")
    print("=" * 70)
    print(f"Entity-level Precision: {p:.4f}")
    print(f"Entity-level Recall:    {r:.4f}")
    print(f"Entity-level F1:        {f1:.4f}")

    print("\n【逐类型指标】")
    print(classification_report(all_golds, all_preds, digits=4, zero_division=0))

    illegal_stats = count_illegal_sequences(all_preds)
    print("【非法 BIO 序列统计】")
    print(f"  总序列数：{illegal_stats['total_seqs']}")
    print(f"  非法开头（I-X 开头）：{illegal_stats['illegal_start']} 条")
    print(f"  非法转移（B-X/I-X → I-Y, X≠Y）：{illegal_stats['illegal_transition']} 条")
    print(f"  合计非法序列：{illegal_stats['total_illegal']} 条")
    pct = illegal_stats["total_illegal"] / max(illegal_stats["total_seqs"], 1) * 100
    if args.use_crf:
        print(f"  → CRF 非法序列 {illegal_stats['total_illegal']} 条（{pct:.2f}%）")
        print("  → 注意：转移矩阵是学出来的，不是硬约束；要彻底杜绝需给非法转移加惩罚")
    else:
        print(f"  → 线性头 {pct:.2f}% 的序列含非法转移")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    result = {
        "model": "BERT+CRF" if args.use_crf else "BERT+Linear",
        "dataset": "peoples_daily",
        "split": args.split,
        "n_samples": len(all_golds),
        "ckpt_epoch": ckpt["epoch"],
        "max_length": max_length,
        "precision": round(p, 6),
        "recall": round(r, 6),
        "f1": round(f1, 6),
        "illegal_stats": illegal_stats,
        # 只保留实体类型的指标，去掉 micro/macro/weighted 汇总行
        # 注意：sklearn 返回的 support 是 numpy.int64，json 不认，必须显式转成 int/float，
        # 否则 json.dump 会在写一半时抛 TypeError，留下一个残缺的 JSON 文件
        "per_type": {
            k: {
                "precision": float(v["precision"]),
                "recall": float(v["recall"]),
                "f1-score": float(v["f1-score"]),
                "support": int(v["support"]),
            }
            for k, v in report_dict.items()
            if k in entity_types(labels)
        },
    }
    suffix = f"_{args.tag}" if args.tag else ""
    out_path = LOG_DIR / f"work_eval_{run_tag}_{args.split}{suffix}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n评估结果已保存 → {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="评估 BERT NER 模型（人民日报数据集）")
    parser.add_argument("--use_crf", action="store_true")
    parser.add_argument("--bert_path", type=Path, default=BERT_PATH)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--split", choices=["validation", "test"], default="validation",
                        help="人民日报数据 test 集带标签，两个都能用")
    parser.add_argument("--max_length", type=int, default=None,
                        help="覆盖 checkpoint 里训练时的 max_length。默认沿用训练值（128），"
                             "但 128 只能读前 126 个字，长句里的实体会整个漏掉；"
                             "要跟看全文的生成式模型对比，得传 512")
    parser.add_argument("--tag", default="",
                        help="输出文件名后缀，如 --tag full → work_eval_crf_validation_full.json。"
                             "加了这个才不会覆盖掉原来的基线结果")
    return parser.parse_args()


if __name__ == "__main__":
    main()
