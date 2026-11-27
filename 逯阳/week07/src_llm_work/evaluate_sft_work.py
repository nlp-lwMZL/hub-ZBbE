"""
SFT 模型评估 —— 加载 LoRA adapter，在验证集 / 测试集上算 NER 的 entity-level F1

参考 ../src_llm/evaluate_sft.py，针对人民日报数据重写。改动：
  1. 主指标换成「（surface, type）多重集匹配」，理由见 common.py 文件头
     （本数据集 7.57% 的实体是句内重复，参考实现的 text.find 口径会低估召回）
  2. 同时输出参考实现口径的「严格位置匹配」F1，方便和 cluener 的 0.6801 对照
  3. 默认在**整个 split** 上评估（不是抽 100 条）—— 这样和 BERT 的
     outputs_work/logs/work_eval_*.json 是同一批样本，数字可以直接比
  4. 新增逐类型 P/R/F1，和 BERT 的 per_type 对齐着看

使用方式：
  python src_llm_work/evaluate_sft_work.py                       # 验证集全部 2318 条
  python src_llm_work/evaluate_sft_work.py --split test          # 测试集全部 4636 条
  python src_llm_work/evaluate_sft_work.py --n_samples 100       # 快速抽查
  python src_llm_work/evaluate_sft_work.py --demo                # 5 条，看输出长什么样
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import re
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

from common import (
    MODEL_PATH, ADAPTER_DIR, LOG_DIR, OUT_DIR, ROOT,
    SYSTEM_PROMPT, ENTITY_TYPES, ENTITY_TYPE_ZH,
    load_split, record_text, bio_to_spans, bio_to_entities,
    parse_entities, compute_metrics_multiset, compute_metrics_strict,
    per_type_metrics, sample_records, _dw, _pad,
)

BERT_LOG_DIR = ROOT / "outputs_work" / "logs"


# ══════════════════════════════════════════════════════════════════════════════
# 模型加载
# ══════════════════════════════════════════════════════════════════════════════

def load_model(model_path: str, ckpt_dir: str, device: torch.device):
    """目录里有 adapter_config.json → 按 LoRA 加载并合并；否则当成完整模型。"""
    ckpt_path = Path(ckpt_dir)
    is_lora = (ckpt_path / "adapter_config.json").exists()

    if is_lora:
        if not PEFT_AVAILABLE:
            raise ImportError("加载 LoRA adapter 需要 peft：pip install peft")
        print(f"检测到 LoRA adapter，加载 base model: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(
            str(Path(model_path).resolve()), trust_remote_code=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            str(Path(model_path).resolve()),
            dtype=torch.float16 if device.type == "cuda" else torch.float32,
            device_map="auto" if device.type == "cuda" else None,
            trust_remote_code=True,
        )
        print(f"加载 adapter: {ckpt_dir}")
        model = PeftModel.from_pretrained(base_model, str(ckpt_path))
        model = model.merge_and_unload()      # 合并进 base，推理更快
    else:
        print(f"检测到完整模型，直接加载: {ckpt_dir}")
        tokenizer = AutoTokenizer.from_pretrained(str(ckpt_path), trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            str(ckpt_path),
            dtype=torch.float16 if device.type == "cuda" else torch.float32,
            device_map="auto" if device.type == "cuda" else None,
            trust_remote_code=True,
        )

    if device.type != "cuda":
        model = model.to(device)
    model.eval()
    print(f"模型加载完成（{'LoRA 已合并' if is_lora else '完整模型'}）\n")
    return model, tokenizer


# ══════════════════════════════════════════════════════════════════════════════
# 推理
# ══════════════════════════════════════════════════════════════════════════════

def generate_ner(text: str, model, tokenizer, device: torch.device,
                 max_new_tokens: int = 192) -> str:
    """生成 JSON。max_new_tokens 给 192：实测目标最长 167 token，留点余量。"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": text},
    ]
    encoding = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_tensors="pt", return_dict=True,
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    prompt_len = input_ids.shape[-1]

    with torch.no_grad():
        output_ids = model.generate(
            input_ids, attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True).strip()


# ══════════════════════════════════════════════════════════════════════════════
# 和 BERT 对比
# ══════════════════════════════════════════════════════════════════════════════

def load_bert_metrics(split: str) -> dict:
    """读 outputs_work/logs/work_eval_{linear,crf}_{split}.json，没有就跳过。"""
    out = {}
    for tag, label in [("linear", "BERT+Linear"), ("crf", "BERT+CRF")]:
        p = BERT_LOG_DIR / f"work_eval_{tag}_{split}.json"
        if p.exists():
            with open(p, encoding="utf-8") as f:
                d = json.load(f)
            out[tag] = {"label": label, "precision": d["precision"],
                        "recall": d["recall"], "f1": d["f1"],
                        "per_type": d.get("per_type", {}),
                        "illegal": d.get("illegal_stats", {}).get("total_illegal"),
                        "ckpt_epoch": d.get("ckpt_epoch")}
    return out


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    ckpt_dir = Path(args.ckpt_dir)
    if not ckpt_dir.exists():
        print(f"[错误] checkpoint 目录不存在：{ckpt_dir}")
        print("请先运行：python src_llm_work/train_sft_work.py")
        return

    # ── 数据 ──────────────────────────────────────────────────────────────────
    data = load_split(args.split)
    n = 5 if args.demo else args.n_samples
    samples = sample_records(data, n, args.seed)
    print(f"评估集: {args.split}，样本 {len(samples)}/{len(data)} 条"
          f"{'（全量）' if len(samples) == len(data) else f'（随机抽样 seed={args.seed}）'}\n")

    model, tokenizer = load_model(args.model_path, str(ckpt_dir), device)

    # ── 逐条推理 ──────────────────────────────────────────────────────────────
    golds, preds = [], []
    gold_spans_list, texts = [], []
    detail = []
    n_format_fail = 0
    n_empty_pred = 0
    t0 = time.time()

    for i, record in enumerate(samples, 1):
        text = record_text(record)
        g_ents = bio_to_entities(record["tokens"], record["ner_tags"])
        raw = generate_ner(text, model, tokenizer, device)
        p_ents, json_ok = parse_entities(raw, text)

        if not json_ok:
            n_format_fail += 1
        if not p_ents:
            n_empty_pred += 1

        golds.append(g_ents)
        preds.append(p_ents)
        gold_spans_list.append(bio_to_spans(record["tokens"], record["ner_tags"]))
        texts.append(text)
        detail.append({
            "text": text,
            "gold": g_ents,
            "pred": p_ents,
            "raw_output": raw,
            "json_ok": json_ok,
        })

        if args.demo:
            print(f"[{i}] {text}")
            print(f"    gold: {g_ents}")
            print(f"    pred: {p_ents}")
            print(f"    raw : {raw[:160]}")
            print()
        elif i % 100 == 0 or i == len(samples):
            print(f"  已处理 {i}/{len(samples)}  ({time.time() - t0:.0f}s)")

    elapsed = time.time() - t0

    # ── 指标 ──────────────────────────────────────────────────────────────────
    main_m = compute_metrics_multiset(golds, preds)
    strict_m = compute_metrics_strict(texts, gold_spans_list, preds)
    per_type = per_type_metrics(golds, preds)
    bert = load_bert_metrics(args.split)

    print(f"\n{'=' * 74}")
    print(f"SFT（LoRA）NER 结果 —— {args.split}，{len(samples)} 条")
    print(f"{'=' * 74}")
    print(f"  Precision   : {main_m['precision']:.4f}")
    print(f"  Recall      : {main_m['recall']:.4f}")
    print(f"  F1          : {main_m['f1']:.4f}     ← 主指标（多重集匹配）")
    print(f"  严格位置匹配 : {strict_m['f1']:.4f}     ← 参考实现口径，和 cluener 的 0.6801 对照用")
    print(f"  格式失败    : {n_format_fail} 条 ({n_format_fail / len(samples) * 100:.1f}%)")
    print(f"  空预测      : {n_empty_pred} 条 ({n_empty_pred / len(samples) * 100:.1f}%)")
    print(f"  实体数      : pred {main_m['pred_total']} / gold {main_m['gold_total']}")
    print(f"  耗时        : {elapsed:.0f}s，均值 {elapsed / len(samples) * 1000:.0f} ms/条")

    # ── 逐类型 ────────────────────────────────────────────────────────────────
    print(f"\n{'逐类型（SFT 主指标）':^56}")
    head = _pad("类型", 6) + _pad("gold", 7, right=True) + _pad("P", 9, right=True) \
        + _pad("R", 9, right=True) + _pad("F1", 9, right=True)
    for b in bert.values():
        head += _pad(f"{b['label']} F1", 17, right=True)
    print(head)
    print("-" * (_dw(head)))
    for etype in ENTITY_TYPES:
        m = per_type[etype]
        row = _pad(etype, 6) + _pad(str(m["gold_count"]), 7, right=True) \
            + _pad(f"{m['precision']:.3f}", 9, right=True) \
            + _pad(f"{m['recall']:.3f}", 9, right=True) \
            + _pad(f"{m['f1']:.3f}", 9, right=True)
        for b in bert.values():
            bt = b["per_type"].get(etype, {})
            row += _pad(f"{bt.get('f1-score', float('nan')):.3f}", 17, right=True)
        print(row)

    # ── 和 BERT 对比 ──────────────────────────────────────────────────────────
    if bert:
        print(f"\n{'与 BERT（同一批样本）对比':^60}")
        head = _pad("方法", 20) + _pad("Precision", 11, right=True) \
            + _pad("Recall", 9, right=True) + _pad("F1", 9, right=True) \
            + _pad("耗时/条", 11, right=True)
        print(head)
        print("-" * _dw(head))
        for b in bert.values():
            print(_pad(b["label"], 20) + _pad(f"{b['precision']:.4f}", 11, right=True)
                  + _pad(f"{b['recall']:.4f}", 9, right=True)
                  + _pad(f"{b['f1']:.4f}", 9, right=True) + _pad("~5 ms", 11, right=True))
        print(_pad("SFT (LoRA)", 20) + _pad(f"{main_m['precision']:.4f}", 11, right=True)
              + _pad(f"{main_m['recall']:.4f}", 9, right=True)
              + _pad(f"{main_m['f1']:.4f}", 9, right=True)
              + _pad(f"{elapsed / len(samples) * 1000:.0f} ms", 11, right=True))
        best_bert = max(bert.values(), key=lambda x: x["f1"])
        gap = main_m["f1"] - best_bert["f1"]
        print(f"\n  SFT 比最强 BERT（{best_bert['label']}）低 {abs(gap):.4f}" if gap < 0
              else f"\n  SFT 比最强 BERT（{best_bert['label']}）高 {gap:.4f}")
    else:
        print(f"\n  [提示] 没找到 BERT 的评估结果（{BERT_LOG_DIR}），跳过对比")

    # ── 保存 ──────────────────────────────────────────────────────────────────
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = LOG_DIR / f"eval_sft_{args.split}.json"
    payload = {
        "model": "Qwen2-0.5B-Instruct + LoRA",
        "dataset": "peoples_daily",
        "split": args.split,
        "n_samples": len(samples),
        "full_split": len(samples) == len(data),
        "metrics_multiset": main_m,
        "metrics_strict": strict_m,
        "per_type": per_type,
        "format_fail": n_format_fail,
        "empty_pred": n_empty_pred,
        "elapsed_s": round(elapsed, 1),
        "ms_per_item": round(elapsed / len(samples) * 1000, 1),
        "bert_reference": bert,
        "detail": detail,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存 → {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="SFT NER 评估（人民日报）")
    parser.add_argument("--model_path", default=str(MODEL_PATH))
    parser.add_argument("--ckpt_dir",   default=str(ADAPTER_DIR))
    parser.add_argument("--split",      default="validation",
                        choices=["validation", "test"])
    parser.add_argument("--n_samples",  default=-1, type=int,
                        help="评估条数；-1 = 整个 split（默认，和 BERT 同批样本）")
    parser.add_argument("--seed",       default=42, type=int)
    parser.add_argument("--demo",       action="store_true", help="只跑 5 条看输出")
    return parser.parse_args()


if __name__ == "__main__":
    main()
