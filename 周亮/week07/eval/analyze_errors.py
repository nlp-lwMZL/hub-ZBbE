#!/usr/bin/env python3
"""
错误分析 —— 从 6 份逐条预测里挖出"可写进简历的具体发现"。

老师 P42 简历亮点③就是这么来的：逐条看预测才发现"数量类论元只差一个『股』"。
本脚本做三件事：
  1. 类型混淆矩阵：gold 类型 × pred 类型（只看有字符重叠的 span 对）
  2. 错误分解：TP / 类型错(边界相同) / 边界错(类型相同) / 都错 / 漏召 / 多预测
     —— 回答"模型是分不清类型，还是定不准边界"
  3. 多模型并排 case 表：挑代表性样本（全对/只有SFT对/全漏/长尾类/长句），
     把 6 臂在同一句上的预测并排展示

口径：span = (type, start, end)，end 右开；匹配看字符重叠 IoU。
不重跑模型，纯离线从 outputs/preds/*.jsonl 读。

产物：outputs/eval/errors_report.md / errors_stats.json / case_table.md

用法：python eval/analyze_errors.py
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))
from common import paths
from common.ner_data import load_records

METHODS = ["bilstm_linear", "bilstm_crf", "roberta_linear", "roberta_crf",
           "qwen_lora_sft", "qwen_full_sft"]


def iou(a: dict, b: dict) -> float:
    s1, e1 = a["start"], a["end"]
    s2, e2 = b["start"], b["end"]
    inter = max(0, min(e1, e2) - max(s1, s2))
    union = max(e1, e2) - min(s1, s2)
    return inter / union if union > 0 else 0.0


def classify_errors(gold: list[dict], pred: list[dict]) -> dict:
    """对一条样本做 span 级错误分解（贪心一对一匹配，按 IoU 降序）。"""
    pairs = [(iou(g, p), gi, pi) for gi, g in enumerate(gold) for pi, p in enumerate(pred)]
    pairs.sort(reverse=True)
    matched_g, matched_p = set(), set()
    stats = {"tp": 0, "type_error": 0, "boundary_error": 0,
             "both_error": 0, "missed": 0, "spurious": 0}
    conf = Counter()
    examples: dict[str, list] = defaultdict(list)
    for ov, gi, pi in pairs:
        if ov <= 0 or gi in matched_g or pi in matched_p:
            continue
        g, p = gold[gi], pred[pi]
        matched_g.add(gi); matched_p.add(pi)
        same_type = g["type"] == p["type"]
        same_span = g["start"] == p["start"] and g["end"] == p["end"]
        if same_type and same_span:
            stats["tp"] += 1
        elif same_type:
            stats["boundary_error"] += 1
            examples["boundary"].append({"gold": g, "pred": p})
        elif same_span:
            stats["type_error"] += 1
            conf[(g["type"], p["type"])] += 1
            examples["type"].append({"gold": g, "pred": p})
        else:
            stats["both_error"] += 1
            conf[(g["type"], p["type"])] += 1
            examples["both"].append({"gold": g, "pred": p})
    stats["missed"] = len(gold) - len(matched_g)
    stats["spurious"] = len(pred) - len(matched_p)
    return {"stats": stats, "conf": conf, "examples": examples}


def fmt_span(s: dict) -> str:
    return f"{s['type']}[{s['start']}:{s['end']}]={s.get('text','')}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=None)
    a = ap.parse_args()
    out_dir = Path(a.out_dir) if a.out_dir else paths.eval_dir()

    data_dir = paths.get_data_dir()
    gold_rows = {r["id"]: r for r in load_records("test", data_dir)}

    preds = {}
    for m in METHODS:
        p = paths.pred_dir() / f"{m}_test.jsonl"
        if p.exists():
            preds[m] = {row["id"]: row["pred_spans"] for row in
                        (json.loads(line) for line in p.read_text(encoding="utf-8").splitlines())}

    report = {"methods": {}, "type_confusion": {}, "case_candidates": {}}
    for m, pm in preds.items():
        agg = Counter()
        conf = Counter()
        ex = defaultdict(list)
        for rid, grows in gold_rows.items():
            gold = grows["entities"]
            pred = pm.get(rid, [])
            r = classify_errors(gold, pred)
            agg.update(r["stats"])
            conf.update(r["conf"])
            for k, v in r["examples"].items():
                ex[k].extend(v)
        tp, te, be, bo, ms, sp = (agg[k] for k in
            ("tp", "type_error", "boundary_error", "both_error", "missed", "spurious"))
        total_gold = tp + te + be + bo + ms
        report["methods"][m] = {
            "counts": dict(agg),
            "gold_spans": total_gold,
            "rates": {
                "tp_rate": round(tp / max(1, total_gold), 4),
                "type_error_rate": round(te / max(1, total_gold), 4),
                "boundary_error_rate": round(be / max(1, total_gold), 4),
                "missed_rate": round(ms / max(1, total_gold), 4),
            },
            "top_type_confusions": [
                {"gold": g, "pred": p, "n": n} for (g, p), n in conf.most_common(10)
            ],
            "boundary_examples": [
                {"gold": fmt_span(e["gold"]), "pred": fmt_span(e["pred"])}
                for e in ex["boundary"][:10]
            ],
            "type_examples": [
                {"gold": fmt_span(e["gold"]), "pred": fmt_span(e["pred"])}
                for e in ex["type"][:10]
            ],
        }

    # case 候选：挑几类代表性句子 id
    ids = list(gold_rows.keys())
    def all_correct(rid):
        gold = gold_rows[rid]["entities"]
        if not gold:                      # 排除 gold 为空的无实体句（否则"全预测空"假装修对）
            return False
        return all(
            (lambda s: s["missed"] == 0 and s["spurious"] == 0 and s["tp"] == len(gold))(
                classify_errors(gold, pm.get(rid, []))["stats"])
            for pm in preds.values())
    candidates = {
        "all_correct": [rid for rid in ids if preds and all_correct(rid)][:3],
        "long_text": [rid for rid in ids if len(gold_rows[rid]["tokens"]) > 256][:3],
        "campaign": [rid for rid in ids if any(e["type"] == "Campaign" for e in gold_rows[rid]["entities"])][:3],
        "qwen_only": [rid for rid in ids
                      if rid in preds.get("qwen_full_sft", {})
                      and preds["qwen_full_sft"][rid]
                      and not any(preds.get(m, {}).get(rid) for m in ("bilstm_crf", "roberta_crf"))][:3],
    }
    report["case_candidates"] = {k: {"ids": v,
        "texts": {rid: gold_rows[rid]["text"][:120] for rid in v}} for k, v in candidates.items()}

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "errors_stats.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # 人读报告
    L = ["# NER 错误分析报告", "",
         "> 从 6 臂逐条预测离线统计；span=(type,start,end)，匹配看字符重叠 IoU。",
         "> 目的：回答『模型是分不清类型，还是定不准边界』——这是选改进方向的关键。", ""]
    L += ["## 1. 错误分解（gold span 的去向）", "",
          "| 方法 | TP | 类型错(边界对) | 边界错(类型对) | 都错 | 漏召 | 多预测 | TP率 |",
          "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for m, d in report["methods"].items():
        c = d["counts"]
        L.append(f"| {m} | {c['tp']} | {c['type_error']} | {c['boundary_error']} | "
                 f"{c['both_error']} | {c['missed']} | {c['spurious']} | {d['rates']['tp_rate']} |")
    L += ["", "读法：", "- 类型错高 → 标签体系/相似类型混淆（如 Attacker vs Tools）；",
          "- 边界错高 → span 边界定义或分词对齐问题（老师『只差一个股』就是这类）；",
          "- 漏召高 → 召回不足（长尾类/截断）；多预测高 → 精确率问题。", ""]

    L += ["## 2. 类型混淆 Top（gold → pred）", ""]
    for m, d in report["methods"].items():
        if not d["top_type_confusions"]:
            continue
        L += [f"### {m}", "", "| gold | pred | 次数 |", "|---|---|---:|"]
        for c in d["top_type_confusions"]:
            L.append(f"| {c['gold']} | {c['pred']} | {c['n']} |")
        L.append("")

    L += ["## 3. 边界错的具体样子（类型对但 span 不同）", ""]
    for m in ("roberta_crf", "qwen_full_sft"):
        if m not in report["methods"]:
            continue
        exs = report["methods"][m]["boundary_examples"]
        L += [f"### {m}", ""]
        for e in exs[:8]:
            L.append(f"- gold `{e['gold']}` vs pred `{e['pred']}`")
        L.append("")

    L += ["## 4. 多模型并排 case（引子）", "",
          "从下列候选 id 里挑 5~8 条，手工查 `outputs/preds/<method>_test.jsonl` 做并排表：", ""]
    for k, v in report["case_candidates"].items():
        L += [f"### {k}"]
        for rid, txt in v["texts"].items():
            L.append(f"- `{rid}`：{txt}…")
        L.append("")
    L.append("并排表格式：`句子 | gold | BiLSTM-CRF | RoBERTa-CRF | Qwen-LoRA | Qwen-full`")
    (out_dir / "errors_report.md").write_text("\n".join(L), encoding="utf-8")

    # case_table.md：对候选句子自动填 6 臂的预测
    C = ["# 多模型并排 case 表", "",
         "> 每个候选取一条，展示 gold 与 6 臂在同一句上的预测 span。", ""]
    for k, v in report["case_candidates"].items():
        for rid in v["ids"][:2]:
            row = gold_rows[rid]
            C += [f"## [{k}] {rid}", "", f"**文本**：{row['text'][:200]}", "",
                  f"**gold**：" + ", ".join(fmt_span(e) for e in row["entities"]), "",
                  "| 方法 | 预测 span |", "|---|---|"]
            for m in METHODS:
                ps = preds.get(m, {}).get(rid, [])
                C.append(f"| {m} | " + (", ".join(fmt_span(s) for s in ps) or "（空）") + " |")
            C.append("")
    (out_dir / "case_table.md").write_text("\n".join(C), encoding="utf-8")

    print("\n".join(L[:32]))
    print(f"\n已写入 {out_dir}/errors_report.md, errors_stats.json, case_table.md")


if __name__ == "__main__":
    main()
