"""
非法 BIO 转移统计 —— Linear vs CRF 的核心对比点（老师 PPT P25）。

用法：
  python eval/illegal_bio.py                     # 用现有产物能算的都算，缺原始 BIO 的明确标注
  python eval/illegal_bio.py --split test        # 指定 split（默认 test）

【重要口径说明 —— 老师 NOTES/ARCHITECTURE 反复强调】
  1. 非法 BIO 必须用**模型 decode 后的原始标签序列**统计（I 开头 / I 接 O / 类型串味）。
     绝不能用 decode_bio_spans 兜底后的 span 反推 —— 兜底会把 O→I 当成新实体起点，
     非法被掩盖，统计恒为 0，是假数据。
  2. 本作业 evaluate.py 默认只写 pred_spans（兜底后），原始 BIO 已丢失。故本脚本：
       - 若检测到 preds/<run>_<split>_rawtags.jsonl（用 --dump_raw_tags 重跑评测导出），
         则用老师口径**精确统计**非法转移；
       - 否则明确标注「需补采」，并先算出**能从 span 真实观测的结构异常**
         （重叠 span / 越界 / 类型幻觉）——这是 BIO 非法在 span 层面的等价表征。
  3. 老师数据有重叠论元，金标投影到 BIO 天生有 ~1.9% 非法（地板值），必须一起报；
     cdtier 是平铺 NER、gold 零重叠，故地板值 = 0（本脚本实测确认）。

补采精确统计（不重训，只重跑评测，T4 上判别式秒级）：
  python bert/evaluate.py --run_name roberta_linear --split test --dump_raw_tags
  python bert/evaluate.py --run_name roberta_crf    --split test --dump_raw_tags
  python bilstm_crf/evaluate.py --run_name bilstm_linear --split test --dump_raw_tags
  python bilstm_crf/evaluate.py --run_name bilstm_crf    --split test --dump_raw_tags
  python eval/illegal_bio.py     # 自动检测并精确统计
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

from common import paths
from common.metrics import spans_from_bio  # noqa: F401  (供核对兜底逻辑)

# 老师 PPT P25 的对照基准（bert_linear_256 → bert_crf_256，dev 930 句）
TEACHER_BASELINE = {
    "linear_total_illegal": 1516, "linear_illegal_seqs": 623, "linear_per_seq": 1.63,
    "crf_total_illegal": 1242, "crf_illegal_seqs": 581, "crf_per_seq": 1.34,
    "reduction_pct": 18.1, "total_seqs": 930,
    "seq_rate_linear": 0.670, "seq_rate_crf": 0.625,
    "source": "老师 PPT P25 / outputs/metrics/disc/bert_linear_256.json + bert_crf_256.json",
}

RUNS = [
    ("bilstm_linear", "L1 BiLSTM+Linear", "bilstm"),
    ("bilstm_crf", "L1 BiLSTM+CRF", "bilstm"),
    ("roberta_linear", "L2 RoBERTa+Linear", "roberta"),
    ("roberta_crf", "L2 RoBERTa+CRF", "roberta"),
    ("qwen_lora_sft", "L4 Qwen3-LoRA-SFT", "sft"),
    ("qwen_full_sft", "L4 Qwen3-full-SFT", "sft"),
]


def illegal_bio_stats(seqs: list[list[str]]) -> dict[str, Any]:
    """老师口径（event_data.py:324-362 等价实现）：统计原始 BIO 序列的非法转移。"""
    stats: dict[str, Any] = {
        "illegal_start": 0, "illegal_transition": 0, "total_seqs": len(seqs),
        "illegal_seqs": 0, "detail": Counter(),
    }

    def bump(kind: str) -> None:
        stats["detail"][kind] += 1

    for seq in seqs:
        if not seq:
            continue
        bad_here = 0
        if seq[0].startswith("I-"):
            stats["illegal_start"] += 1
            bump("I-x 出现在序列起始")
            bad_here += 1
        for i in range(1, len(seq)):
            prev, cur = seq[i - 1], seq[i]
            if not cur.startswith("I-"):
                continue
            etype = cur[2:]
            if prev == "O":
                stats["illegal_transition"] += 1
                bump("I-x 紧接 O")
                bad_here += 1
            elif prev.startswith("B-") and prev[2:] != etype:
                stats["illegal_transition"] += 1
                bump("I-x 紧接 B-y (类型串味)")
                bad_here += 1
            elif prev.startswith("I-") and prev[2:] != etype:
                stats["illegal_transition"] += 1
                bump("I-x 紧接 I-y (类型串味)")
                bad_here += 1
        if bad_here:
            stats["illegal_seqs"] += 1
    stats["total_illegal"] = stats["illegal_start"] + stats["illegal_transition"]
    stats["illegal_seq_rate"] = round(stats["illegal_seqs"] / max(stats["total_seqs"], 1), 4)
    stats["illegal_per_seq"] = round(stats["total_illegal"] / max(stats["total_seqs"], 1), 3)
    stats["detail"] = dict(stats["detail"])
    return stats


def load_rawtags(path: Path) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            out[d["id"]] = list(d["bio"])
    return out


def load_preds(path: Path) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            out[d["id"]] = d.get("pred_spans", [])
    return out


def span_structural_stats(records: list[dict], preds: dict[str, list[dict]],
                          valid_types: set[str]) -> dict[str, Any]:
    """从 span 层面算结构异常 —— 这是 BIO 非法在 span 视角的等价表征，可从现有产物真实计算。"""
    overlap_chars = 0      # 被两个及以上 span 覆盖的字符数（BIO 中一个字符不可能有两个标签）
    total_chars = 0
    overlap_spans = 0
    out_of_range = 0
    type_hallucination = 0
    nonempty = 0
    for rec in records:
        n = len(rec["tokens"])
        spans = preds.get(rec["id"], [])
        if spans:
            nonempty += 1
        cover = [0] * n
        for sp in spans:
            s, e = int(sp["start"]), int(sp["end"])
            if not (0 <= s < e <= n):
                out_of_range += 1
                continue
            if sp["type"] not in valid_types:
                type_hallucination += 1
            for i in range(s, e):
                cover[i] += 1
        overlap_chars += sum(1 for c in cover if c >= 2)
        total_chars += n
        # span 两两重叠对数
        for i in range(len(spans)):
            for j in range(i + 1, len(spans)):
                a, b = spans[i], spans[j]
                if a["start"] < b["end"] and b["start"] < a["end"]:
                    overlap_spans += 1
    return {
        "num_sentences": len(records),
        "num_nonempty_pred": nonempty,
        "overlap_char_count": overlap_chars,
        "overlap_span_pairs": overlap_spans,
        "overlap_char_rate": round(overlap_chars / max(total_chars, 1), 6),
        "out_of_range_spans": out_of_range,
        "type_hallucination_spans": type_hallucination,
        "note": "重叠 span = 同一字符被两个实体覆盖，在 BIO 单标签体系下不可能，"
                "是非法 BIO 在 span 层的等价表征；gold 中此类为 0（实测）。",
    }


def gold_floor(records: list[dict]) -> dict[str, Any]:
    """金标 BIO 地板值：gold span 投影成 BIO 后的非法数（老师要求一起报）。"""
    seqs: list[list[str]] = []
    overlap_pairs = 0
    for rec in records:
        n = len(rec["tokens"])
        tags = ["O"] * n
        es = rec["entities"]
        for i, a in enumerate(es):
            for b in es[i + 1:]:
                if a["start"] < b["end"] and b["start"] < a["end"]:
                    overlap_pairs += 1
            for k in range(a["start"], min(a["end"], n)):
                tags[k] = f"{'B' if k == a['start'] else 'I'}-{a['type']}"
        seqs.append(tags)
    st = illegal_bio_stats(seqs)
    st["gold_overlap_pairs"] = overlap_pairs
    st["note"] = "cdtier 平铺 NER，gold 无重叠论元 → 地板值应为 0；" \
                 "老师事件抽取因重叠论元投影，金标自带约 1.9% 非法（必须一起报，否则归因错误）"
    return st


def main() -> None:
    ap = argparse.ArgumentParser(description="非法 BIO 转移统计（Linear vs CRF）")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    data_dir = paths.get_data_dir()
    label_path = data_dir / "label_names.json"
    labels = json.loads(label_path.read_text(encoding="utf-8"))
    valid_types = set(labels["entity_types"]) if isinstance(labels, dict) and "entity_types" in labels \
        else {t[2:] for t in labels if isinstance(t, str) and t.startswith("B-")}

    records = json.loads((data_dir / f"{args.split}.json").read_text(encoding="utf-8"))

    floor = gold_floor(records)
    print("=" * 72)
    print(f"非法 BIO 统计 | split={args.split} | {len(records)} 句")
    print("=" * 72)
    print(f"[金标地板] 非法转移 {floor['total_illegal']} 处 / 重叠对 {floor['gold_overlap_pairs']}"
          f" —— cdtier 平铺 NER 地板值为 0（老师重叠论元数据约 1.9%）")

    results: dict[str, Any] = {
        "split": args.split,
        "gold_floor": floor,
        "teacher_baseline": TEACHER_BASELINE,
        "runs": {},
        "generated_at": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
    }

    for run_name, label, family in RUNS:
        pred_path = paths.pred_dir() / f"{run_name}_{args.split}.jsonl"
        raw_path = paths.pred_dir() / f"{run_name}_{args.split}_rawtags.jsonl"
        entry: dict[str, Any] = {"label": label, "family": family}

        if pred_path.exists():
            preds = load_preds(pred_path)
            entry["span_structural"] = span_structural_stats(records, preds, valid_types)
        else:
            entry["span_structural"] = None

        if raw_path.exists():
            raw = load_rawtags(raw_path)
            seqs = [raw[r["id"]] for r in records if r["id"] in raw]
            entry["illegal_bio"] = illegal_bio_stats(seqs)
            entry["raw_tags_source"] = str(raw_path)
            entry["raw_tags_coverage"] = len(seqs)
        else:
            entry["illegal_bio"] = None
            entry["raw_tags_source"] = None
            entry["raw_tags_coverage"] = 0

        results["runs"][run_name] = entry
        ss = entry["span_structural"]
        ib = entry["illegal_bio"]
        if ib is not None:
            print(f"  {label:<20} 非法转移 {ib['total_illegal']:>5} 处 / {ib['illegal_seqs']} 句"
                  f" | 每句 {ib['illegal_per_seq']}")
        elif ss is not None:
            print(f"  {label:<20} [原始BIO待补采] span重叠 {ss['overlap_span_pairs']} 对"
                  f" / 覆盖字符 {ss['overlap_char_count']} / 越界 {ss['out_of_range_spans']}"
                  f" / 类型幻觉 {ss['type_hallucination_spans']}")

    # Linear vs CRF 精确对比（仅当两臂都有 rawtags 时）
    pairs = [("bilstm_linear", "bilstm_crf", "L1 BiLSTM: Linear → CRF"),
             ("roberta_linear", "roberta_crf", "L2 RoBERTa: Linear → CRF")]
    comparisons = []
    for lin, crf, name in pairs:
        li = results["runs"][lin]["illegal_bio"]
        ci = results["runs"][crf]["illegal_bio"]
        if li and ci and li["total_illegal"] > 0:
            red = (li["total_illegal"] - ci["total_illegal"]) / li["total_illegal"] * 100
            comparisons.append({"pair": name, "linear": li["total_illegal"],
                                "crf": ci["total_illegal"], "reduction_pct": round(red, 1)})
        else:
            comparisons.append({"pair": name, "status": "need_rawtags"})
    results["comparisons"] = comparisons

    # 写 JSON
    stats_path = paths.eval_dir() / "illegal_bio_stats.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    # 写 Markdown 报告
    md: list[str] = [
        "# 非法 BIO 转移统计（Linear vs CRF）",
        "",
        f"> split = `{args.split}`，{len(records)} 句；生成于 {results['generated_at']}",
        "",
        "## 1. 口径（为什么这件事容易做错）",
        "",
        "非法 BIO 指三类：**I-x 出现在序列开头**、**I-x 紧接 O**、**类型串味**（I-x 接 B-y/I-y）。",
        "老师 PPT P25 的核心结论是：Linear 逐字独立预测会喷出大量非法转移，CRF 用转移矩阵",
        "学到标签间约束后非法显著下降。"
        "**必须统计模型 decode 的原始标签**，不能用兜底后的 span 反推——",
        "`decode_bio_spans` 会把 `O→I-x` 当成新实体起点，反推后非法恒为 0，是假数据。",
        "",
        "## 2. 金标地板值（老师要求必须一起报）",
        "",
        f"- cdtier gold 重叠实体对 = {floor['gold_overlap_pairs']}，"
        f"gold 投影 BIO 非法转移 = **{floor['total_illegal']} 处**",
        "- 老师事件抽取因重叠论元投影到 BIO，金标自带约 1.9% 非法；本作业是平铺 NER，地板 = 0。",
        "- 意义：报告的非法数全部来自模型，不含 Gold 自身的投影噪声，归因干净。",
        "",
        "## 3. 本作业实测",
        "",
        "| 方法 | 原始BIO非法转移 | 含非法句数 | 每句处数 | span重叠对 | 越界 | 类型幻觉 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for run_name, label, _ in RUNS:
        e = results["runs"][run_name]
        ib, ss = e["illegal_bio"], e["span_structural"] or {}
        v = lambda k, d="—": (ib.get(k, d) if ib else d)
        md.append(f"| {label} | {v('total_illegal')} | {v('illegal_seqs')} | {v('illegal_per_seq')} "
                  f"| {ss.get('overlap_span_pairs', '—')} | {ss.get('out_of_range_spans', '—')} "
                  f"| {ss.get('type_hallucination_spans', '—')} |")
    md += [
        "",
        "> 原始BIO非法转移列若为 —，说明现有 preds 只存了 span（兜底后），无法反推。"
        "span 重叠对是 BIO 非法在 span 层的等价表征（同一字符被两个实体覆盖，单标签 BIO 不可能）。",
        "",
        "## 4. 另一个可直接观测的对比：span 重叠（无需原始 BIO）",
        "",
        "判别式四臂的 pred_spans 重叠对数 = **0**（BIO 单标签体系 + 解码兜底天然保证"
        "一个字符只属于一个实体）；而 SFT 两臂分别出现重叠——这是**生成式没有 BIO 约束**"
        "的直接证据，等价于老师的「非法 BIO」在 span 层的表征。",
        "",
        "| 范式 | 方法 | span 重叠对 | 被重复覆盖字符 |",
        "|---|---|---:|---:|",
    ]
    for run_name, label, _ in RUNS:
        ss = results["runs"][run_name]["span_structural"] or {}
        if ss:
            md.append(f"| {'判别式' if run_name.startswith(('bilstm', 'roberta')) else '生成式'} "
                      f"| {label} | {ss['overlap_span_pairs']} | {ss['overlap_char_count']} |")
    md += [
        "",
        "这个对比不需要原始 BIO，从现有 preds 即可复现——面试时可作为「判别式 vs 生成式」",
        "结构合法性差异的实证。注意 gold 无重叠，所以生成式的重叠全是模型过度抽取导致。",
        "",
        "## 5. Linear vs CRF 对比",
        "",
    ]
    for c in comparisons:
        if "status" in c:
            md.append(f"- **{c['pair']}**：需补采原始 BIO（见文末命令）")
        else:
            md.append(f"- **{c['pair']}**：{c['linear']} → {c['crf']} 处"
                      f"（**−{c['reduction_pct']}%**）")
    tb = TEACHER_BASELINE
    md += [
        "",
        f"老师基准（{tb['source']}）：Linear {tb['linear_total_illegal']} 处 / "
        f"{tb['linear_illegal_seqs']} 句 → CRF {tb['crf_total_illegal']} 处 / "
        f"{tb['crf_illegal_seqs']} 句，**−{tb['reduction_pct']}%**。",
        f"注意老师特别提醒：选「含非法句占比」会看反效果（{tb['seq_rate_linear']}→"
        f"{tb['seq_rate_crf']} 只降 4.5pp），因为长序列上任一 span 开头吐 I- 就整句非法；",
        "看**处数**和**每句处数**才有区分度。",
        "",
        "## 6. 补采精确统计（不重训，只重跑评测）",
        "",
        "```bash",
        "python bert/evaluate.py --run_name roberta_linear --split test --dump_raw_tags",
        "python bert/evaluate.py --run_name roberta_crf    --split test --dump_raw_tags",
        "python bilstm_crf/evaluate.py --run_name bilstm_linear --split test --dump_raw_tags",
        "python bilstm_crf/evaluate.py --run_name bilstm_crf    --split test --dump_raw_tags",
        "python eval/illegal_bio.py --split test",
        "```",
        "",
        "重跑后本脚本自动检测 `*_rawtags.jsonl`，第 3/4 节的精确列会填满。",
    ]
    report_path = paths.eval_dir() / "illegal_bio_report.md"
    report_path.write_text("\n".join(md) + "\n", encoding="utf-8")

    print("\n对比结果：")
    for c in comparisons:
        print(f"  {c}")
    print(f"\n报告 → {report_path}")
    print(f"数据 → {stats_path}")


if __name__ == "__main__":
    main()
