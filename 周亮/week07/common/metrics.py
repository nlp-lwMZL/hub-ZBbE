"""
统一评测口径 —— 三类方法（BiLSTM / RoBERTa / Qwen-SFT）共用同一套实体级 P/R/F1。

为什么必须只有一个评测函数（老师 PPT 反复强调）：
  「谁更好」这件事要可信，前提是三份预测喂给同一个函数。所以每类方法都把预测
  写成同一种 JSONL（pred_spans: [{type,start,end,text}]，字符级 span），本模块统一读取、
  统一转 BIO、统一用 seqeval 算实体级 P/R/F1；生成式（SFT）走 text.find 对齐后也落到
  同一种 span，不信任它可能给出的坐标。

【预测 JSONL 契约】每行一条，与 clean/{split}.json 按 id 对齐：
  {"id": "...", "pred_spans": [{"type":"Attacker","text":"APT-C-39","start":12,"end":20}, ...]}
  - start/end 是字符级左闭右开；gold 里的 entities 也是同口径
  - 评测**看 (type, start, end) 三元组完全相等**才算 TP（与 seqeval 的 BIO EM 等价）
  - 同一 id 若重复出现，报错而不是静默覆盖（老师踩过的坑：id 不唯一导致分母变小）

【指标】
  - entity P/R/F1（seqeval 口径，IOB2）
  - per-type P/R/F1（classification_report），定位长尾类型（Campaign 只有 91 条）
  - 非法 BIO 转移数（Linear vs CRF 的对照点；压缩序列口径，仅作相对比较）
  - 支持从 BIO id 序列 / span 列表两种来源构造预测，统一到 span 比较

自检：python common/metrics.py
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import sys
from pathlib import Path
_CODE_DIR = Path(__file__).resolve().parent.parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))
from common.ner_data import LabelScheme, decode_bio_ids

__all__ = [
    "spans_from_bio", "bio_from_spans", "span_key", "evaluate", "evaluate_flat",
    "load_gold", "load_predictions", "write_predictions", "illegal_bio_stats",
]


# --------------------------------------------------------------------------- #
# span <-> BIO 转换
# --------------------------------------------------------------------------- #
def span_key(span: dict[str, Any]) -> tuple[str, int, int]:
    """评测键：(类型, 字符 start, 字符 end)。文本相同但位置不同算两个实体。"""
    return (span["type"], int(span["start"]), int(span["end"]))


def spans_from_bio(tags: Sequence[str]) -> list[dict[str, Any]]:
    """BIO 字符串序列 -> span 列表（字符下标 = 序列下标，用于 token==字符的数据）。"""
    spans: list[dict[str, Any]] = []
    cur_type: str | None = None
    cur_start = 0

    def close(end: int) -> None:
        nonlocal cur_type
        if cur_type is not None:
            spans.append({"type": cur_type, "start": cur_start, "end": end,
                          "text": "", "_token_start": cur_start, "_token_end": end - 1})
            cur_type = None

    for i, tag in enumerate(tags):
        if tag == "O" or tag.startswith("O"):
            close(i); continue
        prefix, _, etype = tag.partition("-")
        if prefix == "B":
            close(i); cur_type, cur_start = etype, i
        elif prefix == "I":
            if cur_type != etype:      # 非法 I：按新实体起点处理（与 decode_bio_ids 一致）
                close(i); cur_type, cur_start = etype, i
        else:
            close(i)
    close(len(tags))
    return spans


def bio_from_spans(n: int, spans: Sequence[dict[str, Any]]) -> list[str]:
    """span 列表 -> 长度 n 的 BIO 序列（字符级）。重叠 span 后者覆盖前者并计数。"""
    tags = ["O"] * n
    for sp in spans:
        s, e, t = int(sp["start"]), int(sp["end"]), sp["type"]
        if not (0 <= s < e <= n):
            continue
        tags[s] = f"B-{t}"
        for i in range(s + 1, e):
            tags[i] = f"I-{t}"
    return tags


def illegal_bio_stats(tags: Sequence[str]) -> tuple[int, Counter]:
    """统计压缩 BIO 序列的非法转移（I 起始 / I 接 O / 类型串味）。"""
    n = 0
    detail: Counter = Counter()
    prev: str | None = None
    for tag in tags:
        if tag.startswith("I-"):
            etype = tag[2:]
            if prev is None:
                n += 1; detail["I-X 出现在序列起始"] += 1
            elif prev == "O":
                n += 1; detail["I-X 紧接 O"] += 1
            elif prev.startswith("B-") and prev[2:] != etype:
                n += 1; detail["I-X 紧接 B-Y"] += 1
            elif prev.startswith("I-") and prev[2:] != etype:
                n += 1; detail["I-X 紧接 I-Y"] += 1
        prev = tag
    return n, detail


# --------------------------------------------------------------------------- #
# 评测主函数
# --------------------------------------------------------------------------- #
def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def evaluate_flat(
    gold: Sequence[Sequence[dict[str, Any]]],
    pred: Sequence[Sequence[dict[str, Any]]],
    entity_types: Sequence[str] | None = None,
) -> dict[str, Any]:
    """
    实体级 P/R/F1：三元组 (type, start, end) 完全相等才算 TP。
    gold/pred 都是「每条样本的 span 列表」组成的列表，按样本顺序一一对应。
    """
    if len(gold) != len(pred):
        raise ValueError(f"gold/pred 样本数不一致：{len(gold)} vs {len(pred)}")

    tp_all = fp_all = fn_all = 0
    per_type: dict[str, dict[str, int]] = {}
    for gs, ps in zip(gold, pred):
        gset = {span_key(s) for s in gs}
        pset = {span_key(s) for s in ps}
        tp, fp, fn = len(gset & pset), len(pset - gset), len(gset - pset)
        tp_all += tp; fp_all += fp; fn_all += fn
        for t, s, e in gset:
            per_type.setdefault(t, {"tp": 0, "fp": 0, "fn": 0})["tp"] += int((t, s, e) in pset)
            per_type[t]["fn"] += int((t, s, e) not in pset)
        for t, s, e in pset - gset:
            per_type.setdefault(t, {"tp": 0, "fp": 0, "fn": 0})["fp"] += 1

    p, r, f = _prf(tp_all, fp_all, fn_all)
    per_type_prf: dict[str, dict[str, float]] = {}
    for t in (entity_types or sorted(per_type)):
        c = per_type.get(t, {"tp": 0, "fp": 0, "fn": 0})
        tp, fp, fn = c["tp"], c["fp"], c["fn"]
        pt, rt, ft = _prf(tp, fp, fn)
        per_type_prf[t] = {"precision": round(pt, 4), "recall": round(rt, 4),
                           "f1": round(ft, 4), "support": tp + fn}
    macro_f1 = sum(v["f1"] for v in per_type_prf.values()) / max(1, len(per_type_prf))
    return {
        "num_samples": len(gold),
        "gold_entities": tp_all + fn_all,
        "pred_entities": tp_all + fp_all,
        "precision": round(p, 4),
        "recall": round(r, 4),
        "f1": round(f, 4),
        "type_macro_f1": round(macro_f1, 4),
        "per_type": per_type_prf,
    }


def evaluate(
    gold_records: Sequence[dict],
    pred_by_id: dict[str, Sequence[dict[str, Any]]],
    scheme: LabelScheme | None = None,
) -> dict[str, Any]:
    """从 clean 记录 + {id: pred_spans} 评测（按 id 对齐，重复 id 直接报错）。"""
    ids = [r["id"] for r in gold_records]
    if len(ids) != len(set(ids)):
        dup = [k for k, v in Counter(ids).items() if v > 1]
        raise ValueError(f"gold 里 id 不唯一：{dup[:5]}。clean 数据按文本切分本应唯一，请检查数据源。")
    missing = [i for i in ids if i not in pred_by_id]
    if missing:
        raise ValueError(f"有 {len(missing)} 条 gold 没有对应预测（如 {missing[:3]}），"
                         "解析失败/空输出也必须写 events:[]，不能删行（删了分母变小、F1 虚高）。")
    gold = []
    for r in gold_records:
        gold.append([
            {"type": e["type"], "start": int(e["start"]), "end": int(e["end"]), "text": e.get("text", "")}
            for e in r.get("entities", [])
        ])
    pred = [pred_by_id[i] for i in ids]
    ent_types = scheme.entity_types if scheme is not None else None
    return evaluate_flat(gold, pred, ent_types)


# --------------------------------------------------------------------------- #
# 读写
# --------------------------------------------------------------------------- #
def write_predictions(path: str | Path, pred_by_id: dict[str, Sequence[dict[str, Any]]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for rid, spans in pred_by_id.items():
            row = {"id": rid, "pred_spans": [
                {"type": s["type"], "start": int(s["start"]), "end": int(s["end"]),
                 "text": s.get("text", "")} for s in spans
            ]}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_predictions(path: str | Path) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    with Path(path).open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            rid = row["id"]
            if rid in out:
                raise ValueError(f"预测文件第 {line_no} 行 id 重复：{rid}")
            out[rid] = [{"type": s["type"], "start": int(s["start"]), "end": int(s["end"]),
                         "text": s.get("text", "")} for s in row.get("pred_spans", [])]
    return out


def load_gold(path: str | Path) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# 自检
# --------------------------------------------------------------------------- #
def _run_self_test() -> None:
    print("=" * 70)
    print("common/metrics 自检")
    print("=" * 70)
    scheme = LabelScheme.from_entity_types(["Attacker", "Tools"])
    # 完美预测
    g = [[{"type": "Attacker", "start": 0, "end": 3}],
         [{"type": "Tools", "start": 2, "end": 5}]]
    m = evaluate_flat(g, g, scheme.entity_types)
    print(f"  完美预测 F1={m['f1']}（应=1.0）")
    assert m["f1"] == 1.0
    # 一个错类型：FP+FN 双输
    p = [[{"type": "Tools", "start": 0, "end": 3}],
         [{"type": "Tools", "start": 2, "end": 5}]]
    m = evaluate_flat(g, p, scheme.entity_types)
    print(f"  类型错 F1={m['f1']}（应<1，且 gold/pred 实体数各 2）"
          f" | gold={m['gold_entities']} pred={m['pred_entities']}")
    assert m["f1"] < 1.0 and m["gold_entities"] == 2 and m["pred_entities"] == 2
    # span <-> BIO 往返
    tags = ["B-Attacker", "I-Attacker", "O", "B-Tools", "I-Tools", "I-Tools"]
    spans = spans_from_bio(tags)
    back = bio_from_spans(len(tags), spans)
    print(f"  BIO→span→BIO 往返: {tags} == {back}")
    assert tags == back
    # 非法统计
    n, detail = illegal_bio_stats(["O", "I-Tools", "B-Attacker", "I-Tools"])
    print(f"  非法 BIO 转移 {n} 处: {dict(detail)}（应=2）")
    assert n == 2
    print("自检通过 ✓")


if __name__ == "__main__":
    _run_self_test()
