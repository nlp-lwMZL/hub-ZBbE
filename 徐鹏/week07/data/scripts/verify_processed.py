# -*- coding: utf-8 -*-
"""验收脚本：逐行解析 processed/ 下的 JSONL，重建 span 并校验契约。

校验项（报告“对齐率”一节直接引用这里的输出）：
  1. 每行都能 json.loads；
  2. 字段齐全、类型正确；
  3. 字符级 span 自洽：text[start:end] == span.text，0 <= start < end <= len(text)；
  4. extract：恰好 1 个事件；detect：has_event / types / n_events 三者一致；
  5. dev_docs：events 非空、每个 trigger/论元 span 自洽；
  6. 统计 train/dev 文本重叠与重复（教学时要提示的泄漏风险）。

用法：
  D:\\Anaconda\\python.exe scripts\\verify_processed.py
"""
import json
from collections import Counter
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent
PROC = DATA / "processed"
STATS = DATA / "stats"


def load(name):
    rows = []
    with open(PROC / name, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            rows.append(json.loads(line))
    return rows


def check_span(row_text, span, where, errors):
    """span 必须是 text[start:end] == span.text 的字符级左闭右开区间。"""
    s, e, t = span.get("start"), span.get("end"), span.get("text")
    if not isinstance(s, int) or not isinstance(e, int):
        errors.append((where, "start/end 不是 int"))
        return
    if not (0 <= s < e <= len(row_text)):
        errors.append((where, "区间越界 start=%r end=%r len=%d" % (s, e, len(row_text))))
        return
    if row_text[s:e] != t:
        errors.append((where, "重建不一致 %r != %r" % (row_text[s:e], t)))


def main():
    errors = []
    out = {}

    # ---- event_schema.json
    schema = json.loads((PROC / "event_schema.json").read_text(encoding="utf-8"))
    enums = schema.get("__enums__", {})
    types = [k for k in schema if k != "__enums__"]
    out["schema"] = {"n_types": len(types), "n_enums": len(enums),
                     "enum_keys": list(enums), "total_roles": sum(len(schema[t]) for t in types)}

    # ---- extract
    for split in ("train", "dev"):
        rows = load("%s_extract.jsonl" % split)
        n_span = n_arg = 0
        n_ev_types = Counter()
        dup_keys = Counter()
        for i, r in enumerate(rows):
            where = "%s_extract.jsonl#%d" % (split, i)
            for k in ("id", "sent_id", "text", "event"):
                if k not in r:
                    errors.append((where, "缺字段 " + k))
            ev = r["event"]
            for k in ("type", "trigger", "args"):
                if k not in ev:
                    errors.append((where, "缺 event." + k))
            if ev.get("type") not in types:
                errors.append((where, "未知事件类型 " + str(ev.get("type"))))
            n_ev_types[ev["type"]] += 1
            check_span(r["text"], ev["trigger"], where + ".trigger", errors)
            n_span += 1
            for j, a in enumerate(ev["args"]):
                if "role" not in a:
                    errors.append((where, "论元缺 role"))
                if a.get("role") not in schema.get(ev["type"], []):
                    errors.append((where, "论元角色不在 schema：%s.%s" % (ev["type"], a.get("role"))))
                check_span(r["text"], a, where + ".args[%d]" % j, errors)
                n_span += 1
                n_arg += 1
            dup_keys[(r["id"], r["sent_id"])] += 1
        out[split + "_extract"] = {
            "n_rows": len(rows), "n_spans_checked": n_span, "n_args": n_arg,
            "n_type": len(n_ev_types), "types": dict(n_ev_types.most_common()),
            "n_dup_id_sentid": sum(v - 1 for v in dup_keys.values() if v > 1),
            "n_empty_args_rows": sum(1 for r in rows if not r["event"]["args"]),
        }
        out[split + "_extract_keys"] = sorted(rows[0].keys()) if rows else []

    # ---- detect
    for split in ("train", "dev"):
        rows = load("%s_detect.jsonl" % split)
        for i, r in enumerate(rows):
            where = "%s_detect.jsonl#%d" % (split, i)
            for k in ("id", "sent_id", "text", "has_event", "types", "n_events"):
                if k not in r:
                    errors.append((where, "缺字段 " + k))
            if r["has_event"] != bool(r["n_events"]):
                errors.append((where, "has_event 与 n_events 不一致"))
            if bool(r["types"]) != bool(r["n_events"]):
                errors.append((where, "types 与 n_events 不一致"))
            if r["types"] != sorted(set(r["types"])):
                errors.append((where, "types 未去重/未排序"))
            if set(r["types"]) - set(types):
                errors.append((where, "未知事件类型"))
        out[split + "_detect"] = {
            "n_rows": len(rows),
            "n_positive": sum(1 for r in rows if r["has_event"]),
            "n_negative": sum(1 for r in rows if not r["has_event"]),
        }

    # ---- dev_docs
    rows = load("dev_docs.jsonl")
    n_spans = 0
    for i, r in enumerate(rows):
        where = "dev_docs.jsonl#%d" % i
        for k in ("id", "title", "text", "events"):
            if k not in r:
                errors.append((where, "缺字段 " + k))
        if not r["events"]:
            errors.append((where, "events 为空"))
        for j, ev in enumerate(r["events"]):
            if ev["type"] not in types:
                errors.append((where, "未知事件类型"))
            check_span(r["text"], ev["trigger"], where + ".events[%d].trigger" % j, errors)
            n_spans += 1
            for k, a in enumerate(ev["args"]):
                check_span(r["text"], a, where + ".events[%d].args[%d]" % (j, k), errors)
                n_spans += 1
    out["dev_docs"] = {"n_docs": len(rows), "n_events": sum(len(r["events"]) for r in rows),
                       "n_spans_checked": n_spans}

    # ---- 重复 / 泄漏
    tr = [json.loads(x)["text"] for x in open(PROC / "train_detect.jsonl", encoding="utf-8")]
    dv = [json.loads(x)["text"] for x in open(PROC / "dev_detect.jsonl", encoding="utf-8")]
    out["dedup_leakage"] = {
        "train_dup_texts": len(tr) - len(set(tr)),
        "dev_dup_texts": len(dv) - len(set(dv)),
        "train_dev_overlap": len(set(tr) & set(dv)),
    }

    out["errors"] = errors[:50]
    out["n_errors"] = len(errors)
    (STATS / "verify_stats.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2)[:4000])
    print("\n== 校验结论 ==")
    print("错误数：", len(errors))
    if errors:
        raise SystemExit(1)
    print("全部 JSONL 可逐行解析，所有 span 均满足 text[start:end] == span.text")


if __name__ == "__main__":
    main()
