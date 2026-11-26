# -*- coding: utf-8 -*-
"""原始数据对齐审计：DuEE-Fin 句子级（sentence/train|dev.json）自带 index 的真实性质。

要回答三个问题（报告“offset 不可信”一节直接引用）：
  1. 原始 trigger_start_index / argument_start_index 能否与 text 自洽？
  2. 它们是不是“按首次出现回填”的？——统计与 text.find() 不一致的条数 + 同串多次出现比例；
  3. 有多少论元根本没有 index（归一化取值，如 环节=筹备上市）。

输出：stats/audit_raw.json
"""
import json
from collections import Counter
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent
RAW = DATA / "raw" / "DuEE_Fin_mirror"
STATS = DATA / "stats"
FILES = {
    "train": RAW / "DuEE-pytorch__data__DuEE-Fin__sentence__train.json",
    "dev": RAW / "DuEE-pytorch__data__DuEE-Fin__sentence__dev.json",
}


def iter_lines(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def audit(split):
    n_events = 0
    n_span = n_index = n_index_ok = n_index_diff_from_find = 0
    n_no_index = n_no_index_found = n_no_index_missing = 0
    n_ambiguous = 0
    missing_by_role = Counter()
    diff_examples = []
    for row in iter_lines(FILES[split]):
        text = row["text"]
        for ev in row.get("event_list") or []:
            n_events += 1
            spans = [("__trigger__", ev["trigger"], ev.get("trigger_start_index"))]
            spans += [(a["role"], a["argument"], a.get("argument_start_index"))
                      for a in ev.get("arguments") or []]
            for role, span, idx in spans:
                n_span += 1
                first = text.find(span)
                if len([i for i in range(len(text)) if text.startswith(span, i)]) > 1:
                    n_ambiguous += 1
                if idx is None:
                    n_no_index += 1
                    missing_by_role[role] += 1
                    if first >= 0:
                        n_no_index_found += 1
                    else:
                        n_no_index_missing += 1
                    continue
                n_index += 1
                if text[idx:idx + len(span)] == span:
                    n_index_ok += 1
                if first != idx:
                    n_index_diff_from_find += 1
                    if len(diff_examples) < 5:
                        diff_examples.append([text[:60], span, idx, first])
    return {
        "n_events": n_events,
        "n_spans": n_span,
        "n_span_with_index": n_index,
        "n_span_index_consistent": n_index_ok,
        "n_span_index_differs_from_first_find": n_index_diff_from_find,
        "n_span_without_index": n_no_index,
        "n_no_index_recoverable_by_find": n_no_index_found,
        "n_no_index_not_in_sentence": n_no_index_missing,
        "n_spans_string_ambiguous": n_ambiguous,
        "no_index_by_role": dict(missing_by_role.most_common()),
        "diff_examples": diff_examples,
    }


if __name__ == "__main__":
    STATS.mkdir(parents=True, exist_ok=True)
    out = {split: audit(split) for split in ("train", "dev")}
    (STATS / "audit_raw.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    for split, d in out.items():
        print(split, json.dumps(d, ensure_ascii=False)[:700])
