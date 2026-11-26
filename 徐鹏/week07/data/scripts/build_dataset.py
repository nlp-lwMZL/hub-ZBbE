# -*- coding: utf-8 -*-
"""按已定契约，把 DuEE-Fin（GitHub 公开镜像 maopademiao/duee_fin 的 DuEE-pytorch/data/DuEE-Fin/）
处理成 week7 事件抽取教学用的四类数据文件 + 事件 schema。

为什么以 DuEE-Fin 为主数据源：
  - 金融公告域，13 个事件类型（质押/回购/减持/增持/收购/破产/上市/融资/中标/亏损/高管变动/被约谈/解除质押），
    与「八斗投研工作台」的公告抽取场景完全同域；
  - 句子级版本（sentence/train|dev.json）中位句长 ~102 字，适合 BERT 512 截断与课堂演示。

产出的固定契约（写入 data/processed/）：
  1) event_schema.json          {"事件类型": [角色, ...], ..., "__enums__": {"事件类型.角色": [枚举值]}}
  2) train_extract.jsonl       单事件句子（event_list 恰好 1 个），带 trigger/论元字符级 span
  3) dev_extract.jsonl         同上，dev 侧
  4) train_detect.jsonl        全量句子（含无事件负样本），句子级“有没有事件/哪些类型”
  5) dev_detect.jsonl
  6) dev_docs.jsonl            篇章级 dev 中有事件的文档（端到端系统演示用）

span 口径（报告必写）：
  - 字符级 offset（Python str 下标），左闭右开，text[start:end] == span.text；
  - 一律用 text.find() 在「所属句子/篇章」内自行定位，**不采信原始 argument_start_index**
    （镜像里的 index 是按首次出现回填的，见 build_stats.json 里的对齐审计）；
  - 同一字符串在文本中多次出现时取第一次出现的位置，这是本数据集的已知局限；
  - 定位不到的论元丢弃并计数；trigger 定位不到的样本整条丢弃并计数（预期 0）。

复现命令：
  D:\\Anaconda\\python.exe scripts\\build_dataset.py
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent
RAW = DATA / "raw" / "DuEE_Fin_mirror"
PROC = DATA / "processed"
STATS = DATA / "stats"
SENT = {
    "train": RAW / "DuEE-pytorch__data__DuEE-Fin__sentence__train.json",
    "dev": RAW / "DuEE-pytorch__data__DuEE-Fin__sentence__dev.json",
}
DOC = {
    "train": RAW / "DuEE-pytorch__data__DuEE-Fin__train.json",
    "dev": RAW / "DuEE-pytorch__data__DuEE-Fin__dev.json",
}
SCHEMA_PATH = RAW / "DuEE-pytorch__conf__DuEE-Fin__event_schema.json"


def iter_lines(path):
    """DuEE-Fin 的 json 文件实际是 JSONL（每行一个 JSON 对象），逐行读。"""
    with open(path, encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            for row in json.load(f):
                yield row
        else:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)


def find_span(text, span_text):
    """在 text 中定位 span_text，返回 (start, end) 或 None；多次出现取第一次。

    为什么不用 str.index：index 抛异常，需要 try/except；find 返回 -1 更直观。
    """
    pos = text.find(span_text)
    if pos < 0:
        return None
    return pos, pos + len(span_text)


def pct(values):
    """分位数摘要（报告里所有“长度/论元数”数字都从这里来，口径统一）。"""
    if not values:
        return {}
    s = sorted(values)
    return {"n": len(s), "mean": round(sum(s) / len(s), 2), "p50": s[len(s) // 2],
            "p90": s[int(len(s) * 0.9)], "p99": s[int(len(s) * 0.99)], "max": s[-1]}


def dump_jsonl(path, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------- 1) schema
def build_schema():
    """13 行 jsonl → 事件类型→角色列表；enum 单独放 __enums__，避免和事件类型混淆。"""
    type_roles, enums = {}, {}
    for row in iter_lines(SCHEMA_PATH):
        et = row["event_type"]
        roles = []
        for item in row["role_list"]:
            roles.append(item["role"])
            if item.get("enum_items"):
                enums["%s.%s" % (et, item["role"])] = item["enum_items"]
        type_roles[et] = roles
    type_roles["__enums__"] = enums
    (PROC / "event_schema.json").write_text(
        json.dumps(type_roles, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return type_roles


# ---------------------------------------------------------------- 2/3) extract
def build_extract(sentence_rows, split, stats):
    """单事件句子 → 带 span 的训练样本；统计 trigger/论元定位率与丢弃量。"""
    rows_out = []
    n_all = len(sentence_rows)
    n_one = 0
    n_sample_dropped = 0
    n_args_total = n_args_kept = 0
    drop_by_role = Counter()
    ambig_spans = n_spans = 0
    lens_extract = []
    args_per_ev = []
    trig_lens = []
    n_trigger_eq_arg = 0
    n_overlap_events = 0
    for row in sentence_rows:
        evs = row.get("event_list") or []
        if len(evs) != 1:
            continue
        n_one += 1
        text = row["text"]
        ev = evs[0]
        trig = find_span(text, ev["trigger"])
        if trig is None:  # 理论上不会发生（trigger 都在句内）
            n_sample_dropped += 1
            continue
        n_spans += 1
        if text.count(ev["trigger"]) > 1:
            ambig_spans += 1
        args = []
        for a in ev.get("arguments") or []:
            n_args_total += 1
            role, arg = a["role"], a["argument"]
            sp = find_span(text, arg)
            if sp is None:  # 论元是“归一化取值”（原文没有这串字）或落在别的句子里
                drop_by_role[role] += 1
                continue
            n_spans += 1
            if text.count(arg) > 1:
                ambig_spans += 1
            args.append({"role": role, "text": arg, "start": sp[0], "end": sp[1]})
            n_args_kept += 1
        # 教学坑：同一句里 trigger 与论元、论元与论元可能落在同一/重叠区间（如 高管变动 的"担任"
        # 既是触发器又是 变动类型 论元），做 BIO 序列标注时会互相覆盖，必须定优先级。
        spans_all = [(trig[0], trig[1])] + [(a["start"], a["end"]) for a in args]
        if any((a["start"], a["end"]) == (trig[0], trig[1]) for a in args):
            n_trigger_eq_arg += 1
        overlap = False
        for i in range(len(spans_all)):
            for j in range(i + 1, len(spans_all)):
                x, y = spans_all[i], spans_all[j]
                if x[0] < y[1] and y[0] < x[1]:
                    overlap = True
        if overlap:
            n_overlap_events += 1
        rows_out.append({
            "id": row.get("id"),
            "sent_id": row.get("sent_id"),
            "text": text,
            "event": {
                "type": ev["event_type"],
                "trigger": {"text": ev["trigger"], "start": trig[0], "end": trig[1]},
                "args": args,
            },
        })
        lens_extract.append(len(text))
        args_per_ev.append(len(args))
        trig_lens.append(len(ev["trigger"]))
    stats["extract_" + split] = {
        "n_sentence_rows": n_all,
        "n_rows_with_exactly_1_event": n_one,
        "n_output_rows": len(rows_out),
        "n_samples_dropped_trigger_not_found": n_sample_dropped,
        "n_args_total": n_args_total,
        "n_args_kept": n_args_kept,
        "n_args_dropped": n_args_total - n_args_kept,
        "arg_drop_rate": round((n_args_total - n_args_kept) / max(1, n_args_total), 6),
        "arg_drop_by_role": dict(drop_by_role.most_common()),
        "n_spans": n_spans,
        "n_spans_string_ambiguous": ambig_spans,
        "span_ambiguity_rate": round(ambig_spans / max(1, n_spans), 6),
        "n_events_trigger_equals_arg_span": n_trigger_eq_arg,
        "n_events_with_any_span_overlap": n_overlap_events,
        "sentence_len": pct(lens_extract),
        "trigger_len": pct(trig_lens),
        "args_per_event": pct(args_per_ev),
    }
    dump_jsonl(PROC / ("%s_extract.jsonl" % split), rows_out)
    return rows_out


# ---------------------------------------------------------------- 4/5) detect
def build_detect(sentence_rows, split, stats):
    """全部句子 → 句子级事件检测样本（含负样本）。"""
    rows_out = []
    n_ev_hist = Counter()
    type_dist = Counter()
    lens = []
    for row in sentence_rows:
        evs = row.get("event_list") or []
        types = sorted({ev["event_type"] for ev in evs})
        rows_out.append({
            "id": row.get("id"),
            "sent_id": row.get("sent_id"),
            "text": row["text"],
            "has_event": bool(evs),
            "types": types,
            "n_events": len(evs),
        })
        lens.append(len(row["text"]))
        n_ev_hist[len(evs)] += 1
        for t in types:
            type_dist[t] += 1
    stats["detect_" + split] = {
        "n_rows": len(rows_out),
        "n_positive": sum(1 for r in rows_out if r["has_event"]),
        "n_negative": sum(1 for r in rows_out if not r["has_event"]),
        "positive_rate": round(sum(1 for r in rows_out if r["has_event"]) / max(1, len(rows_out)), 4),
        "n_events_hist": dict(sorted(n_ev_hist.items())),
        "text_len": pct(lens),
        "type_distribution": dict(type_dist.most_common()),
    }
    dump_jsonl(PROC / ("%s_detect.jsonl" % split), rows_out)
    return rows_out


# ---------------------------------------------------------------- 6) dev docs
def build_dev_docs(stats):
    """篇章级 dev 中有事件的文档；span 在整篇 text 内 find 定位。"""
    rows_out = []
    n_all = 0
    n_with_ev = 0
    n_trig_missing = 0
    n_docs_all_events_dropped = 0
    n_args_total = n_args_kept = 0
    drop_by_role = Counter()
    doc_lens = []
    title_dup = 0
    for row in iter_lines(DOC["dev"]):
        n_all += 1
        text = row["text"]
        title = row.get("title") or ""
        # 公告正文常常把标题原样抄在首段（"原标题：xxx"），统计这种重复率，作为"title 与正文重复"的坑
        if title and title in text:
            title_dup += 1
        evs = row.get("event_list") or []
        if not evs:
            continue
        n_with_ev += 1
        events = []
        for ev in evs:
            trig = find_span(text, ev["trigger"])
            if trig is None:
                n_trig_missing += 1
                continue
            args = []
            for a in ev.get("arguments") or []:
                n_args_total += 1
                sp = find_span(text, a["argument"])
                if sp is None:
                    drop_by_role[a["role"]] += 1
                    continue
                args.append({"role": a["role"], "text": a["argument"], "start": sp[0], "end": sp[1]})
                n_args_kept += 1
            events.append({
                "type": ev["event_type"],
                "trigger": {"text": ev["trigger"], "start": trig[0], "end": trig[1]},
                "args": args,
            })
        if not events:
            # 整篇所有事件的 trigger 都在正文里定位不到（标题式/归一化 trigger，或正文残缺），放弃该篇，
            # 保证交付的 dev_docs 每篇都至少有 1 个可抽取事件（下游端到端演示需要）。
            n_docs_all_events_dropped += 1
            continue
        rows_out.append({"id": row.get("id"), "title": row.get("title"), "text": text, "events": events})
        doc_lens.append(len(text))
    doc_lens.sort()
    stats["dev_docs"] = {
        "n_doc_rows": n_all,
        "n_docs_with_event": n_with_ev,
        "n_docs_without_event": n_all - n_with_ev,
        "n_output_docs": len(rows_out),
        "n_docs_all_events_dropped": n_docs_all_events_dropped,
        "n_events_in_output": sum(len(r["events"]) for r in rows_out),
        "n_trigger_missing": n_trig_missing,
        "n_args_total": n_args_total,
        "n_args_kept": n_args_kept,
        "n_args_dropped": n_args_total - n_args_kept,
        "arg_drop_rate": round((n_args_total - n_args_kept) / max(1, n_args_total), 6),
        "arg_drop_by_role": dict(drop_by_role.most_common()),
        "doc_len": {
            "p50": doc_lens[len(doc_lens) // 2],
            "p90": doc_lens[int(len(doc_lens) * 0.9)],
            "p99": doc_lens[int(len(doc_lens) * 0.99)],
            "max": doc_lens[-1],
            "mean": round(sum(doc_lens) / len(doc_lens), 1),
        },
        "n_docs_title_appears_in_text": title_dup,
    }
    dump_jsonl(PROC / "dev_docs.jsonl", rows_out)
    return rows_out


# ---------------------------------------------------------------- 主流程
def main():
    PROC.mkdir(parents=True, exist_ok=True)
    STATS.mkdir(parents=True, exist_ok=True)
    stats = {}
    schema = build_schema()
    stats["schema"] = {
        "n_event_types": len([k for k in schema if k != "__enums__"]),
        "n_roles_per_type": {k: len(v) for k, v in schema.items() if k != "__enums__"},
        "n_enum_roles": len(schema.get("__enums__", {})),
        "enums": schema.get("__enums__", {}),
    }

    sent = {split: list(iter_lines(SENT[split])) for split in ("train", "dev")}
    extract = {split: build_extract(sent[split], split, stats) for split in ("train", "dev")}
    build_detect(sent["train"], "train", stats)
    build_detect(sent["dev"], "dev", stats)

    # 单/多事件比例（句子级，供报告“为什么只取单事件子集”）
    for split in ("train", "dev"):
        hist = Counter(len(r.get("event_list") or []) for r in sent[split])
        stats["sentence_events_hist_" + split] = dict(sorted(hist.items()))
        stats["sentence_events_hist_" + split]["_n_rows"] = len(sent[split])

    # 类型 / 角色分布（extract 子集，用于图表与报告）
    def dist(rows):
        types, roles = Counter(), Counter()
        for r in rows:
            types[r["event"]["type"]] += 1
            for a in r["event"]["args"]:
                roles[a["role"]] += 1
        return dict(types.most_common()), dict(roles.most_common())

    stats["extract_train_type_distribution"], stats["extract_train_role_distribution"] = dist(extract["train"])
    stats["extract_dev_type_distribution"], _ = dist(extract["dev"])

    dev_docs = build_dev_docs(stats)

    # 全部句子级事件的类型分布（对比：过滤前后是否偏）
    all_types = Counter()
    for split in ("train", "dev"):
        for r in sent[split]:
            for ev in r.get("event_list") or []:
                all_types[ev["event_type"]] += 1
    stats["sentence_all_type_distribution"] = dict(all_types.most_common())

    # 重复 / 泄漏检查
    tr = [r["text"].strip() for r in sent["train"]]
    dv = [r["text"].strip() for r in sent["dev"]]
    stats["dedup_leakage"] = {
        "train_dup_texts": len(tr) - len(set(tr)),
        "dev_dup_texts": len(dv) - len(set(dv)),
        "train_dev_overlap": len(set(tr) & set(dv)),
        "dev_docs_dup_texts": len(dev_docs) - len({r["text"] for r in dev_docs}),
    }
    # 文档 id 在 sentence train/dev 间是否重叠（同一篇公告被切到两侧 = 泄漏）
    ids_tr = {r.get("id") for r in sent["train"]}
    ids_dv = {r.get("id") for r in sent["dev"]}
    stats["dedup_leakage"]["doc_id_train_dev_overlap"] = len(ids_tr & ids_dv)

    # 篇长 / 每篇事件数分布（doc train，用于图表）
    doc_train_lens, doc_train_ev = [], []
    for row in iter_lines(DOC["train"]):
        doc_train_lens.append(len(row["text"]))
        doc_train_ev.append(len(row.get("event_list") or []))
    ds = sorted(doc_train_lens)
    stats["doc_train"] = {
        "n_docs": len(doc_train_lens),
        "n_docs_with_event": sum(1 for x in doc_train_ev if x),
        "events_per_doc_hist": dict(sorted(Counter(doc_train_ev).items())),
        "doc_len": {"p50": ds[len(ds) // 2], "p90": ds[int(len(ds) * 0.9)],
                    "p99": ds[int(len(ds) * 0.99)], "max": ds[-1],
                    "mean": round(sum(ds) / len(ds), 1)},
    }

    # 文件体积（验收：处理后 ≤30MB）
    sizes = {}
    for p in sorted(PROC.iterdir()):
        sizes[p.name] = round(p.stat().st_size / 1024 / 1024, 2)
    stats["processed_file_size_mb"] = sizes
    stats["processed_total_mb"] = round(sum(sizes.values()), 2)

    (STATS / "build_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print("schema types:", stats["schema"]["n_event_types"], "| enum roles:", stats["schema"]["n_enum_roles"])
    for split in ("train", "dev"):
        e = stats["extract_" + split]
        d = stats["detect_" + split]
        print("%s: extract rows=%d (可抽取=%d, trigger丢弃=%d, 论元 %d/%d 保留, 丢弃率 %.2f%%) | detect rows=%d (pos=%d)" % (
            split, e["n_output_rows"], e["n_rows_with_exactly_1_event"], e["n_samples_dropped_trigger_not_found"],
            e["n_args_kept"], e["n_args_total"], 100 * e["arg_drop_rate"], d["n_rows"], d["n_positive"]))
    print("dev_docs:", json.dumps({k: v for k, v in stats["dev_docs"].items() if not isinstance(v, dict)}, ensure_ascii=False))
    print("processed total MB:", stats["processed_total_mb"])
    print("leakage:", json.dumps(stats["dedup_leakage"], ensure_ascii=False))


if __name__ == "__main__":
    main()
