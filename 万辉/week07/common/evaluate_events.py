# -*- coding: utf-8 -*-
"""
事件抽取的统一评测（全项目唯一口径，禁止各脚本各写一套）

为什么单独做一个"共享评测"：
  1. 这节课要比 6 种做法（规则 / BiLSTM-CRF / BERT / 两段式 / LLM / 组合），
     如果每个脚本自己算指标，"谁更好"这件事就不可信了 —— 常见坑是：
     - 有人用 text 完全匹配、有人用字符重叠 IoU，数字会差好几个点；
     - 有人只算 trigger、有人只算论元，看起来都叫 F1；
     - 同一个 dev 集，有的脚本过滤掉解析失败的样本（分母变小，F1 虚高）。
  2. 所以：所有脚本把预测写成同一种 JSONL，统一跑这里，输出同一张表。

评测对象（一条公告/一句话 → 0..N 个事件）：
  事件 = { 事件类型, 触发词, 论元列表[(角色, 论元文本)] }

三种粒度的指标：
  - trigger  P/R/F1：触发词文本是否抽对（不看类型）
  - argument P/R/F1：(角色 + 论元文本) 二元组是否抽对（最常用的论元级指标）
  - event    P/R/F1：一个事件"完全正确"才算对 —— 类型对、触发词文本对、论元集合完全一致
    （这是最严格的口径；PPT 上要同时给 event 和 argument，否则只看 event 会显得很低）

额外给：
  - type_macro_f1：只看"事件类型"判对没（把多事件拍平成 (文档,类型) 集合），用来对比
    "先分类再抽槽"的两段式里第一段有多准
  - per_type / per_role：定位瓶颈用（长尾类型、难角色）

输入格式（gold 与 pred 一致，id 必须能对上）：
  {"id": "...", "events": [
      {"type": "股份回购",
       "trigger": {"text": "回购", "start": 12, "end": 14},          # start/end 可缺省
       "args": [{"role": "回购方", "text": "众安集团", "start": 0, "end": 4}]}   # 同上
  ]}
  单事件样本也允许写成 {"id","event":{...}}（评测前会被包成 events 列表）。
  解析失败/空输出：pred 里 events 为空列表即可，不做额外惩罚（分母不变，F1 自然降）。
  —— 注意：**不要**把解析失败的行删掉，删了分母会变小。

用法：
  python common/evaluate_events.py --gold data/processed/dev_extract.jsonl \
                                   --pred outputs/pred_bert_crf.jsonl
  python common/evaluate_events.py --gold ... --pred ... --json outputs/metrics/bert_crf.json
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


# ─────────────────────────── 读数据 ───────────────────────────

def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def row_key(row):
    """
    样本唯一键。

    坑（课上会讲这个真实坑）：DuEE-Fin 的 `id` 是「文档 id」，不是「样本 id」——
    一篇公告能切出多句带事件的句子，它们共用同一个 id。下游如果只按 id 对齐预测，
    后面的句子会互相覆盖，指标莫名其妙掉几个点（本项目首次跑 gold-vs-gold 只有 0.93 就是这么来的）。
    所以这里优先用 uid / (id + sent_id) 做键；预测文件必须把这两个字段一起带出来。
    """
    if row.get("uid"):
        return str(row["uid"])
    if row.get("sent_id"):
        return f'{row.get("id", "")}|{row["sent_id"]}'
    return str(row.get("id", ""))


def as_event_list(row):
    """把一行样本统一成 events 列表；兼容 {"event": {...}} 单事件写法。"""
    if "events" in row and row["events"] is not None:
        evs = row["events"]
    elif "event" in row and row["event"]:
        evs = [row["event"]]
    else:
        evs = []
    out = []
    for e in evs:
        if not isinstance(e, dict):
            continue
        trig = e.get("trigger") or {}
        trig_text = (trig.get("text") if isinstance(trig, dict) else str(trig)) or ""
        args = []
        for a in e.get("args") or e.get("arguments") or []:
            if not isinstance(a, dict):
                continue
            text = a.get("text", a.get("argument", "")) or ""
            role = a.get("role", "") or ""
            if text:
                args.append((str(role).strip(), str(text).strip()))
        out.append({
            "type": str(e.get("type", e.get("event_type", ""))).strip(),
            "trigger": str(trig_text).strip(),
            # 论元用 multiset（同一角色可能重复出现），比较时按 Counter 比
            "args": args,
        })
    return out


# ─────────────────────────── 指标工具 ───────────────────────────

def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4),
            "tp": tp, "fp": fp, "fn": fn}


def compare_events(gold_events, pred_events):
    """
    逐事件对齐，返回 (trigger_tp/fp/fn, arg_tp/fp/fn, event_tp/fp/fn)。

    对齐规则（简单、可解释，课上会讲为什么这么定）：
      - 同一个事件类型下，按下标顺序一对一匹配（单事件句基本无歧义；
        多事件场景故意保持简单，因为本周主任务是"一句一个事件"）。
      - trigger：文本完全相等才算对（大小写/空格已 strip）。
      - argument：按 (角色, 文本) 的 Counter 求交集 —— 多抽的算 FP，漏的算 FN。
      - event 完全正确：类型 + trigger 文本 + 论元 Counter 全等。
    """
    g_by_type, p_by_type = defaultdict(list), defaultdict(list)
    for e in gold_events:
        g_by_type[e["type"]].append(e)
    for e in pred_events:
        p_by_type[e["type"]].append(e)

    tr_tp = tr_fp = tr_fn = 0
    ar_tp = ar_fp = ar_fn = 0
    ev_tp = ev_fp = ev_fn = 0

    for t in set(g_by_type) | set(p_by_type):
        gs, ps = g_by_type.get(t, []), p_by_type.get(t, [])
        n = max(len(gs), len(ps))
        for i in range(n):
            g = gs[i] if i < len(gs) else None
            p = ps[i] if i < len(ps) else None
            if g is None:  # 多抽了一个事件
                ev_fp += 1
                tr_fp += 1 if p["trigger"] else 0
                ar_fp += len(p["args"])
                continue
            if p is None:  # 漏了一个事件
                ev_fn += 1
                tr_fn += 1 if g["trigger"] else 0
                ar_fn += len(g["args"])
                continue
            # trigger
            if p["trigger"] and p["trigger"] == g["trigger"]:
                tr_tp += 1
            else:
                tr_fp += 1 if p["trigger"] else 0
                tr_fn += 1 if g["trigger"] else 0
            # arguments（Counter 交集）
            gc, pc = Counter(g["args"]), Counter(p["args"])
            inter = sum((gc & pc).values())
            ar_tp += inter
            ar_fp += sum(pc.values()) - inter
            ar_fn += sum(gc.values()) - inter
            # event 完全正确
            if p["trigger"] == g["trigger"] and gc == pc:
                ev_tp += 1
            else:
                ev_fp += 1
                ev_fn += 1
    return (tr_tp, tr_fp, tr_fn), (ar_tp, ar_fp, ar_fn), (ev_tp, ev_fp, ev_fn)


def type_macro_f1(gold_rows, pred_rows):
    """只看事件类型：把每篇的 (文档, 类型) 当集合，算 13 类各自的 P/R/F1 再平均。"""
    g_map, p_map = {}, {}
    for r in gold_rows:
        g_map[row_key(r)] = Counter(e["type"] for e in as_event_list(r))
    for r in pred_rows:
        p_map[row_key(r)] = Counter(e["type"] for e in as_event_list(r))
    types = set(t for c in g_map.values() for t in c) | set(t for c in p_map.values() for t in c)
    per = {}
    for t in sorted(types):
        tp = sum(min(g_map[i].get(t, 0), p_map.get(i, Counter()).get(t, 0)) for i in g_map)
        gp = sum(g_map[i].get(t, 0) for i in g_map)
        pp = sum(p_map.get(i, Counter()).get(t, 0) for i in g_map)
        per[t] = prf(tp, pp - tp, gp - tp)
    macro = sum(v["f1"] for v in per.values()) / len(per) if per else 0.0
    return round(macro, 4), per


def detection_metrics(gold_rows, pred_rows, threshold=0.5):
    """
    句子级"有没有事件"的二分类指标（给两段式 / 级联用）。
    pred 行里的 has_event 可以是 bool 或 0~1 的概率。
    """
    p_map = {row_key(r): r for r in pred_rows}
    tp = fp = tn = fn = 0
    for g in gold_rows:
        # 检测金标行自己带 has_event；抽取金标行（有 event）才用事件列表推断
        if "has_event" in g:
            gt = int(bool(g["has_event"]))
        else:
            gt = 1 if as_event_list(g) else 0
        p = p_map.get(row_key(g))
        if p is None:
            pt = 0  # 没预测 = 判成"无事件"
        else:
            v = p.get("has_event", 0)
            pt = int(float(v) >= threshold) if not isinstance(v, bool) else int(v)
        if gt and pt: tp += 1
        elif gt and not pt: fn += 1
        elif not gt and pt: fp += 1
        else: tn += 1
    out = prf(tp, fp, fn)
    out["accuracy"] = round((tp + tn) / max(1, tp + tn + fp + fn), 4)
    out["confusion"] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn}
    return out


# ─────────────────────────── 主流程 ───────────────────────────

def evaluate(gold_rows, pred_rows, verbose=True):
    """统一入口：返回可直接 json.dump 的结果字典（其它脚本 import 这个函数）。"""
    p_map = {row_key(r): r for r in pred_rows}
    missing = 0
    tr = [0, 0, 0]
    ar = [0, 0, 0]
    ev = [0, 0, 0]
    per_role_tp, per_role_gp, per_role_pp = Counter(), Counter(), Counter()
    per_type_tp, per_type_gp, per_type_pp = Counter(), Counter(), Counter()

    for g in gold_rows:
        gold_events = as_event_list(g)
        p = p_map.get(row_key(g))
        if p is None:
            missing += 1  # 预测文件里缺这条（比如跑挂了），按"全漏"处理，不删分母
            pred_events = []
        else:
            pred_events = as_event_list(p)

        t, a, e = compare_events(gold_events, pred_events)
        for i in range(3):
            tr[i] += t[i]; ar[i] += a[i]; ev[i] += e[i]

        # per-type（事件级精确率/召回率，用于定位长尾类型）
        gc = Counter(x["type"] for x in gold_events)
        pc = Counter(x["type"] for x in pred_events)
        for ty in set(gc) | set(pc):
            per_type_tp[ty] += min(gc[ty], pc[ty])
            per_type_gp[ty] += gc[ty]
            per_type_pp[ty] += pc[ty]
        # per-role（论元级，定位难角色）
        gargs = Counter((x["type"], r, txt) for x in gold_events for (r, txt) in x["args"])
        pargs = Counter((x["type"], r, txt) for x in pred_events for (r, txt) in x["args"])
        for key in gargs:
            per_role_gp[key[1]] += gargs[key]
        for key in pargs:
            per_role_pp[key[1]] += pargs[key]
        for key in set(gargs) | set(pargs):
            per_role_tp[key[1]] += min(gargs[key], pargs[key])

    res = {
        "num_samples": len(gold_rows),
        "missing_predictions": missing,
        "trigger": prf(*tr),
        "argument": prf(*ar),
        "event_exact": prf(*ev),
    }

    per_type = {}
    for ty in sorted(per_type_gp):
        per_type[ty] = prf(per_type_tp[ty], per_type_pp[ty] - per_type_tp[ty],
                           per_type_gp[ty] - per_type_tp[ty])
    per_role = {}
    for role in sorted(per_role_gp):
        per_role[role] = prf(per_role_tp[role], per_role_pp[role] - per_role_tp[role],
                             per_role_gp[role] - per_role_tp[role])
    res["per_type"] = per_type
    res["per_role"] = per_role
    res["type_macro_f1"], _ = type_macro_f1(gold_rows, pred_rows)

    if verbose:
        print(f"样本数 {res['num_samples']}（缺预测 {missing} 条）")
        for k in ("trigger", "argument", "event_exact"):
            m = res[k]
            print(f"  {k:12s} P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f} "
                  f"(tp={m['tp']} fp={m['fp']} fn={m['fn']})")
        print(f"  type_macro_f1 = {res['type_macro_f1']:.4f}")
    return res


def main():
    ap = argparse.ArgumentParser(description="事件抽取统一评测（trigger / argument / event 三级指标）")
    ap.add_argument("--gold", required=True, help="金标 JSONL：dev_extract.jsonl")
    ap.add_argument("--pred", required=True, help="预测 JSONL：每行 {id, events:[...]}")
    ap.add_argument("--json", default="", help="把结果写到这个 json（建议 outputs/metrics/xxx.json）")
    ap.add_argument("--detect", action="store_true",
                    help="额外算句子级'有没有事件'的二分类指标（pred 需带 has_event）")
    ap.add_argument("--gold_detect", default="", help="配合 --detect 的检测金标（默认 dev_detect.jsonl 同目录）")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    gold = load_jsonl(args.gold)
    pred = load_jsonl(args.pred)
    res = evaluate(gold, pred, verbose=not args.quiet)

    if args.detect:
        gd = Path(args.gold_detect) if args.gold_detect else Path(args.gold).with_name("dev_detect.jsonl")
        if gd.exists():
            g_rows, p_rows = load_jsonl(gd), pred
            # 预测文件一般只含正样本句；缺的按"无事件"处理，所以这里用并集
            known = {row_key(r) for r in p_rows}
            for g in g_rows:
                if row_key(g) not in known:
                    p_rows = p_rows + [{"uid": row_key(g), "has_event": 0}]
            res["detection"] = detection_metrics(g_rows, p_rows)
            if not args.quiet:
                d = res["detection"]
                print(f"  detection    P={d['precision']:.4f} R={d['recall']:.4f} "
                      f"F1={d['f1']:.4f} acc={d['accuracy']:.4f}")
        else:
            print(f"[warn] 找不到检测金标 {gd}，跳过 detection")

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        print("written:", args.json)


if __name__ == "__main__":
    main()

