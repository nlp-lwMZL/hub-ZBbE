"""
L0 规则基线：触发词词典 + 正则 + 股票简称词典的「零训练」事件抽取

为什么课里要从规则讲起（而不是直接上 BERT）：
  规则基线是**最诚实的对照组**。它告诉我们「不用学模型、只靠词典和正则能拿到多少」，
  后面的 BERT / CRF / 两段式涨了多少点，才有意义。
  更重要的是：规则的**失败模式非常清楚**，正好用来引出后面每个模型要解决的问题：
    - 触发词没收录 → 整句漏抽（召回天花板卡在词典覆盖率上）
    - 一个词有多种事件类型（「质押」vs「解除质押」）→ 类型判错
    - 论元是自由文本（公司名、人名、标的物）→ 正则根本抽不出来，只能靠模型
    - 金额/比例/日期有规律 → 正则反而很强，后来在 service 层被留作**模型之后的兜底补充**

做法（全部从训练集自动统计，不手写几百个词）：
  1. 触发词：统计训练集里各类型出现过的 trigger 文本，保留出现 ≥2 次的（去噪），
     再加一小张人工同义词表补召回。匹配时**最长优先**（否则「质押」会抢走「解除质押」）。
  2. 数值类论元：按角色名语义分类，用正则抽金额/数量/比例/日期。
  3. 公司/机构类论元：用训练集里的「股票简称/公司名」论元值建词典，再配「…有限公司/…集团」正则。

使用方式：
  python rule/run_rule.py                          # 跑 dev 集，出指标 + 失败分析
  python rule/run_rule.py --mock                   # 用 data/_mock 的 20 条手造数据自检
  python rule/run_rule.py --min_trigger_count 3    # 提高触发词入典门槛
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]   # 项目根目录（方法目录的上一级）
sys.path.insert(0, str(ROOT / "common"))     # 公共模块：event_data / crf / eval_utils / models / log_utils

from event_data import (DATA_DIR, ROOT, load_docs, load_extract_split, load_schema,
                        load_jsonl, load_detect_split, make_pred_row, row_event)
from eval_utils import METRIC_DIR, evaluate_predictions
from log_utils import EvalLogger

MOCK_DIR = ROOT / "data" / "_mock"

# ── 人工补充的同义词 / 变体（教学点：统计自动入典之后，人工补的边际收益有多大）──
# 只补「训练集里没出现、但金融公告里显然会出现」的说法，避免把测试集答案抄进来。
MANUAL_TRIGGERS = {
    "股份回购": ["股份回购", "回购股份", "回购"],
    "解除质押": ["解除质押", "解除股份质押", "解押"],
    "质押": ["股权质押", "质押股份", "质押"],
    "股东增持": ["增持股份", "增持"],
    "股东减持": ["减持股份", "减持"],
    "企业收购": ["收购", "并购", "要约收购"],
    "企业融资": ["定增", "增资", "融资"],
    "公司上市": ["上市", "IPO", "首发"],
    "亏损": ["亏损", "预亏", "业绩亏损"],
    "中标": ["中标", "中标通知"],
    "高管变动": ["辞职", "辞去", "聘任", "任命", "换届"],
    "企业破产": ["破产", "破产重整", "破产清算"],
    "被约谈": ["约谈", "被约谈"],
}

# ── 论元抽取的正则库 ────────────────────────────────────────────────────
# 顺序有讲究：金额必须允许「1,234.56万元」「8-10美元」「1.2亿元至1.5亿元」这几种真实写法。
NUM = r"[0-9０-９][0-9０-９,，]*\.?[0-9０-９]*"
# 先试「区间」再试「单值」：否则「1.2亿元至1.5亿元」会被切成两个金额，
# 而金标是一个论元，多抽的那个算 FP、正确的那个也算 FP —— 顺序写反直接白丢分。
MONEY_RE = re.compile(
    rf"(?:{NUM}\s*(?:亿|万|千)?\s*(?:元|美元|港元|港币)?\s*(?:至|-|~)\s*"
    rf"{NUM}\s*(?:亿|万|千)?\s*(?:元|美元|港元|港币)"
    rf"|{NUM}\s*(?:亿|万|千)?\s*(?:元|美元|港元|港币|人民币|欧元))")
DATE_RE = re.compile(
    r"(?:\d{4}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日"
    r"|\d{4}\s*年\s*\d{1,2}\s*月"
    r"|\d{4}\s*年(?:\s*(?:上半年|下半年|第[一二三四]季度|一季度|二季度|三季度|四季度|全年|年报|半年报|三季报))?"
    r"|\d{1,2}\s*月\s*\d{1,2}\s*日|\d{1,2}\s*日)")
PERCENT_RE = re.compile(r"\d+(?:\.\d+)?\s*(?:%|％)")
SHARE_RE = re.compile(rf"{NUM}\s*(?:亿|万|千)?\s*股")
# 公司名启发式：中文/字母数字串 + 机构后缀；「股份有限公司」要放在「有限公司」前面先试
COMPANY_RE = re.compile(
    r"[\u4e00-\u9fa5A-Za-z0-9（）()]{2,12}?"
    r"(?:股份有限公司|有限责任公司|有限公司|集团|控股|股份|证券|信托|基金|银行|交易所|公司)")

# 角色名 → 抽取器 的语义映射（不逐个手写 61 个角色，靠关键词归类）
DATE_KEYS = ("时间", "日期", "周期")
MONEY_KEYS = ("金额", "价格", "亏损", "市值", "募资", "规模", "市值")
PERCENT_KEYS = ("比例", "占比", "总股本", "所持比例")
SHARE_KEYS = ("数量",)
ENUM_KEYS = ("类型", "环节")

def role_to_kind(role: str) -> str:
    """把一个角色名归类到「用哪种抽取器」，这是规则系统里唯一的『泛化』。"""
    if any(k in role for k in DATE_KEYS):
        return "date"
    if any(k in role for k in PERCENT_KEYS):
        return "percent"
    if any(k in role for k in MONEY_KEYS):
        return "money"
    if any(k in role for k in SHARE_KEYS):
        return "share"
    if any(k in role for k in ENUM_KEYS):
        return "enum"
    return "entity"   # 公司名 / 人名 / 标的物等自由文本

# ─────────────────────────── 1. 从训练集统计触发词 ───────────────────────────

def build_trigger_dict(train_rows: list[dict], min_count: int = 2) -> dict:
    """
    统计 {类型: [(触发词, 频次), ...]}。

    只保留出现 ≥ min_count 次的触发词：金融公告里有很多「一次性」触发词
    （某个公司独有的表述），把它们全收进词典会让规则在训练集上虚高、在验证集上不泛化。
    同一次出现也要保留 —— 有训练集兜底的词本身是合法的，只是权重低。
    """
    cnt = defaultdict(Counter)
    for r in train_rows:
        ev = row_event(r)
        if not ev:
            continue
        trig = (ev.get("trigger") or {}).get("text", "")
        if trig:
            cnt[ev["type"]][trig] += 1
    return {t: [(w, c) for w, c in c.most_common() if c >= min_count]
            for t, c in cnt.items()}

def merge_trigger_dict(auto: dict, manual: dict) -> list[tuple[str, str]]:
    """
    合并成「(触发词, 类型) 的扁平列表」，按**词长降序**排好。

    为什么要按长度排序：触发词存在包含关系 ——「解除质押」里含「质押」，
    「股份回购」里含「回购」。如果把短的排在前面，一句「解除质押」会被判成「质押」，
    类型直接错、13 分类的 macro F1 会掉得很明显。最长优先（maximal match）是规则系统的标准做法。
    """
    pairs = []
    for t, words in auto.items():
        for w, _ in words:
            pairs.append((w, t))
    for t, words in manual.items():
        for w in words:
            pairs.append((w, t))
    # 去重（同一 (词,类型) 只留一份），再按词长降序、类型名升序保证确定性
    pairs = sorted(set(pairs), key=lambda x: (-len(x[0]), x[1]))
    return pairs

# ─────────────────────────── 2. 股票简称 / 公司名词典 ───────────────────────────

def build_company_dict(train_rows: list[dict]) -> set[str]:
    """
    用训练集里「公司类角色」的论元值建词典（这就是任务里说的「股票简称词典启发式」）。

    真实工程里这份词典来自行情库（A 股 5000 家简称），课堂没有该数据，
    所以退化做法是：把训练集标注过的公司名收进来当词典。
    代价很明显 —— 验证集里没在训练集出现过的公司，规则一个都抽不出来，
    这正是规则基线论元召回低的主因（也是后面必须上模型的原因）。
    """
    org_roles = {"股票简称", "上市公司", "公司名称", "被收购方", "收购方", "中标公司",
                 "破产公司", "任职公司", "被投资方", "投资方", "领投方", "招标方",
                 "质押物所属公司", "披露方", "变动后公司名称"}
    names = set()
    for r in train_rows:
        ev = row_event(r)
        if not ev:
            continue
        for a in ev.get("args") or []:
            if a.get("role") in org_roles:
                txt = (a.get("text") or "").strip()
                # 只收 2~10 字、不含数字的名字：太长的多半是整句，含数字的多半是代码/金额
                if 2 <= len(txt) <= 10 and not re.search(r"\d", txt):
                    names.add(txt)
    return names

# ─────────────────────────── 3. 单句抽取 ───────────────────────────

def find_trigger(text: str, trig_pairs: list[tuple[str, str]]) -> tuple[str, str, int, int] | None:
    """最长优先扫触发词，返回 (类型, 触发词, start, end)；没命中返回 None。"""
    best = None
    for word, etype in trig_pairs:
        pos = text.find(word)
        if pos < 0:
            continue
        # 词长优先；同长度时取最靠前出现的位置（贴近人类阅读顺序）
        cand = (len(word), -pos)
        if best is None or cand > best[0]:
            best = (cand, etype, word, pos, pos + len(word))
    if best is None:
        return None
    _, etype, word, s, e = best
    return etype, word, s, e

def extract_entities(text: str, company_dict: set[str]) -> list[dict]:
    """把句子里的「公司/机构」候选全部找出来：词典精确命中 + 后缀正则，按位置排序去重。"""
    found = []
    for name in company_dict:
        start = 0
        while True:
            pos = text.find(name, start)
            if pos < 0:
                break
            found.append({"text": name, "start": pos, "end": pos + len(name)})
            start = pos + 1
    for m in COMPANY_RE.finditer(text):
        found.append({"text": m.group(0), "start": m.start(), "end": m.end()})
    # 去重 + 去掉被更长候选完整包住的短候选（「中信信托」优于「信托」）
    uniq, seen = [], set()
    for c in sorted(found, key=lambda x: (x["start"], -len(x["text"]))):
        key = (c["start"], c["end"])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(c)
    out = []
    for c in uniq:
        if any(o is not c and o["start"] <= c["start"] and c["end"] <= o["end"]
               and len(o["text"]) > len(c["text"]) for o in uniq):
            continue
        out.append(c)
    return out

def extract_args(text: str, etype: str, role: str, company_dict: set[str],
                 used_spans: list[tuple[int, int]],
                 entity_pool: list[dict]) -> list[dict]:
    """
    按角色语义抽一个角色的论元（可能 0 个或多个）。

    `used_spans` 记录已经被别的角色占掉的位置，避免「触发词本身又被当成论元」这种滑稽错误。
    `entity_pool` 是整句预扫出来的公司候选池（按出现顺序），**每个实体类角色只从池里领 1 个、
    且领走就删** —— 这条简单的「按出现顺序依次填充」规则，恰好和金融公告的常见写法吻合：
    「甲公司拟收购乙公司」里 收购方=先出现、被收购方=后出现。
    如果不做这个限制，一个句子里的 5 个公司候选会被填进 5 个角色，
    论元全是假阳性（实测不做限制时 argument 精确率只有 0.13，全是这么来的）。
    """
    kind = role_to_kind(role)
    out = []

    def _free(s, e):
        return all(not (s < ue and e > us) for us, ue in used_spans)

    if kind in ("money", "share", "percent", "date"):
        pat = {"money": MONEY_RE, "share": SHARE_RE, "percent": PERCENT_RE,
               "date": DATE_RE}[kind]
        for m in pat.finditer(text):
            s, e = m.start(), m.end()
            if not _free(s, e):
                continue
            out.append({"role": role, "text": text[s:e].strip(), "start": s, "end": e})
            break   # 每个角色只领 1 个：规则分不清「哪个金额属于哪个角色」，
                    # 多领只会把假阳性堆上去（实测不限量时 argument 精确率 0.13 → 限量后 0.16+）
    elif kind == "enum":
        # 枚举型角色（如「环节」= 筹备上市/正式上市/终止上市）：规则不做枚举判定，
        # 明确留空交给模型 —— 不要为了凑分在这里假装能抽（会变成不可解释的假阳性）。
        out = []
    else:
        # 自由文本实体：从候选池里领一个最靠前的、还没被占用的
        for i, c in enumerate(entity_pool):
            if _free(c["start"], c["end"]):
                out.append({"role": role, "text": c["text"],
                            "start": c["start"], "end": c["end"]})
                entity_pool.pop(i)   # 领走，别的角色不能再用同一个
                break
    return out

def extract_one(text: str, schema: dict, trig_pairs: list[tuple[str, str]],
                company_dict: set[str]) -> dict | None:
    """对一句话抽一个事件；抽不到返回 None（评测时写成 events: []）。"""
    hit = find_trigger(text, trig_pairs)
    if hit is None:
        return None
    etype, word, ts, te = hit
    roles = schema.get(etype, [])
    used = [(ts, te)]   # 触发词占位，论元不能和它重叠
    entity_pool = extract_entities(text, company_dict)   # 整句扫一次公司候选，各角色轮流领
    args = []
    for role in roles:
        got = extract_args(text, etype, role, company_dict, used, entity_pool)
        for a in got:
            used.append((a["start"], a["end"]))
        args.extend(got)
    return {"type": etype, "trigger": {"text": word, "start": ts, "end": te},
            "args": args}

# ─────────────────────────── 4. 失败分析（教学点）───────────────────────────

def analyze_failures(gold_rows: list[dict], pred_rows: list[dict], top_n: int = 8) -> dict:
    """
    把错误拆成互斥的几类，告诉学生「规则到底输在哪」：

      no_trigger      没扫到触发词 → 整句漏抽（词典覆盖率 = 召回天花板）
      wrong_type      触发词命中了，但类型判错（「质押」vs「解除质押」）
      wrong_trigger   类型对了，触发词边界/文本不对
      arg_missing     事件对了，但论元漏抽
      arg_spurious    事件对了，但论元多抽
      exact           完全正确
    """
    pred_map = {(r.get("id"), r.get("sent_id")): r for r in pred_rows}
    cats = Counter()
    examples = defaultdict(list)
    n_trig_hit = 0

    for g in gold_rows:
        ge = row_event(g) or {}
        p = pred_map.get((g.get("id"), g.get("sent_id")), {"events": []})
        pe = (p.get("events") or [{}])[0] if p.get("events") else {}
        g_trig = (ge.get("trigger") or {}).get("text", "")
        p_trig = (pe.get("trigger") or {}).get("text", "")
        g_args = {(a["role"], a["text"]) for a in ge.get("args") or []}
        p_args = {(a["role"], a["text"]) for a in pe.get("args") or []}

        if not p_trig:
            cat = "no_trigger"
        else:
            n_trig_hit += 1
            if pe.get("type") != ge.get("type"):
                cat = "wrong_type"
            elif p_trig != g_trig:
                cat = "wrong_trigger"
            elif p_args == g_args:
                cat = "exact"
            elif g_args - p_args and not (p_args - g_args):
                cat = "arg_missing"
            else:
                cat = "arg_spurious" if p_args - g_args else "arg_missing"
        cats[cat] += 1
        if len(examples[cat]) < top_n:
            examples[cat].append({"text": g.get("text", ""),
                                  "gold": {"type": ge.get("type"), "trigger": g_trig,
                                           "args": sorted(g_args)},
                                  "pred": {"type": pe.get("type"), "trigger": p_trig,
                                           "args": sorted(p_args)}})

    total = max(len(gold_rows), 1)
    return {
        "counts": dict(cats),
        "ratios": {k: round(v / total, 4) for k, v in cats.items()},
        "trigger_hit_rate": round(n_trig_hit / total, 4),
        "examples": dict(examples),
    }

# ─────────────────────────── 主流程 ───────────────────────────

def run(data_dir: Path, name: str, min_trigger: int) -> dict:  # noqa: C901
    # 规则基线不训练，用 EvalLogger 记录"这次评测用了什么词典/正则、结果如何"
    eval_logger = EvalLogger(method=name, root=ROOT, notes="L0 规则基线（触发词词典 + 正则论元）")
    schema = load_schema(data_dir / "event_schema.json")
    train_rows = load_extract_split("train", data_dir)
    dev_rows = load_extract_split("dev", data_dir)

    auto = build_trigger_dict(train_rows, min_count=min_trigger)
    trig_pairs = merge_trigger_dict(auto, MANUAL_TRIGGERS)
    company_dict = build_company_dict(train_rows)
    print(f"触发词炼成：{len(trig_pairs)} 条（自动 {sum(len(v) for v in auto.values())} + "
          f"人工 {sum(len(v) for v in MANUAL_TRIGGERS.values())}，含去重）")
    print(f"公司名词典：{len(company_dict)} 个")

    import time
    t0 = time.perf_counter()
    preds = []
    for r in dev_rows:
        ev = extract_one(r["text"], schema, trig_pairs, company_dict)
        preds.append(make_pred_row(r, [ev] if ev else []))
    elapsed = time.perf_counter() - t0
    ms_per_sent = elapsed * 1000 / max(len(dev_rows), 1)
    print(f"推理完成：{len(dev_rows)} 句，耗时 {elapsed:.2f}s（{ms_per_sent:.3f} ms/句，纯 CPU）")

    meta = {
        "display_name": "L0 规则（词典+正则）",
        "train_s": 0,
        "params_m": 0,
        "ms_per_sentence": round(ms_per_sent, 3),
        "num_triggers": len(trig_pairs),
        "num_company_dict": len(company_dict),
        "min_trigger_count": min_trigger,
        "note": "零训练基线；触发词从 train 统计 + 人工同义词，论元用正则/词典",
    }
    res = evaluate_predictions(name, preds, meta=meta,
                              gold_path=data_dir / "dev_extract.jsonl", with_detect=False)

    failures = analyze_failures(dev_rows, preds)
    res["failure_analysis"] = failures
    METRIC_DIR.mkdir(parents=True, exist_ok=True)
    with open(METRIC_DIR / f"{name}.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    print("\n【规则基线失败分析】")
    for k, v in sorted(failures["counts"].items(), key=lambda x: -x[1]):
        print(f"  {k:14s} {v:5d} 条 ({failures['ratios'][k]:.1%})")
    print(f"  触发词命中率（整句有事件的比例）：{failures['trigger_hit_rate']:.1%}")
    eval_logger.finalize(
        metrics={k: res[k]["f1"] for k in ("trigger", "argument", "event_exact")},
        type_macro_f1=res.get("type_macro_f1"),
        num_triggers=len(trig_pairs), num_company_dict=len(company_dict),
        trigger_hit_rate=failures["trigger_hit_rate"],
        failure_counts=failures["counts"],
    )
    return res

def main():
    ap = argparse.ArgumentParser(description="L0 规则基线：触发词词典 + 正则论元")
    ap.add_argument("--mock", action="store_true", help="用 data/_mock 的 20 条手造数据自检")
    ap.add_argument("--data_dir", default="", help="数据目录；留空=全量 data/processed")
    ap.add_argument("--name", default="", help="预测/指标文件名，默认 rule")
    ap.add_argument("--min_trigger_count", type=int, default=2)
    args = ap.parse_args()

    data_dir = (Path(args.data_dir) if args.data_dir
                else (MOCK_DIR if args.mock else DATA_DIR))
    name = args.name or ("rule_mock" if args.mock else "rule")
    print(f"数据目录：{data_dir}")
    run(data_dir, name, args.min_trigger_count)

if __name__ == "__main__":
    main()
