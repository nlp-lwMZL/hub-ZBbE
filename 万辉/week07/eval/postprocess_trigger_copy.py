"""
后处理规则：把「触发词本身也是某个论元」的角色回填（纯 CPU，不需要重训、不需要 GPU）

发现的真实问题（P27 的一个好素材）：
  本数据里 1.1% 的论元**与触发词区间重叠**，而且高度集中：
    「高管变动.变动类型」274 条 —— 因为触发词「辞职/辞去/聘任」本身就是"变动类型"这个论元！
    「公司上市.环节」18 条 —— 「正式上市」既是触发词也是"环节"
  BIO 只能给一个字符一个标签，而我们的投影规则让**触发词优先**（合理：没触发词就没事件），
  于是这些论元在金标的 BIO 里**根本不存在** → 序列标注模型无论多强都学不会，
  实测「变动类型」的 dev F1 恰好是 **0.0000**（支持数 42，一条都对不上）。

解法（标准做法，叫「后处理规则」/ post-hoc rule，不涉及重训）：
  从**训练集**自动统计：对每个 (事件类型, 角色)，它和触发词重叠的比例有多高。
  比例 ≥ 阈值的，就学成一条规则「该角色 = 触发词文本」，推理时如果没有预测出这个角色，
  就用触发词文本回填。规则全部来自训练集统计，不看 dev，也不手写类型知识。

  为什么这类"补规则"在工业界很常见：模型擅长判断"有没有事件、触发词在哪"，
  而 schema 里的固定关系（某角色的取值恒等于触发词）用规则表达**更准确、可解释、零成本**。
  这是「模型 + 规则」混合系统的典型形态，也是 week7 讲"落地"时的核心页。

使用方式：
  python eval/postprocess_trigger_copy.py                       # 对 BERT+CRF 预测做后处理
  python eval/postprocess_trigger_copy.py --base bert_linear_256 # 对 Linear 预测做后处理
  python eval/postprocess_trigger_copy.py --threshold 0.5

产物：outputs/disc/pred_<base>_rule.jsonl、outputs/metrics/disc/<base>_rule.json
（原预测文件不动，方便对比「规则前后」）
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # Windows: torch 与 numpy 各带 libiomp5md，不设会 OMP Error #15

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]   # 项目根目录（方法目录的上一级）
sys.path.insert(0, str(ROOT / "common"))     # 公共模块：event_data / crf / eval_utils / models / log_utils

from eval_utils import METRIC_DIR, evaluate_predictions
from event_data import DATA_DIR, load_extract_split, row_event

def learn_copy_rules(train_rows: list[dict], threshold: float = 0.5) -> tuple[dict, dict]:
    """
    从训练集学「该角色 = 触发词文本」的规则。

    返回 (rules, stats)：
      rules = {事件类型: {角色: 命中率}}
      stats = {(类型, 角色): (重叠数, 总数)}  —— 打印用，便于课上展示依据
    """
    total = Counter()
    overlap = Counter()
    for r in train_rows:
        ev = row_event(r)
        if not ev:
            continue
        t = ev["type"]
        ts = (ev.get("trigger") or {}).get("start")
        te = (ev.get("trigger") or {}).get("end")
        for a in ev.get("args") or []:
            key = (t, a["role"])
            total[key] += 1
            # 判据用「**文本完全相等**」而不是「区间重叠」——
            # 实测区间重叠的规则回填后论元 F1 反而 -0.0015：因为触发词和论元常常同义不同形
            # （触发词「辞去」vs 论元「辞职」），回填错的文本只会增加假阳性。
            # 只学「这段文字确实一字不差地兼任两个角色」的情况才安全。
            if a.get("text") and a["text"] == (ev.get("trigger") or {}).get("text", ""):
                overlap[key] += 1

    rules: dict[str, dict] = defaultdict(dict)
    stats = {}
    for key, n in total.items():
        if n < 3:      # 样本太少的规则不可信（3 条以下不学）
            continue
        ratio = overlap[key] / n
        stats[key] = (overlap[key], n)
        if ratio >= threshold:
            rules[key[0]][key[1]] = round(ratio, 3)
    return dict(rules), stats

def apply_rules(pred_rows: list[dict], rules: dict) -> tuple[list[dict], int]:
    """对预测结果回填：该角色没抽到就用触发词文本补一条。返回 (新预测, 补充条数)。"""
    out, n_added = [], 0
    for p in pred_rows:
        events = []
        for ev in p.get("events") or []:
            ev = json.loads(json.dumps(ev))   # 深拷贝，别改原文件的对象
            role_rules = rules.get(ev.get("type", ""), {})
            have = {a.get("role") for a in ev.get("args") or []}
            trig = (ev.get("trigger") or {}).get("text", "")
            for role in role_rules:
                if role in have or not trig:
                    continue
                ev.setdefault("args", []).append(
                    {"role": role, "text": trig, "source": "trigger_copy"})
                n_added += 1
            events.append(ev)
        out.append({**p, "events": events})
    return out, n_added

def main():
    ap = argparse.ArgumentParser(description="触发词回填后处理规则")
    ap.add_argument("--base", default="bert_crf_256", help="基础方法名（读 pred_<base>.jsonl）")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--data_dir", default="")
    args = ap.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else DATA_DIR
    pred_path = ROOT / "outputs" / "disc" / f"pred_{args.base}.jsonl"
    if not pred_path.exists():
        print(f"找不到预测文件 {pred_path}（先跑对应训练脚本）")
        return
    pred_rows = [json.loads(l) for l in open(pred_path, encoding="utf-8") if l.strip()]

    train_rows = load_extract_split("train", data_dir)
    rules, stats = learn_copy_rules(train_rows, args.threshold)

    print(f"从训练集学到 {sum(len(v) for v in rules.values())} 条「角色=触发词」规则"
          f"（阈值 重叠率≥{args.threshold:.0%}，样本数≥3）：")
    for t, roles in rules.items():
        for role, ratio in roles.items():
            ov, n = stats[(t, role)]
            print(f"  {t}.{role:<12} 重叠 {ov}/{n} = {ratio:.0%} → 回填触发词")

    new_rows, n_added = apply_rules(pred_rows, rules)
    print(f"\n回填了 {n_added} 条论元（覆盖 {len(pred_rows)} 条预测）")

    meta = {
        "display_name": f"{args.base} + 触发词回填规则",
        "train_s": 0,
        "ms_per_sentence": 0.0,
        "base_method": args.base,
        "rule_threshold": args.threshold,
        "n_rules": sum(len(v) for v in rules.values()),
        "n_args_filled": n_added,
        "rules": rules,
        "note": ("后处理规则：训练集里某角色与触发词重叠率≥阈值时，"
                 "推理时若未抽出该角色就用触发词文本回填；不重训模型"),
    }
    res = evaluate_predictions(f"{args.base}_rule", new_rows, meta=meta,
                              gold_path=data_dir / "dev_extract.jsonl")

    # 前后对照（读基础方法的 JSON；读不到就只打印绝对数字）
    base_path = METRIC_DIR / f"{args.base}.json"
    if base_path.exists():
        base = json.loads(base_path.read_text(encoding="utf-8"))
        print("\n【规则前 → 规则后】")
        for k, label in (("trigger", "触发词"), ("argument", "论元"), ("event_exact", "事件级")):
            b, a = base[k]["f1"], res[k]["f1"]
            print(f"  {label:<5} F1 {b:.4f} → {a:.4f}  ({a - b:+.4f})")

if __name__ == "__main__":
    main()
