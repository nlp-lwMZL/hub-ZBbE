"""
方法一：规则 / 关键词（L0 基线）

教学重点：
  1. 这是**最便宜的**方法：不训练、不加载模型，10 行代码。
     先跑它，是为了给后面所有方法一个「下限」——连它都没有的东西不配谈提升。
  2. 它的天花板由「命中率」决定：实测只有 6% 的标题里出现了类别名，
     所以 Macro-F1 只有 0.069（15 类随机基线是 0.067）。
     这不是代码写得差，是这类任务天生不适合规则。
  3. 规则并非一无是处：在**格式固定的公告标题**上（"XX公司：2023 年年度业绩预告"），
     一条正则就能覆盖大部分，生产里常用它做「快筛」挡掉模型调用。

用法：
  python rule/classify_rule.py            # 全量干净验证集
  python rule/classify_rule.py --n 500    # 抽样 500 条
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.data import load_clean_val, load_splits
from common.labels import LABELS
from common.metrics import classification_report, pred_distribution, to_markdown_table


def classify(text: str, fallback: int) -> int:
    """
    最简单的规则：标题里出现哪个类别名，就判成哪一类。

    为什么按 LABELS 的顺序取第一个命中：
      这里故意不做打分/优先级——规则法就是要暴露「它有多脆弱」，
      加上优先级表就变成人工调参了，反而掩盖 L0 的真实上限。
    """
    for i, name in enumerate(LABELS):
        if name in text:
            return i
    return fallback          # 一个类别名都没有 → 退化成「猜最多的那一类」


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0, help="抽样条数，0=用全部干净样本")
    ap.add_argument("--out", default="outputs/rule_pred.json")
    args = ap.parse_args()

    samples, meta = load_clean_val(args.n or None)
    train, _ = load_splits()

    # 兜底类：训练集里样本最多的类别（这叫 majority baseline，是所有方法的下限）
    from collections import Counter
    fallback = Counter(d["label"] for d in train).most_common(1)[0][0]

    y_true = [d["label"] for d in samples]
    y_pred = [classify(d["sentence"], fallback) for d in samples]

    report = classification_report(y_true, y_pred)
    report["method"] = "rule_keyword"
    report["pred_distribution"] = pred_distribution(y_pred)

    out = Path(__file__).resolve().parent.parent / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"meta": meta, "report": report}, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"样本 {meta['n_eval']} 条（干净集，重复已剔除 {meta['n_dup']} 条）")
    print(f"命中率 {sum(1 for d in samples if any(n in d['sentence'] for n in LABELS)) / len(samples):.2%}")
    print(to_markdown_table({"规则/关键词": report}))
    print(f"→ 结果写入 {out}")


if __name__ == "__main__":
    main()
