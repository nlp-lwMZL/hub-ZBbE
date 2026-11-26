"""
量化「金标自身的非法 BIO」——P25 那张「非法序列」页的正确基线

为什么这个脚本很重要（课上必讲的一个因果纠正）：
  序列标注实验里常做「linear 头 vs CRF」的非法 BIO 对比，很容易顺手写成
  「线性头产生了 67% 的非法序列，CRF 解决了它」。这个表述**是错的**，因为它把因果归错了：
    - 金标自己是「重叠标注」：同一段文字可以同时是「收购标的」和「被收购方」，
      而 BIO 天生只能给一个字符一个标签 → 投影时必然产生少量非法（本数据 dev 18 处 / 1.9%）
    - 模型预测的非法率（linear 67%）远高于这个地板值，说明**大部分非法确实是模型的问题**
    - 但正确的表述是「非法率有个 1.9% 的不可消除地板，linear 到 67%，CRF 把它拉回多少、
      加硬约束能不能压到地板以下」
  如果不测这个地板值，PPT 上「67% 非法」就会被误读成「任务本身有问题」或者
  「模型完全在乱喷」，而真实情况是「模型乱喷 + 标注重叠的固有代价」。

同时统计 span 覆盖冲突：同一字符被两条 span 覆盖的次数（论文里叫 overlap conflict），
它直接解释了非法地板的来源。

使用方式：
  python eval/check_bio_gold.py                  # 算 train/dev 的金标非法量与重叠冲突
  python eval/check_bio_gold.py --split dev      # 只算 dev

产物：outputs/metrics/disc/bio_gold_baseline.json
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # Windows: torch 与 numpy 各带 libiomp5md，不设会 OMP Error #15

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]   # 项目根目录（方法目录的上一级）
sys.path.insert(0, str(ROOT / "common"))     # 公共模块：event_data / crf / eval_utils / models / log_utils

from event_data import (DATA_DIR, build_joint_schema, illegal_bio_stats, load_schema,

                        row_event, span_to_char_tags)

def gold_illegal(rows: list[dict], scheme: dict) -> tuple[dict, Counter]:
    """
    把金标 span 投影成 BIO，再统计 IL 法量。

    返回 (illegal_stats, conflict_counter)：
      - illegal_stats 用和模型预测完全相同的 illegal_bio_stats()，口径一致才可比
      - conflict_counter 记录「同一字符被两条 span 覆盖」的次数（论元重叠的直接证据）
    """
    conflicts = Counter()
    tag_seqs = []
    id2label = scheme["id2label"]
    for r in rows:
        ev = row_event(r)
        if not ev:
            tag_seqs.append([])
            continue
        ids = span_to_char_tags(r["text"], ev, scheme, conflict_counter=conflicts)
        tag_seqs.append([id2label[i] for i in ids])
    return illegal_bio_stats(tag_seqs), conflicts

def main():
    ap = argparse.ArgumentParser(description="统计金标自身的非法 BIO 与 span 重叠冲突")
    ap.add_argument("--split", nargs="*", default=["train", "dev"])
    ap.add_argument("--data_dir", default="")
    args = ap.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else DATA_DIR
    scheme = build_joint_schema(load_schema(data_dir / "event_schema.json"))
    out = {"label_scheme": f"joint({len(scheme['labels'])} 标签)", "splits": {}}

    for split in args.split:
        rows = [json.loads(l) for l in
                open(data_dir / f"{split}_extract.jsonl", encoding="utf-8") if l.strip()]
        stats, conflicts = gold_illegal(rows, scheme)
        out["splits"][split] = {"n": len(rows), "illegal": stats,
                                "overlap_conflicts": dict(conflicts)}
        print(f"[{split}] {len(rows)} 句：金标非法转移 {stats['illegal_transition']} 处 / "
              f"{stats['illegal_seqs']} 句（{stats['illegal_seq_rate']:.1%}），"
              f"非法开头 {stats['illegal_start']} 处")
        print(f"         span 覆盖冲突：论元 {conflicts.get('arg', 0)} 处 / "
              f"触发词 {conflicts.get('trigger', 0)} 处"
              f"（论元可重叠，如「收购标的」包含「被收购方」→ BIO 表达不了，只能投影）")

    out["note"] = ("金标自身因「重叠论元投影到 BIO」就有少量非法，这是不可消除的地板值；"
                   "模型预测的非法率应和这个地板值比，而不是和 0 比。")
    dest = ROOT / "outputs" / "metrics" / "disc" / "bio_gold_baseline.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n已写入 → {dest.relative_to(ROOT)}")

if __name__ == "__main__":
    main()
