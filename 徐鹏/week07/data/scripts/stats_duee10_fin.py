# -*- coding: utf-8 -*-
"""统计 DuEE 1.0 的金融相关子集规模，并核对“同一字符串多次出现”时标注选的是第几次。

为什么单独看这个：
  DuEE 1.0 的 index 是人工标注，DuEE-Fin 句子级镜像的 index 是用 text.find() 回填的。
  如果 DuEE 1.0 里存在“选了非首次出现位置”的 span，就能证明两者性质不同（一个是标注，一个是启发式回填），
  这也是报告里“标注质量”一栏的核心证据。
"""
import glob
import json
from collections import Counter
from pathlib import Path

RAW = Path(__file__).resolve().parent.parent / "raw" / "DuEE_hf"


def iter_rows():
    import pandas as pd

    for split in ["train", "validation"]:
        f = glob.glob(str(RAW / (split + "-*.parquet")))[0]
        df = pd.read_parquet(f)
        for _, r in df.iterrows():
            yield split, r["text"], r["events"]


FIN_GROUPS = {
    "财经/交易": None,  # 前缀匹配
    "组织关系": {"组织关系-裁员", "组织关系-解雇", "组织关系-辞/离职", "组织关系-停职", "组织关系-解约", "组织关系-退出"},
    "司法行为": {"司法行为-起诉", "司法行为-立案", "司法行为-开庭", "司法行为-入狱", "司法行为-罚款", "司法行为-约谈", "司法行为-举报"},
    "产品行为": {"产品行为-发布", "产品行为-召回", "产品行为-下架"},
}

if __name__ == "__main__":
    per_group = Counter()
    per_type = Counter()
    sent_per_group = Counter()
    ambig = ambig_nonfirst = 0
    spans = 0
    lens = []
    events_per_sent = Counter()
    roles_per_event = []
    for split, text, events in iter_rows():
        lens.append(len(text))
        events_per_sent[len(events)] += 1
        groups_hit = set()
        for ev in events:
            lab = ev["label"]
            per_type[lab] += 1
            g = label = None
            for gname, members in FIN_GROUPS.items():
                if gname == "财经/交易":
                    if lab.startswith("财经/交易-"):
                        g = gname
                elif lab in members:
                    g = gname
            if g:
                per_group[g] += 1
                groups_hit.add(g)
            roles_per_event.append(len(ev["args"]))
            for sp in [ev["trigger"]] + list(ev["args"]):
                spans += 1
                idx = list(sp["indices"])
                s = sp["text"]
                occ = [i for i in range(len(text)) if text.startswith(s, i)]
                if len(occ) > 1:
                    ambig += 1
                    if idx and idx[0] != occ[0]:
                        ambig_nonfirst += 1
        for g in groups_hit:
            sent_per_group[g] += 1

    fin_types = {k: v for k, v in per_type.items() if k.startswith("财经/交易-")}
    # 句子级“含至少 1 个金融相关事件”的句子数（用 (split,text) 去重不必要，数据本身不重复）
    covered_sents = 0
    for split, text, events in iter_rows():
        if any(
            ev["label"].startswith("财经/交易-")
            or ev["label"].startswith("组织关系-")
            or ev["label"].startswith("司法行为-")
            or ev["label"].startswith("产品行为-")
            for ev in events
        ):
            covered_sents += 1

    print("=== DuEE 1.0 金融相关子集 ===")
    for g, c in per_group.items():
        print("%-6s events=%d" % (g, c))
    print("财经/交易 9 类明细:", json.dumps(fin_types, ensure_ascii=False))
    print("严格金融（仅 财经/交易-*）事件数:", sum(fin_types.values()))
    print("含对公/企业类事件的句子数（财经+组织关系+司法行为+产品行为）:", covered_sents)
    print("\n=== 标注细节 ===")
    print("span 总数:", spans, "| 字符串在句中出现多次:", ambig, "| 其中标注选了非首次出现:", ambig_nonfirst,
          "(%.4f%%)" % (100.0 * ambig_nonfirst / max(1, ambig)))
    print("每句事件数分布:", dict(sorted(events_per_sent.items())))
    s = sorted(lens)
    print("句长: p50=%d p90=%d p99=%d max=%d" % (s[len(s) // 2], s[int(len(s) * 0.9)], s[int(len(s) * 0.99)], s[-1]))
