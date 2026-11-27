# -*- coding: utf-8 -*-
"""
BIO 标签 ↔ start/end 实体列表：本作业的「数据格式处理」核心。

为什么这文件是作业的重点（对应课上同学的提问）：
  序列标注的数据集**不是只有一种格式**。课上 DuEE 金融事件用的是
  「原文 + start/end 偏移」——一个 span 用两个整数表示，天然能表达重叠/嵌套；
  而大量 NER 数据集（人民日报、MSRA、CoNLL）用的是 **BIO 逐 token 标签**——
  每个字/词只背一个标签，B/I/O 三个字母把「实体边界」编码进标签序列里。
  两者可以互相转换，但**不是无损的**：BIO 一个位置只能有一个标签，
  所以重叠实体转过去必然丢信息。这个文件把转换和损失都显式写出来。

偏移约定（最容易踩的坑，先钉死）：
  - 本文件所有 `start` / `end` 都是 **Python 切片语义：左闭右开 [start, end)**。
  - 例：`"厦门"` 在 `"海钓比赛地点在厦门..."` 中 start=7, end=9。
  - 原始 BIO 标签本身没有偏移，start/end 是**由标签序列推出来的**；
    如果别的数据集给你的是闭区间 end（含最后一个字），**必须自己 +1**，
    否则每个实体都会少最后一个字，而且评估时不一定报错，只会静默掉点。

使用方式：
  from bio_labels import bio_to_spans, spans_to_bio, check_bio
"""

from collections import Counter
from typing import Iterable


def load_label_names(data_dir) -> list[str]:
    """
    读取 label_names.json，返回标签顺序列表。

    为什么标签顺序要单独存：
      模型输出的是 0..N-1 的 id，id→标签的映射一旦错位，
      指标会离奇地低，而且代码不会报错。所以训练/评估必须读**同一份** label_names.json。
    """
    import json
    from pathlib import Path

    path = Path(data_dir) / "label_names.json"
    with open(path, "r", encoding="utf-8") as f:
        labels = json.load(f)

    if not isinstance(labels, list) or not labels:
        raise ValueError(f"label_names.json 应该是一个非空列表：{path}")
    if labels[0] != "O":
        raise ValueError(f"BIO 标签列表第 0 个应该是 O，实际是 {labels[0]!r}")
    if len(set(labels)) != len(labels):
        raise ValueError(f"label_names.json 有重复标签：{labels}")
    return labels


def build_label_maps(label_names: list[str]) -> tuple[dict[str, int], dict[int, str]]:
    """标签名 ↔ id 的双向映射。顺序完全按 label_names.json，不排序。"""
    label2id = {name: i for i, name in enumerate(label_names)}
    id2label = {i: name for name, i in label2id.items()}
    return label2id, id2label


def split_bio_tag(tag: str) -> tuple[str, str | None]:
    """
    把 'B-LOC' 拆成 ('B', 'LOC')；'O' 拆成 ('O', None)。

    这里显式校验前缀，是因为不同数据集的标签体系并不统一：
      - BIO：B-X / I-X / O
      - BIOES（也叫 BMES）：B-X / I-X / E-X / S-X / O
    本数据集只有 BIO。如果以后换成 BIOES，这里的解析和下面的合并逻辑都要改，
    不能假设「前缀只有 B/I/O」。
    """
    if tag == "O":
        return "O", None
    if "-" not in tag:
        raise ValueError(f"无法解析的 BIO 标签：{tag!r}（应为 O 或 B-X / I-X）")
    prefix, etype = tag.split("-", 1)
    if prefix not in {"B", "I"}:
        raise ValueError(f"本数据集只支持 BIO，遇到前缀 {prefix!r}（标签 {tag!r}）")
    if not etype:
        raise ValueError(f"标签 {tag!r} 缺少实体类型")
    return prefix, etype


def check_bio(tags: Iterable[str]) -> dict:
    """
    统计一条（或一批）BIO 序列里有多少「非法转移」。

    什么叫非法：
      - 序列以 I-X 开头（前面没有 B-X）
      - O 后面直接跟 I-X
      - B-X / I-X 后面跟 I-Y（X != Y，实体类型中途变了）

    为什么要统计：
      线性分类头是**逐 token 独立**预测的，它完全可能输出 I-LOC 开头这种序列；
      CRF 会减少但（用 pytorch-crf 时）不保证归零。这个数字是解释
      「线性头 vs CRF」最直观的证据，也是检查数据/解码是否写错的第一道体检。
    """
    tags = list(tags)
    stats = {
        "n_tokens": len(tags),
        "illegal_start": 0,
        "illegal_transition": 0,
        "n_sequences_with_illegal": 0,
    }
    seq_has_illegal = False

    prev_prefix, prev_type = None, None
    for i, tag in enumerate(tags):
        prefix, etype = split_bio_tag(tag)

        if prefix == "I":
            if i == 0:
                stats["illegal_start"] += 1
                seq_has_illegal = True
            elif prev_prefix == "O":
                stats["illegal_transition"] += 1
                seq_has_illegal = True
            elif prev_type != etype:
                stats["illegal_transition"] += 1
                seq_has_illegal = True

        prev_prefix, prev_type = prefix, etype

    if seq_has_illegal:
        stats["n_sequences_with_illegal"] = 1
    return stats


def bio_to_spans(tokens: list[str], tags: list[str]) -> list[dict]:
    """
    BIO 标签序列 → start/end 实体列表（左闭右开）。

    这是把「BIO 数据集」喂给「按 span 评估/展示」的通用入口：
      tokens: ['海','钓',...,'厦','门',...]
      tags:   ['O','O',...,'B-LOC','I-LOC',...]
      → [{'type':'LOC','start':7,'end':9,'text':'厦门'}, ...]

    对非法序列的处理（重要）：
      真实模型输出可能以 I-X 开头，或者 B-X 后面接 I-Y。这里采用和 seqeval
      一致的「宽容修复」策略——把孤立的 I-X 当成新实体的开头，
      让评估仍能继续进行；非法本身由 check_bio() 单独统计。
      **不要**因为「能修」就忽略它：修出来的 span 往往就是错误来源。
    """
    if len(tokens) != len(tags):
        raise ValueError(f"tokens 与 tags 长度不一致：{len(tokens)} vs {len(tags)}")

    spans: list[dict] = []
    cur_type: str | None = None
    cur_start: int | None = None

    def close(end: int):
        nonlocal cur_type, cur_start
        if cur_type is not None and cur_start is not None:
            spans.append({
                "type": cur_type,
                "start": cur_start,
                "end": end,
                "text": "".join(tokens[cur_start:end]),
            })
        cur_type, cur_start = None, None

    for i, tag in enumerate(tags):
        prefix, etype = split_bio_tag(tag)

        if prefix == "O":
            close(i)
        elif prefix == "B":
            close(i)                 # 上一个实体在这里结束
            cur_type, cur_start = etype, i
        else:  # prefix == "I"
            if cur_type == etype:
                continue             # 正常的实体内部
            # 非法：I-X 开头 / O→I-X / B-X→I-Y。按 seqeval 的口径修复成新实体
            close(i)
            cur_type, cur_start = etype, i

    close(len(tags))
    return spans


def spans_to_bio(tokens: list[str], spans: list[dict], label_names: list[str]) -> tuple[list[str], list[dict]]:
    """
    start/end 实体列表 → BIO 标签序列。

    这是学生换数据集时最常要写的函数：很多公开数据集给的是
    `{"text": ..., "entities": [{"start":..., "end":..., "type":...}]}`，
    要转成 BIO 才能喂给逐 token 分类模型。

    返回值：
      tags:      长度 == len(tokens) 的 BIO 标签列表
      conflicts: 重叠/嵌套导致的冲突列表（空列表 = 无损转换）

    为什么会有 conflicts：
      BIO 的硬限制是「一个 token 只能有一个标签」。如果两个实体在同一个位置重叠，
      后写的会被丢掉。这不是实现 bug，是格式本身的表达力上限——
      想保留重叠/嵌套，只能用 start/end（或 pointer network 之类）的方案。
    """
    tags = ["O"] * len(tokens)
    valid_types = {name[2:] for name in label_names if name != "O"}
    conflicts: list[dict] = []

    for span in spans:
        etype = span["type"]
        start, end = int(span["start"]), int(span["end"])

        if etype not in valid_types:
            raise ValueError(f"实体类型 {etype!r} 不在标签体系里：{sorted(valid_types)}")
        if not (0 <= start < end <= len(tokens)):
            raise ValueError(
                f"非法 span [{start}, {end})：tokens 长度 {len(tokens)}（end 是左闭右开）"
            )

        for i in range(start, end):
            new_tag = f"B-{etype}" if i == start else f"I-{etype}"
            if tags[i] != "O":
                # 已经有实体占了这位：保留先写入的，记录冲突（后写者被丢弃）
                conflicts.append({
                    "position": i,
                    "kept": tags[i],
                    "dropped": new_tag,
                    "span": [start, end],
                    "type": etype,
                })
                continue
            tags[i] = new_tag

    return tags, conflicts


def spans_to_class_style(text: str, spans: list[dict]) -> dict:
    """
    把 span 列表包装成「课上那种 start/end JSON」的样子，方便并排对照。

    课上的事件抽取是：
      {"text": "...", "event": {"trigger": {"start":..,"end":..}, "args": [...]}}
    本质就是「原文 + 若干 start/end」。这里用更中性的 entities 字段演示。
    """
    return {
        "text": text,
        "entities": [
            {
                "type": s["type"],
                "start": s["start"],
                "end": s["end"],
                "text": text[s["start"]:s["end"]],
            }
            for s in spans
        ],
    }


def entity_type_counts(spans: list[dict]) -> Counter:
    """按实体类型计数，用于数据探查与分类别指标。"""
    return Counter(s["type"] for s in spans)


def spans_exact_equal(a: list[dict], b: list[dict]) -> bool:
    """两个 span 列表是否完全一致（类型 + 起止都相同），用于句子级 exact match。"""
    key = lambda s: (s["type"], s["start"], s["end"])
    return sorted(map(key, a)) == sorted(map(key, b))


def error_breakdown(gold_spans: list[dict], pred_spans: list[dict]) -> dict:
    """
    把句子级错误分成四类，方便 bad case 分析：
      - miss:     金标有、预测完全没有（漏）
      - spurious: 预测有、金标没有（多）
      - type:     边界完全相同但类型错了（如 ORG 判成 LOC）
      - boundary: 类型对、边界错（如「厦门大学」只标出「厦门」）

    判定顺序：先找完全匹配；剩下的金标实体里，优先看「同边界不同类型」，
    再看「同类型有重叠」，最后才算 miss。这样分类更符合人的直觉。
    """
    def key(s):
        return (s["type"], s["start"], s["end"])

    gold_left = {key(s): s for s in gold_spans}
    pred_left = {key(s): s for s in pred_spans}

    # 完全匹配的先消掉
    for k in list(gold_left):
        if k in pred_left:
            del gold_left[k]
            del pred_left[k]

    stats = {"miss": 0, "spurious": 0, "type": 0, "boundary": 0}
    examples = []

    for k, g in list(gold_left.items()):
        same_boundary = [p for p in pred_left.values()
                         if p["start"] == g["start"] and p["end"] == g["end"]]
        if same_boundary:
            p = same_boundary[0]
            stats["type"] += 1
            examples.append({"kind": "type", "gold": g, "pred": p})
            del pred_left[key(p)]
            del gold_left[k]
            continue

        same_type_overlap = [p for p in pred_left.values()
                             if p["type"] == g["type"]
                             and p["start"] < g["end"] and g["start"] < p["end"]]
        if same_type_overlap:
            p = same_type_overlap[0]
            stats["boundary"] += 1
            examples.append({"kind": "boundary", "gold": g, "pred": p})
            del pred_left[key(p)]
            del gold_left[k]
            continue

        stats["miss"] += 1
        examples.append({"kind": "miss", "gold": g, "pred": None})
        del gold_left[k]

    for p in pred_left.values():
        stats["spurious"] += 1
        examples.append({"kind": "spurious", "gold": None, "pred": p})

    return {"counts": stats, "examples": examples}


def format_bio_table(tokens: list[str], tags: list[str], max_rows: int = 60) -> str:
    """
    把「字符 / 标签」排成两行表格，方便在终端里直观看 BIO。

    超过 max_rows 就截断并标注，避免一条长文本把屏幕刷满。
    """
    shown = min(len(tokens), max_rows)
    chars = " ".join(tokens[:shown])
    labels = " ".join(tags[:shown])
    suffix = f"\n...（共 {len(tokens)} 个 token，只显示前 {shown} 个）" if len(tokens) > shown else ""
    return f"字符：{chars}\n标签：{labels}{suffix}"
