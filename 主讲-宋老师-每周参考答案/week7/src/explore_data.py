# -*- coding: utf-8 -*-
"""
数据探查：先看清楚「BIO 数据集长什么样」，再动手训练。

本脚本回答三个问题（也是作业要学生写进报告的部分）：
  1. 这份数据集有多少条、多长、标签/实体怎么分布？有没有脏数据？
  2. 同一条样本，用 BIO 表示和用课上那种 start/end 表示，分别长什么样？
  3. 两种格式互相转换时，哪里会丢信息？（重叠实体）

运行：
  python src/explore_data.py --bert_path <bert-base-chinese 目录>
  python src/explore_data.py --skip_tokenizer      # 没装 transformers 也能看纯数据统计
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

from bio_labels import (
    bio_to_spans,
    build_label_maps,
    check_bio,
    entity_type_counts,
    format_bio_table,
    load_label_names,
    spans_to_bio,
    spans_to_class_style,
)
from paths import DATA_DIR, LOG_DIR, ensure_output_dirs, resolve_bert_path, resolve_data_dir

SPLITS = ["train", "validation", "test"]


def percentile(sorted_values: list[int], q: float) -> int:
    """简单分位数：不依赖 numpy，取最近秩即可（教学报告够用）。"""
    if not sorted_values:
        return 0
    idx = min(len(sorted_values) - 1, int(len(sorted_values) * q))
    return sorted_values[idx]


def basic_stats(records: list[dict], label_names: list[str]) -> dict:
    """纯数据层统计：句长、标签分布、实体分布、非法 BIO、实体密度。"""
    tag_counter = Counter()
    entity_counter = Counter()
    char_lens = []
    entities_per_sent = []
    illegal_sequences = 0
    total_illegal_transitions = 0

    for row in records:
        tags = row["ner_tags"]
        char_lens.append(len(tags))
        tag_counter.update(tags)

        # 逐条检查非法 BIO；金标理论上应该全是 0，不为 0 说明数据本身有问题
        bad = check_bio(tags)
        if bad["n_sequences_with_illegal"]:
            illegal_sequences += 1
            total_illegal_transitions += bad["illegal_start"] + bad["illegal_transition"]

        spans = bio_to_spans(row["tokens"], tags)
        entity_counter.update(s["type"] for s in spans)
        entities_per_sent.append(len(spans))

    char_lens.sort()
    total_tags = sum(tag_counter.values()) or 1

    return {
        "n_sentences": len(records),
        "n_chars": sum(char_lens),
        "char_len": {
            "mean": round(statistics.mean(char_lens), 2),
            "p50": percentile(char_lens, 0.50),
            "p90": percentile(char_lens, 0.90),
            "p95": percentile(char_lens, 0.95),
            "p99": percentile(char_lens, 0.99),
            "max": max(char_lens),
        },
        "tag_counts": dict(tag_counter),
        "o_ratio": round(tag_counter.get("O", 0) / total_tags, 4),
        "entity_counts": dict(entity_counter),
        "n_entities": sum(entity_counter.values()),
        "entities_per_sentence": {
            "mean": round(statistics.mean(entities_per_sent), 3),
            "max": max(entities_per_sent),
            "zero": sum(1 for x in entities_per_sent if x == 0),
        },
        "illegal_sequences": illegal_sequences,
        "illegal_transitions": total_illegal_transitions,
    }


def tokenizer_stats(records: list[dict], tokenizer, max_length: int, batch_size: int = 1000) -> dict:
    """
    用真正的 BERT tokenizer 统计：
      - 完整 token 长度分布（含 [CLS]/[SEP]）
      - max_length 截断的句子数、被截掉的字符数
      - 被截断的实体数（完全掉尾 / 跨边界）
      - 一个字符被切成多个子词的次数（中文通常为 0，但换数据集就可能不是 0）

    为什么不在训练日志里现算：
      这一步纯 CPU、只跑一次，提前算好写进 data_report.json，
      报告里就能解释「为什么 max_length=128 会掉 recall」，而不用靠猜。
    """
    full_lens = []
    n_truncated = 0
    dropped_chars = 0
    dropped_entities = 0
    partial_entities = 0
    multi_subword_chars = 0

    # 分批 tokenize：比一条一条快一个数量级；truncation=False 拿完整长度
    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        enc = tokenizer(
            [r["tokens"] for r in batch],
            is_split_into_words=True,
            truncation=False,
            padding=False,
        )
        for i, row in enumerate(batch):
            ids = enc["input_ids"][i]
            wids = enc.word_ids(i)
            full_lens.append(len(ids))

            kept = len(row["tokens"])
            if len(ids) > max_length:
                n_truncated += 1
                # 模拟 truncation=True：保留 [CLS] + 前 max_length-2 个内容 token + [SEP]
                kept = max(
                    (w for w in wids[1:max_length - 1] if w is not None),
                    default=-1,
                ) + 1
                dropped_chars += len(row["tokens"]) - kept

            # 统计一个字符被切成多个子词的次数
            per_char = Counter(w for w in wids if w is not None)
            multi_subword_chars += sum(1 for v in per_char.values() if v > 1)

            for span in bio_to_spans(row["tokens"], row["ner_tags"]):
                if span["start"] >= kept:
                    dropped_entities += 1          # 整个实体都在被截掉的尾巴里
                elif span["end"] > kept:
                    partial_entities += 1          # 实体跨过截断边界，只保留前半截

    full_lens.sort()
    return {
        "token_len": {
            "mean": round(statistics.mean(full_lens), 2),
            "p50": percentile(full_lens, 0.50),
            "p90": percentile(full_lens, 0.90),
            "p95": percentile(full_lens, 0.95),
            "p99": percentile(full_lens, 0.99),
            "max": max(full_lens),
        },
        "max_length": max_length,
        "n_truncated": n_truncated,
        "truncated_ratio": round(n_truncated / max(len(records), 1), 4),
        "dropped_chars": dropped_chars,
        "dropped_entities": dropped_entities,
        "partial_entities": partial_entities,
        "multi_subword_chars": multi_subword_chars,
    }


def duplicate_overlap(splits: dict[str, list[dict]]) -> dict:
    """
    检查 train 与 validation/test 是否有重复句子。

    为什么必须查：如果验证集混进训练集，F1 会虚高，而且这种错在代码里完全看不出来。
    """
    def norm(rec): return "".join(rec["tokens"])

    train = {norm(r) for r in splits["train"]}
    out = {}
    for split in ("validation", "test"):
        other = {norm(r) for r in splits[split]}
        out[f"train_vs_{split}"] = len(train & other)
        out[f"{split}_size"] = len(other)
    return out


def show_one_sample(records: list[dict], index: int, label_names: list[str]) -> dict:
    """打印一条样本的三种表示，并做 BIO→span→BIO 回环校验。"""
    row = records[index]
    tokens, tags = row["tokens"], row["ner_tags"]
    text = "".join(tokens)
    spans = bio_to_spans(tokens, tags)
    class_style = spans_to_class_style(text, spans)
    back_tags, conflicts = spans_to_bio(tokens, spans, label_names=label_names)

    print("\n" + "=" * 78)
    print(f"样本 #{index}：BIO 原始 JSON")
    print("=" * 78)
    print(json.dumps(row, ensure_ascii=False))
    print("\nBIO 逐字对照：")
    print(format_bio_table(tokens, tags, max_rows=120))
    print("\n转成 start/end（左闭右开）后：")
    print(json.dumps(class_style, ensure_ascii=False, indent=2))
    print("\n再转回 BIO 是否与原始标签完全一致：", back_tags == tags, "（冲突数：", len(conflicts), "）")

    # 逐字符核对 text[start:end] 是否真等于实体文本 —— 偏移差一位时这里会立刻暴露
    print("\n偏移体检（text[start:end] 必须等于 span 文本）：")
    for s in spans:
        ok = text[s["start"]:s["end"]] == s["text"]
        print(f"  [{s['start']:>3}, {s['end']:>3}) {s['type']:<4} {s['text']!r}  {'OK' if ok else '!! 偏移错误'}")
        if not ok:
            raise AssertionError("span 偏移与文本不一致")
    return {"text": text, "spans": spans, "class_style": class_style}


def show_overlap_limitation():
    """
    演示 BIO 的表达力上限：重叠实体无法无损写进一条标签序列。

    这是回答「序列标注是不是都是这个格式」的关键：
      - start/end 可以同时保存「北京大学」是 ORG、「北京」是 LOC（重叠/嵌套）；
      - BIO 每个位置只能有一个标签，转换时必然丢掉一个。
    """
    print("\n" + "=" * 78)
    print("格式边界：重叠/嵌套实体在 BIO 里表达不了")
    print("=" * 78)
    text = "北京大学"
    tokens = list(text)
    spans = [
        {"type": "ORG", "start": 0, "end": 4, "text": "北京大学"},
        {"type": "LOC", "start": 0, "end": 2, "text": "北京"},
    ]
    tags, conflicts = spans_to_bio(tokens, spans, label_names=["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"])
    print("原文：", text)
    print("span 列表：", json.dumps(spans, ensure_ascii=False))
    print("转成 BIO：", tags)
    print(f"冲突 {len(conflicts)} 处（后写的 LOC 被 ORG 覆盖，信息丢失）：")
    for c in conflicts:
        print("  ", c)
    print("\n结论：BIO 适合「一个 token 最多属于一个实体」的 NER；")
    print("     重叠/嵌套实体、同一句多个事件、带角色的论元，通常还是用 start/end 或结构化 JSON。")


def main():
    args = parse_args()
    data_dir = resolve_data_dir(args.data_dir)
    label_names = load_label_names(data_dir)
    label2id, _ = build_label_maps(label_names)

    splits = {sp: json.load(open(data_dir / f"{sp}.json", "r", encoding="utf-8")) for sp in SPLITS}

    print("=" * 78)
    print("人民日报 NER（BIO）数据探查")
    print("=" * 78)
    print(f"数据目录：{data_dir}")
    print(f"标签体系（{len(label_names)} 个）：{label_names}")
    print(f"实体类型（3 个）：PER 人名 / ORG 机构 / LOC 地名")

    report = {"data_dir": str(data_dir), "label_names": label_names, "splits": {}}

    for sp in SPLITS:
        stats = basic_stats(splits[sp], label_names)
        report["splits"][sp] = stats
        print(f"\n[{sp}] {stats['n_sentences']} 条，{stats['n_chars']} 字")
        print(f"  句长（字）：mean={stats['char_len']['mean']}  p50={stats['char_len']['p50']}  "
              f"p90={stats['char_len']['p90']}  p95={stats['char_len']['p95']}  "
              f"p99={stats['char_len']['p99']}  max={stats['char_len']['max']}")
        print(f"  O 占比：{stats['o_ratio']:.2%}（标签极不均衡，所以不能只看 accuracy）")
        print(f"  实体：{stats['entity_counts']}，共 {stats['n_entities']} 个")
        print(f"  每句实体数：mean={stats['entities_per_sentence']['mean']}  "
              f"max={stats['entities_per_sentence']['max']}  "
              f"零实体句={stats['entities_per_sentence']['zero']}")
        print(f"  金标非法 BIO：{stats['illegal_sequences']} 条（转移 {stats['illegal_transitions']} 次）")

    report["duplicates"] = duplicate_overlap(splits)
    print(f"\n[重复检查] {report['duplicates']}")

    # tokenizer 统计需要本地 BERT；没装/没找到就跳过，不影响纯数据探查
    if not args.skip_tokenizer:
        try:
            from transformers import AutoTokenizer
            bert_path = resolve_bert_path(args.bert_path)
            tokenizer = AutoTokenizer.from_pretrained(str(bert_path), use_fast=True)
            print(f"\n[tokenizer] {bert_path}（{type(tokenizer).__name__}）")
            for sp in SPLITS:
                tstats = tokenizer_stats(splits[sp], tokenizer, args.max_length)
                report["splits"][sp]["tokenizer"] = tstats
                print(f"  [{sp}] token 长度：p50={tstats['token_len']['p50']} "
                      f"p95={tstats['token_len']['p95']} p99={tstats['token_len']['p99']} "
                      f"max={tstats['token_len']['max']}")
                print(f"        截断 @ max_length={args.max_length}：{tstats['n_truncated']} 条 "
                      f"({tstats['truncated_ratio']:.2%})，丢 {tstats['dropped_entities']} 个实体，"
                      f"{tstats['partial_entities']} 个实体跨边界")
                print(f"        一个字符被切成多个子词的次数：{tstats['multi_subword_chars']}")
        except Exception as e:
            print(f"\n[跳过 tokenizer 统计] {type(e).__name__}: {e}")
            print("  如需统计，用 --bert_path 指定本地 bert-base-chinese 目录。")

    show_one_sample(splits["train"], args.sample_index, label_names)
    show_overlap_limitation()

    ensure_output_dirs()
    out_path = LOG_DIR.parent / "data_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n[report] 数据统计已写入 → {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="人民日报 NER（BIO）数据探查")
    parser.add_argument("--data_dir", type=str, default=None, help="数据目录（默认 data/peoples_daily）")
    parser.add_argument("--bert_path", type=str, default=None, help="本地 bert-base-chinese 目录")
    parser.add_argument("--max_length", type=int, default=128, help="与训练一致的截断长度")
    parser.add_argument("--sample_index", type=int, default=0, help="展示第几条训练样本")
    parser.add_argument("--skip_tokenizer", action="store_true", help="跳过 tokenizer 长度统计")
    return parser.parse_args()


if __name__ == "__main__":
    main()
