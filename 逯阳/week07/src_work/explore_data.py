"""
人民日报 NER 数据集探索与可视化（作业用）

和 cluener 版的差异：
  - cluener 是 span 格式，统计时要遍历 {类型: {实体: [位置]}} 两层字典
  - 这里是 BIO 序列，直接扫 ner_tags 找 B-X 起点，向后数 I-X 得到实体长度
  - 额外统计：原始标注的非法 BIO 数（验证数据本身是干净的）、截断比例

输出（都在 outputs_work/figures/ 下）：
  work_entity_distribution.png        3 类实体频次
  work_text_length_distribution.png   文本长度分布（标注 max_length=128 的截断线）
  work_entity_length_distribution.png 实体长度分布
  work_data_stats.json                以上所有数字的机器可读版本

使用方式：
  python src_work/explore_data.py
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import argparse
from pathlib import Path
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
matplotlib.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "peoples_daily"
FIG_DIR = ROOT / "outputs_work" / "figures"
LOG_DIR = ROOT / "outputs_work" / "logs"

SPLITS = ["train", "validation", "test"]


def load(split: str) -> list:
    with open(DATA_DIR / f"{split}.json", "r", encoding="utf-8") as f:
        return json.load(f)


def extract_entities(tags: list[str]) -> list[tuple[str, int]]:
    """从 BIO 标签序列里抽实体，返回 [(类型, 长度), ...]。

    遇到 B-X 就开始记一个实体，后面连续 I-X 算同一个；遇到别的标签就结束。
    这也顺便验证了数据合法性：如果出现 I-X 打头，这里不会识别成实体。
    """
    ents = []
    cur_type, cur_len = None, 0
    for tag in tags:
        if tag.startswith("B-"):
            if cur_type:
                ents.append((cur_type, cur_len))
            cur_type, cur_len = tag[2:], 1
        elif tag.startswith("I-"):
            if cur_type == tag[2:]:
                cur_len += 1
            else:
                # I-X 但前面不是 B-X/I-X：非法，单独算一个长度为 1 的实体
                if cur_type:
                    ents.append((cur_type, cur_len))
                cur_type, cur_len = tag[2:], 1
        else:
            if cur_type:
                ents.append((cur_type, cur_len))
            cur_type, cur_len = None, 0
    if cur_type:
        ents.append((cur_type, cur_len))
    return ents


def count_illegal(tags: list[str]) -> int:
    """统计原始标注里的非法转移数（B-X → I-Y, X≠Y；O → I-X）。"""
    n = 0
    for i in range(1, len(tags)):
        prev, curr = tags[i - 1], tags[i]
        if curr.startswith("I-") and (prev == "O" or prev[2:] != curr[2:]):
            n += 1
    return n


def main():
    args = parse_args()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    stats = {}
    for split in SPLITS:
        rows = load(split)
        ent_counter = Counter()
        ent_len_counter = Counter()
        text_lens = []
        illegal = 0

        for row in rows:
            tags = row["ner_tags"]
            text_lens.append(len(row["tokens"]))
            illegal += count_illegal(tags)
            for etype, elen in extract_entities(tags):
                ent_counter[etype] += 1
                ent_len_counter[elen] += 1

        stats[split] = {
            "n_samples": len(rows),
            "n_entities": sum(ent_counter.values()),
            "entities_per_sample": round(sum(ent_counter.values()) / len(rows), 3),
            "entity_type_counts": dict(ent_counter.most_common()),
            "text_len": {
                "min": min(text_lens),
                "avg": round(sum(text_lens) / len(text_lens), 1),
                "max": max(text_lens),
                "p50": sorted(text_lens)[len(text_lens) // 2],
                "p95": sorted(text_lens)[int(len(text_lens) * 0.95)],
                "p99": sorted(text_lens)[int(len(text_lens) * 0.99)],
            },
            # 128 减去 [CLS]/[SEP] 两个位置
            "truncated_by_max128": sum(1 for x in text_lens if x > 126),
            "truncated_ratio": round(sum(1 for x in text_lens if x > 126) / len(text_lens), 4),
            "illegal_bio_in_gold": illegal,
        }

    # ── 终端打印 ────────────────────────────────────────────────────────────
    print("=" * 72)
    print("人民日报 NER 数据集统计")
    print("=" * 72)
    for split in SPLITS:
        s = stats[split]
        print(f"\n[{split}]  {s['n_samples']} 条，{s['n_entities']} 个实体，"
              f"平均每条 {s['entities_per_sample']} 个")
        print(f"  实体分布：{s['entity_type_counts']}")
        print(f"  文本长度：min={s['text_len']['min']} avg={s['text_len']['avg']} "
              f"p50={s['text_len']['p50']} p95={s['text_len']['p95']} "
              f"p99={s['text_len']['p99']} max={s['text_len']['max']}")
        print(f"  >126 字（max_length=128 会被截断）：{s['truncated_by_max128']} 条 "
              f"({s['truncated_ratio'] * 100:.2f}%)")
        print(f"  原始标注非法 BIO：{s['illegal_bio_in_gold']} 处")

    train = stats["train"]
    total = sum(train["entity_type_counts"].values())
    print("\n训练集类别占比（衡量不均衡程度）：")
    for etype, cnt in train["entity_type_counts"].items():
        print(f"  {etype:<5} {cnt:>6}  {cnt / total * 100:>5.1f}%")

    # ── 图 1：实体频次（三个 split 并排）────────────────────────────────────
    etypes = list(train["entity_type_counts"].keys())
    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.8 / len(SPLITS)
    for k, split in enumerate(SPLITS):
        counts = [stats[split]["entity_type_counts"].get(t, 0) for t in etypes]
        # 用比例而不是绝对数，否则 test 的柱子永远比 train 矮一大截
        ratios = [c / stats[split]["n_samples"] for c in counts]
        ax.bar([i + (k - 1) * width for i in range(len(etypes))], ratios, width,
               label=split)
    ax.set_xticks(range(len(etypes)))
    ax.set_xticklabels(etypes)
    ax.set_ylabel("每条样本平均实体数")
    ax.set_title("实体类型分布（按每条样本平均数量归一化）")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "work_entity_distribution.png", dpi=150)
    plt.close(fig)

    # ── 图 2：文本长度分布 ─────────────────────────────────────────────────
    lens = [len(r["tokens"]) for r in load("train")]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(lens, bins=60, color="#4C72B0", alpha=0.85)
    ax.axvline(126, color="#C44E52", linestyle="--", linewidth=1.6,
               label=f"max_length=128 截断线（超出 {train['truncated_ratio'] * 100:.1f}%）")
    ax.set_xlabel("文本长度（字）")
    ax.set_ylabel("样本数")
    ax.set_title("训练集文本长度分布")
    ax.set_xlim(0, 300)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "work_text_length_distribution.png", dpi=150)
    plt.close(fig)

    # ── 图 3：实体长度分布（按类型分组）────────────────────────────────────
    by_type = {t: Counter() for t in etypes}
    for row in load("train"):
        for etype, elen in extract_entities(row["ner_tags"]):
            by_type[etype][min(elen, 10)] += 1  # 10 及以上合并成一根柱子

    fig, ax = plt.subplots(figsize=(9, 5))
    xs = sorted({k for c in by_type.values() for k in c})
    width = 0.8 / len(etypes)
    for k, etype in enumerate(etypes):
        vals = [by_type[etype].get(x, 0) for x in xs]
        ax.bar([i + (k - (len(etypes) - 1) / 2) * width for i in range(len(xs))],
               vals, width, label=etype)
    ax.set_xticks(range(len(xs)))
    ax.set_xticklabels([str(x) if x < 10 else "10+" for x in xs])
    ax.set_xlabel("实体长度（字）")
    ax.set_ylabel("实体数")
    ax.set_title("训练集实体长度分布（按类型）")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "work_entity_length_distribution.png", dpi=150)
    plt.close(fig)

    out_path = LOG_DIR / "work_data_stats.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"\n图表已保存 → {FIG_DIR}")
    print(f"统计已保存 → {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="人民日报 NER 数据集探索")
    return parser.parse_args()


if __name__ == "__main__":
    main()
