# -*- coding: utf-8 -*-
"""生成 DATA_REPORT 用的 5 张图（中文标签、白底、150dpi，可直接贴进 PPT）。

图 1 事件类型分布（单事件子集）       —— train_extract.jsonl
图 2 每篇事件数分布                   —— doc train.json / build_stats.json
图 3 角色出现次数 top30               —— train_extract.jsonl
图 4 句子长度分布                     —— train_detect.jsonl（含负样本，与线上数据同分布）
图 5 篇长分布                         —— doc train.json / build_stats.json

用法：
  D:\\Anaconda\\python.exe scripts\\make_figures.py
"""
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 无界面后端，避免在服务器/无显示器环境下报错
import matplotlib.pyplot as plt

DATA = Path(__file__).resolve().parent.parent
PROC = DATA / "processed"
STATS = DATA / "stats"
FIG = DATA / "figures"

# 中文字体：Windows 本机有 SimHei / Microsoft YaHei，按优先级挑第一个存在的
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans SC"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 11
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
BLUE = "#2E5E8C"
ORANGE = "#D9822B"


def load_jsonl(name):
    with open(PROC / name, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", FIG / name)


def fig_type_distribution():
    counts = Counter(r["event"]["type"] for r in load_jsonl("train_extract.jsonl"))
    items = counts.most_common()
    labels = [k for k, _ in items][::-1]
    values = [v for _, v in items][::-1]
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    bars = ax.barh(labels, values, color=BLUE, height=0.68)
    ax.bar_label(bars, padding=2, fontsize=10)
    ax.set_xlabel("单事件句子数（train n=5,747）")
    ax.set_title("DuEE-Fin 13 类事件分布（单事件子集）", fontsize=13, pad=10)
    ax.set_xlim(0, max(values) * 1.12)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "fig_event_type_distribution.png")


def fig_docs_events_per_doc():
    stats = json.loads((STATS / "build_stats.json").read_text(encoding="utf-8"))
    hist = {int(k): v for k, v in stats["doc_train"]["events_per_doc_hist"].items()}
    # 6 篇以上合并成一组，避免长尾把柱子压成一条线
    buckets = {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "≥6": 0}
    for k, v in hist.items():
        buckets["≥6" if k >= 6 else str(k)] += v
    labels = list(buckets)
    values = [buckets[k] for k in labels]
    total = sum(values)
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    bars = ax.bar(labels, values, color=BLUE, width=0.62)
    for b, v in zip(bars, values):
        ax.annotate("%d\n(%.1f%%)" % (v, 100 * v / total), (b.get_x() + b.get_width() / 2, v),
                    ha="center", va="bottom", fontsize=10)
    ax.set_xlabel("每篇公告的事件数")
    ax.set_ylabel("公告篇数")
    ax.set_title("DuEE-Fin 每篇公告事件数分布（train 7,047 篇）", fontsize=13, pad=10)
    ax.set_ylim(0, max(values) * 1.22)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "fig_docs_events_per_doc.png")


def fig_role_top30():
    roles = Counter()
    for r in load_jsonl("train_extract.jsonl"):
        for a in r["event"]["args"]:
            roles[a["role"]] += 1
    items = roles.most_common(30)
    labels = [k for k, _ in items][::-1]
    values = [v for _, v in items][::-1]
    fig, ax = plt.subplots(figsize=(8.2, 8.4))
    bars = ax.barh(labels, values, color=ORANGE, height=0.7)
    ax.bar_label(bars, padding=2, fontsize=9)
    ax.set_xlabel("出现次数（train_extract，共 %d 个角色）" % len(roles))
    ax.set_title("角色（论元槽位）出现次数 Top30", fontsize=13, pad=10)
    ax.set_xlim(0, max(values) * 1.12)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "fig_role_top30.png")


def fig_sentence_length():
    lens = sorted(len(r["text"]) for r in load_jsonl("train_detect.jsonl"))
    p50, p90, p99 = lens[len(lens) // 2], lens[int(len(lens) * 0.9)], lens[int(len(lens) * 0.99)]
    clip = 400  # 直方图只画到 400 字，长尾在注释里说明
    shown = [min(x, clip) for x in lens]
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.hist(shown, bins=40, color=BLUE, alpha=0.9)
    for x, label in [(p50, "中位数 %d" % p50), (p90, "p90 %d" % p90), (p99, "p99 %d" % p99)]:
        ax.axvline(x, color=ORANGE, linestyle="--", linewidth=1.4)
        ax.annotate(label, (x, ax.get_ylim()[1] * 0.92), rotation=90, color=ORANGE,
                    fontsize=10, ha="right", va="top")
    ax.set_xlabel("句子字符数（train 全部 %d 句，直方图截断在 %d 字）" % (len(lens), clip))
    ax.set_ylabel("句子数")
    ax.set_title("DuEE-Fin 句子长度分布（含负样本）", fontsize=13, pad=10)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "fig_sentence_length.png")


def fig_doc_length():
    lengths = []
    for line in open(DATA / "raw" / "DuEE_Fin_mirror" / "DuEE-pytorch__data__DuEE-Fin__train.json",
                     encoding="utf-8"):
        line = line.strip()
        if line:
            lengths.append(len(json.loads(line)["text"]))
    s = sorted(lengths)
    p50, p90, p99 = s[len(s) // 2], s[int(len(s) * 0.9)], s[int(len(s) * 0.99)]
    clip = 1600
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.hist([min(x, clip) for x in lengths], bins=50, color=BLUE, alpha=0.9)
    for x, label in [(p50, "中位数 %d" % p50), (p90, "p90 %d" % p90), (p99, "p99 %d" % p99)]:
        ax.axvline(x, color=ORANGE, linestyle="--", linewidth=1.4)
        ax.annotate(label, (x, ax.get_ylim()[1] * 0.92), rotation=90, color=ORANGE,
                    fontsize=10, ha="right", va="top")
    ax.set_xlabel("公告正文字符数（train 全部 %d 篇，直方图截断在 %d 字）" % (len(lengths), clip))
    ax.set_ylabel("公告篇数")
    ax.set_title("DuEE-Fin 篇章长度分布", fontsize=13, pad=10)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "fig_doc_length.png")


def fig_args_per_event():
    """每事件的论元数分布（原任务要求的“论元数分布”，也给课件讲“一篇公告抽几个槽位”用）。"""
    counts = Counter(len(r["event"]["args"]) for r in load_jsonl("train_extract.jsonl"))
    keys = sorted(counts)
    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    bars = ax.barh([str(k) for k in keys], [counts[k] for k in keys], color=BLUE, height=0.68)
    total = sum(counts.values())
    ax.bar_label(bars, labels=["%d" % counts[k] for k in keys], padding=3, fontsize=9)
    ax.set_xlabel("事件数（train_extract n=5,747；每事件平均 4.38 个论元）")
    ax.set_ylabel("每事件论元个数")
    ax.set_title("每事件论元数分布", fontsize=13, pad=10)
    ax.set_xlim(0, max(counts.values()) * 1.12)
    ax.invert_yaxis()  # 论元少的在上面，读图顺序与表格一致
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "fig_args_per_event.png")


if __name__ == "__main__":
    fig_type_distribution()
    fig_docs_events_per_doc()
    fig_role_top30()
    fig_sentence_length()
    fig_doc_length()
    fig_args_per_event()
