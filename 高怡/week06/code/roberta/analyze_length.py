"""
统计多分类数据集（cls_train/val/test）的 token 长度分布，并画出截断风险图。

用法
----
python roberta/analyze_length.py                 # 默认 max_len=128
python roberta/analyze_length.py --max_len 256
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer

BASE_DIR = Path(__file__).resolve().parent.parent
SPLIT_DIR = BASE_DIR / "data" / "dataset" / "split"
MODEL_DIR = BASE_DIR / "pretrain_models" / "chinese-roberta-wwm-ext"
OUT_DIR = BASE_DIR / "output" / "roberta"

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

SPLITS = ["cls_train.jsonl", "cls_val.jsonl", "cls_test.jsonl"]


def load_texts() -> dict[str, list[str]]:
    out = {}
    for name in SPLITS:
        texts = []
        for line in (SPLIT_DIR / name).open(encoding="utf-8"):
            rec = json.loads(line)
            texts.append(rec["text"])
        out[name.replace("cls_", "").replace(".jsonl", "")] = texts
    return out


def describe(tag: str, arr: np.ndarray, max_len: int) -> dict:
    n = len(arr)
    over = int((arr > max_len).sum())
    return {
        "split": tag,
        "n": n,
        "mean": round(float(arr.mean()), 1),
        "median": int(np.median(arr)),
        "p90": int(np.percentile(arr, 90)),
        "p95": int(np.percentile(arr, 95)),
        "p99": int(np.percentile(arr, 99)),
        "max": int(arr.max()),
        "over_max_len": over,
        "over_ratio": round(over / n * 100, 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_len", type=int, default=128)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    data = load_texts()

    lengths = {}
    for tag, texts in data.items():
        lengths[tag] = np.array([len(tok(t)["input_ids"]) for t in texts])

    all_len = np.concatenate(list(lengths.values()))
    stats = [describe(k, v, args.max_len) for k, v in lengths.items()]
    stats.append(describe("all", all_len, args.max_len))

    for s in stats:
        print(f"[{s['split']:5s}] n={s['n']:6d} mean={s['mean']:6.1f} median={s['median']:4d} "
              f"p90={s['p90']:4d} p95={s['p95']:4d} p99={s['p99']:4d} max={s['max']:4d} "
              f"超{args.max_len}占比={s['over_ratio']}% ({s['over_max_len']} 条)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 图 1：长度直方图（全量）+ 分位数标注
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))

    ax = axes[0]
    upper = int(np.percentile(all_len, 99.5))
    bins = np.arange(0, max(upper, args.max_len) + 10, 5)
    ax.hist(all_len, bins=bins, color="#4C78A8", edgecolor="white", linewidth=0.6)
    ax.axvline(args.max_len, color="#E45756", linestyle="--", linewidth=1.8,
               label=f"max_length={args.max_len}")
    for p, c in [(50, "#54A24B"), (90, "#F58518"), (99, "#B279A2")]:
        v = np.percentile(all_len, p)
        ax.axvline(v, color=c, linestyle=":", linewidth=1.3)
        ax.text(v, ax.get_ylim()[1] * 0.92, f"P{p}={int(v)}", color=c,
                fontsize=8, rotation=90, ha="right", va="top")
    ax.set_xlim(0, max(upper, args.max_len) + 5)
    ax.set_xlabel("token 长度（含 [CLS]/[SEP]）")
    ax.set_ylabel("样本数")
    ax.set_title(f"全部样本长度分布（N={len(all_len):,}）")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.25, axis="y")

    # 图 2：累计分布曲线，直观看出覆盖率
    ax = axes[1]
    xs = np.arange(0, int(all_len.max()) + 2)
    cdf = np.array([(all_len <= x).mean() * 100 for x in xs])
    ax.plot(xs, cdf, color="#4C78A8", linewidth=1.8)
    ax.axvline(args.max_len, color="#E45756", linestyle="--", linewidth=1.8)
    cov = (all_len <= args.max_len).mean() * 100
    ax.scatter([args.max_len], [cov], color="#E45756", zorder=5, s=36)
    ax.annotate(f"max_length={args.max_len}\n覆盖 {cov:.2f}%",
                xy=(args.max_len, cov), xytext=(args.max_len + 25, cov - 12),
                fontsize=9, color="#E45756",
                arrowprops=dict(arrowstyle="->", color="#E45756", lw=1.2))
    ax.set_xlabel("token 长度")
    ax.set_ylabel("累计样本占比 (%)")
    ax.set_title("累计分布与截断覆盖率")
    ax.grid(alpha=0.25)
    ax.set_ylim(0, 102)

    fig.tight_layout()
    out = OUT_DIR / "length_distribution.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[图] {out}")

    (OUT_DIR / "length_stats.json").write_text(
        json.dumps({"max_len": args.max_len, "stats": stats}, indent=2, ensure_ascii=False),
        encoding="utf-8")


if __name__ == "__main__":
    main()
