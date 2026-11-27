"""
汇总 BERT+Linear 与 BERT+CRF 在人民日报 NER 上的结果（作业用）

只读 outputs_work/logs/work_eval_*.json，不加载模型，秒级出结果。
所有数字都从文件里读，不写死 —— 改一次超参重跑训练，这里跟着变。

输出：
  终端：整体对比表 + 逐类型 F1
  outputs_work/work_compare_{split}.md          对比报告
  outputs_work/logs/work_compare_{split}.json   机器可读
  outputs_work/figures/work_model_compare_{split}.png  对比图

使用方式：
  python src_work/compare_results.py
  python src_work/compare_results.py --split test
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
matplotlib.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).parent.parent
WORK_DIR = ROOT / "outputs_work"
LOG_DIR = WORK_DIR / "logs"
FIG_DIR = WORK_DIR / "figures"

MODELS = [
    {"key": "linear", "label": "BERT + Linear", "color": "#4C72B0"},
    {"key": "crf",    "label": "BERT + CRF",    "color": "#DD8452"},
]


def _dw(s: str) -> int:
    """字符串显示宽度：中文算 2 列，用于终端表格对齐。"""
    import unicodedata
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)


def _pad(s: str, width: int, right: bool = False) -> str:
    fill = " " * max(0, width - _dw(s))
    return fill + s if right else s + fill


def load_results(split: str, tag: str = "") -> dict:
    """读取两个模型的评估结果，缺哪个就跳过哪个。

    tag 是 evaluate.py --tag 的后缀，传 "full" 就读放开截断的那份（max_length=512）。
    """
    out = {}
    suffix = f"_{tag}" if tag else ""
    for m in MODELS:
        path = LOG_DIR / f"work_eval_{m['key']}_{split}{suffix}.json"
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                out[m["key"]] = json.load(f)
        else:
            print(f"[警告] 缺少 {path.name}，请先运行 evaluate.py")
    return out


def main():
    args = parse_args()
    tag_suffix = f"_{args.tag}" if args.tag else ""
    res = load_results(args.split, args.tag)
    if not res:
        print("没有任何评估结果，先跑 train.py + evaluate.py")
        return

    avail = [m for m in MODELS if m["key"] in res]
    train_logs = {}
    for m in avail:
        p = LOG_DIR / f"work_train_{m['key']}.json"
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                train_logs[m["key"]] = json.load(f)

    # ── 终端：整体指标 ──────────────────────────────────────────────────────
    print("\n" + "=" * 74)
    print(f"人民日报 NER — 模型对比（评估集：{args.split}）")
    print("=" * 74)
    print(_pad("模型", 18) + _pad("Precision", 12, True) + _pad("Recall", 12, True)
          + _pad("F1", 12, True) + _pad("非法序列", 12, True) + _pad("样本数", 10, True))
    print("-" * 76)
    for m in avail:
        r = res[m["key"]]
        print(_pad(m["label"], 18) + _pad(f"{r['precision']:.4f}", 12, True)
              + _pad(f"{r['recall']:.4f}", 12, True) + _pad(f"{r['f1']:.4f}", 12, True)
              + _pad(str(r["illegal_stats"]["total_illegal"]), 12, True)
              + _pad(str(r["n_samples"]), 10, True))

    # ── 终端：逐类型 F1 ────────────────────────────────────────────────────
    types = []
    for m in avail:
        for t in res[m["key"]]["per_type"]:
            if t not in types:
                types.append(t)

    print("\n" + _pad("实体类型", 12) + _pad("样本数", 10, True)
          + "".join(_pad(m["label"], 18, True) for m in avail))
    print("-" * (22 + 18 * len(avail)))
    for t in types:
        sup = res[avail[0]["key"]]["per_type"][t]["support"]
        row = _pad(t, 12) + _pad(str(int(sup)), 10, True)
        for m in avail:
            row += _pad(f"{res[m['key']]['per_type'][t]['f1-score']:.4f}", 18, True)
        print(row)

    # ── 结论（全部从数据推出来，不写死）─────────────────────────────────────
    lines = []
    print("\n" + "=" * 74)
    if "linear" in res and "crf" in res:
        dl = res["linear"]
        dc = res["crf"]
        f1_diff = dc["f1"] - dl["f1"]
        ill_l = dl["illegal_stats"]["total_illegal"]
        ill_c = dc["illegal_stats"]["total_illegal"]
        sign = "↑" if f1_diff >= 0 else "↓"
        print("关键结论：")
        print(f"  1. CRF vs Linear：F1 {sign}{abs(f1_diff):.4f}"
              f"（{dl['f1']:.4f} → {dc['f1']:.4f}）")
        print(f"  2. 非法序列：Linear {ill_l} 条 → CRF {ill_c} 条"
              f"（减少 {ill_l - ill_c} 条）")
        print("     CRF 的转移矩阵是学出来的，不是硬约束，所以不保证 0 条")
        if ill_c > 0:
            print("     想彻底杜绝，需要给非法转移位置加惩罚（本作业未加，保持纯结构对比）")
        lines.append(f"- CRF vs Linear：F1 {dl['f1']:.4f} → {dc['f1']:.4f}"
                     f"（{f1_diff:+.4f}）")
        lines.append(f"- 非法序列：Linear {ill_l} 条 → CRF {ill_c} 条（减少 {ill_l - ill_c} 条）")
    else:
        print("  只跑了一个模型，无法对比 CRF 与 Linear 的差异")
    print("=" * 74)

    # ── 图：逐类型 F1 + 整体指标 ───────────────────────────────────────────
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1.25, 1]})

    ax = axes[0]
    x = range(len(types))
    w = 0.8 / len(avail)
    for k, m in enumerate(avail):
        vals = [res[m["key"]]["per_type"][t]["f1-score"] for t in types]
        ax.bar([i + (k - (len(avail) - 1) / 2) * w for i in x], vals, w,
               label=m["label"], color=m["color"], alpha=0.9)
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{t}\n({int(res[avail[0]['key']]['per_type'][t]['support'])})"
                        for t in types])
    ax.set_ylabel("F1")
    # 两个模型都在 0.90 以上，纵轴从 0 起会把差异压成一条线，所以截到 0.85 起
    ax.set_ylim(0.85, 1.0)
    ax.set_title("逐类型 F1（括号内为实体数）")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    for k, m in enumerate(avail):
        for i, t in enumerate(types):
            v = res[m["key"]]["per_type"][t]["f1-score"]
            ax.text(i + (k - (len(avail) - 1) / 2) * w, v + 0.003, f"{v:.3f}",
                    ha="center", fontsize=8.5)

    ax = axes[1]
    metric_names = ["Precision", "Recall", "F1"]
    metric_keys = ["precision", "recall", "f1"]
    x = range(len(metric_names))
    for k, m in enumerate(avail):
        vals = [res[m["key"]][key] for key in metric_keys]
        ax.bar([i + (k - (len(avail) - 1) / 2) * w for i in x], vals, w,
               label=m["label"], color=m["color"], alpha=0.9)
        for i, v in enumerate(vals):
            ax.text(i + (k - (len(avail) - 1) / 2) * w, v + 0.01, f"{v:.3f}",
                    ha="center", fontsize=9)
    ax.set_xticks(list(x))
    ax.set_xticklabels(metric_names)
    ax.set_ylim(0.85, 1.0)
    ax.set_title("整体指标")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    fig.suptitle(f"人民日报 NER：BERT+Linear vs BERT+CRF（{args.split} 集）", fontsize=13)
    fig.tight_layout()
    fig_path = FIG_DIR / f"work_model_compare_{args.split}{tag_suffix}.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    # ── 保存 md / json ─────────────────────────────────────────────────────
    md = [f"# 人民日报 NER：模型对比（{args.split} 集）\n"]
    md.append("| 模型 | Precision | Recall | F1 | 非法序列 | 样本数 |")
    md.append("|---|---|---|---|---|---|")
    for m in avail:
        r = res[m["key"]]
        md.append(f"| {m['label']} | {r['precision']:.4f} | {r['recall']:.4f} | "
                  f"{r['f1']:.4f} | {r['illegal_stats']['total_illegal']} | {r['n_samples']} |")
    md.append("\n## 逐类型 F1\n")
    md.append("| 实体类型 | 样本数 | " + " | ".join(m["label"] for m in avail) + " |")
    md.append("|---|---|" + "---|" * len(avail))
    for t in types:
        cells = [f"{res[m['key']]['per_type'][t]['f1-score']:.4f}" for m in avail]
        md.append(f"| {t} | {int(res[avail[0]['key']]['per_type'][t]['support'])} | "
                  + " | ".join(cells) + " |")
    if lines:
        md.append("\n## 结论\n")
        md.extend(lines)
    if train_logs:
        md.append("\n## 训练过程\n")
        md.append("| 模型 | epoch | train_loss | val_loss | val_F1 | 耗时(s) |")
        md.append("|---|---|---|---|---|---|")
        for m in avail:
            for rec in train_logs.get(m["key"], []):
                md.append(f"| {m['label']} | {rec['epoch']} | {rec['train_loss']:.4f} | "
                          f"{rec['val_loss']:.4f} | {rec['val_entity_f1']:.4f} | "
                          f"{rec['elapsed_s']:.0f} |")
    md.append("\n> 评估口径：seqeval entity-level F1（类型与边界完全一致才算 TP）。")
    md.append("> 数据：人民日报 NER，train 20864 / validation 2318 / test 4636，3 类实体。")
    if args.tag:
        md.append(f"> 评估用 `max_length` 来自 `--tag {args.tag}` 对应的评估文件（放开截断）；"
                  f"上面「训练过程」表的 val_F1 是训练时口径（max_length=128）测的，"
                  f"两处会差 0.001 量级，不是笔误。")

    md_path = WORK_DIR / f"work_compare_{args.split}{tag_suffix}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    json_path = LOG_DIR / f"work_compare_{args.split}{tag_suffix}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"split": args.split,
                   "models": [{"key": m["key"], "label": m["label"],
                               **{k: v for k, v in res[m["key"]].items()
                                  if k != "per_type"}} for m in avail],
                   "per_type": {m["key"]: res[m["key"]]["per_type"] for m in avail}},
                  f, ensure_ascii=False, indent=2)

    print(f"\n对比报告 → {md_path}")
    print(f"对比数据 → {json_path}")
    print(f"对比图   → {fig_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="汇总人民日报 NER 两模型对比")
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--tag", default="",
                        help="读 work_eval_*_{split}{tag}.json，输出也带同样的后缀。"
                             "传 full 就是放开截断（max_length=512）的那份")
    return parser.parse_args()


if __name__ == "__main__":
    main()
