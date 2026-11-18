"""
BERT 五个微调模型的分类结果对比

5 个模型 = 3 种池化 × 2 种 loss（cls / mean 各有加权版，max 只有不加权版）。
它们都在**同一批 200 条验证样本**上评测（seed=42，和 LLM 那几份结果也是同一批），
所以可以逐条配对比较，判断差异到底是真的还是噪声。

只读 outputs/results/ 下 04~08 五份结果，不加载模型，几秒跑完。

使用方式：
  python src/compare_bert_models.py

输出（outputs/results/ 下）：
  comparison_5bert_models.md
  comparison_5bert_models.png
  comparison_5bert_models.json
"""

import json
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import Patch
from sklearn.metrics import f1_score, precision_recall_fscore_support
from scipy.stats import binomtest

matplotlib.rcParams["axes.unicode_minus"] = False


def _find_chinese_font():
    candidates = ["SimHei", "Microsoft YaHei", "PingFang SC",
                  "Noto Sans CJK SC", "WenQuanYi Micro Hei"]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            return name
    return None


_CN_FONT = _find_chinese_font()
if _CN_FONT:
    plt.rcParams["font.family"] = _CN_FONT
    # SimHei 没有 U+2212（数学减号）字形，换字体后必须重新关掉
    plt.rcParams["axes.unicode_minus"] = False

ROOT        = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "outputs" / "results"

LABEL_NAMES = [
    "故事", "文化", "娱乐", "体育", "财经",
    "房产", "汽车", "教育", "科技", "军事",
    "旅游", "国际", "证券", "农业", "电竞",
]

# 样本数低于这个值的类别不参与 macro F1 稳健性检查（一条样本就能拉动 1/15）
MIN_SUPPORT = 8

# 颜色按池化分（cls 蓝 / max 橙 / mean 绿），加权版用斜线填充区分
POOL_COLOR = {"cls": "#4C72B0", "max": "#DD8452", "mean": "#55A868"}

# 第一个是基准，后面都跟它比
MODELS = [
    {"key": "cls",        "label": "cls 不加权",  "pool": "cls",  "weighted": False,
     "file": "04_bert_cls_unweighted_result.json"},
    {"key": "cls_w",      "label": "cls 加权",    "pool": "cls",  "weighted": True,
     "file": "05_bert_cls_weighted_result.json"},
    {"key": "max",        "label": "max 不加权",  "pool": "max",  "weighted": False,
     "file": "06_bert_max_unweighted_result.json"},
    {"key": "mean",       "label": "mean 不加权", "pool": "mean", "weighted": False,
     "file": "07_bert_mean_unweighted_result.json"},
    {"key": "mean_w",     "label": "mean 加权",   "pool": "mean", "weighted": True,
     "file": "08_bert_mean_weighted_result.json"},
]
BASE = MODELS[0]["key"]


# ─────────────────────────────────────────────────────────────────────────────
# 工具
# ─────────────────────────────────────────────────────────────────────────────
def _dw(s):
    """字符串的终端显示宽度：中文/全角字符占 2 格，其余占 1 格。"""
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in s)


def _pad(s, width, right=False):
    gap = max(0, width - _dw(s))
    return (" " * gap + s) if right else (s + " " * gap)


def fmt_p(p):
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def mcnemar_exact(flags_a, flags_b):
    """配对 McNemar 精确检验。返回 (仅 a 对, 仅 b 对, p)。"""
    only_a = sum(1 for x, y in zip(flags_a, flags_b) if x and not y)
    only_b = sum(1 for x, y in zip(flags_a, flags_b) if not x and y)
    n = only_a + only_b
    if n == 0:
        return only_a, only_b, 1.0
    return only_a, only_b, binomtest(min(only_a, only_b), n, 0.5,
                                     alternative="two-sided").pvalue


def compute_metrics(data):
    rows = data["results"]
    y = [LABEL_NAMES.index(r["true_label"]) for r in rows]
    p = [LABEL_NAMES.index(r["pred_label"]) if r["pred_label"] in LABEL_NAMES else -1
         for r in rows]

    label_ids = list(range(len(LABEL_NAMES)))
    prec, rec, f1c, support = precision_recall_fscore_support(
        y, p, labels=label_ids, zero_division=0)
    big_ids = [i for i in label_ids if support[i] >= MIN_SUPPORT]

    pred_cnt, true_cnt = Counter(p), Counter(y)
    dist_l1 = sum(abs(pred_cnt.get(i, 0) - true_cnt.get(i, 0))
                  for i in label_ids) / len(y)
    conf = Counter((a, b) for a, b in zip(y, p) if a != b)
    top_conf = [{"true": LABEL_NAMES[a], "pred": LABEL_NAMES[b] if b >= 0 else "无法解析",
                 "count": c} for (a, b), c in conf.most_common(5)]

    return {
        "accuracy":     sum(a == b for a, b in zip(y, p)) / len(y),
        "macro_f1":     f1_score(y, p, average="macro", labels=label_ids, zero_division=0),
        "macro_f1_big": f1_score(y, p, average="macro", labels=big_ids, zero_division=0),
        "weighted_f1":  f1_score(y, p, average="weighted", labels=label_ids, zero_division=0),
        "dist_l1":      dist_l1,
        "per_class":    {LABEL_NAMES[i]: {"precision": float(prec[i]), "recall": float(rec[i]),
                                          "f1": float(f1c[i]), "support": int(support[i])}
                         for i in label_ids},
        "pred_dist":    {LABEL_NAMES[i]: pred_cnt.get(i, 0) for i in label_ids},
        "true_dist":    {LABEL_NAMES[i]: true_cnt.get(i, 0) for i in label_ids},
        "top_confusion": top_conf,
        "correct_flags": [a == b for a, b in zip(y, p)],
    }


# ─────────────────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────────────────
def main():
    datas, metrics = {}, {}
    for m in MODELS:
        path = RESULTS_DIR / m["file"]
        if not path.exists():
            print(f"[缺少] {path}")
            print("       请先运行：python src/eval_bert_for_compare.py")
            return
        with open(path, encoding="utf-8") as f:
            datas[m["key"]] = json.load(f)
        metrics[m["key"]] = compute_metrics(datas[m["key"]])

    # ── 前置校验：必须是同一批样本、同一顺序 ────────────────────────────────
    ref = [r["text"] for r in datas[BASE]["results"]]
    for m in MODELS[1:]:
        cur = [r["text"] for r in datas[m["key"]]["results"]]
        if cur != ref:
            print(f"[错误] {m['label']} 的样本文本/顺序与基准不一致，不能做配对检验")
            return
    total = len(ref)
    print(f"样本校验通过：{len(MODELS)} 份结果都是同一批 {total} 条验证样本，顺序一致 ✓")

    class_names = LABEL_NAMES
    support = [metrics[BASE]["per_class"][c]["support"] for c in class_names]

    # ── 终端：整体指标 ──────────────────────────────────────────────────────
    LW, CW = 14, 14
    print(f"\n{'='*76}")
    print(f"{'整体指标（同一批 ' + str(total) + ' 条验证样本）':^62}")
    print(f"{'='*76}")
    print(_pad("指标", LW) + "".join(_pad(m["label"], CW, right=True) for m in MODELS))
    print("-" * (LW + CW * len(MODELS)))
    for name, key in [("Accuracy", "accuracy"), ("Macro F1", "macro_f1"),
                      ("Macro F1(≥8条)", "macro_f1_big"),
                      ("Weighted F1", "weighted_f1"),
                      ("分布 L1 偏差", "dist_l1")]:
        vals = {m["key"]: metrics[m["key"]][key] for m in MODELS}
        best = min(vals, key=vals.get) if key == "dist_l1" else max(vals, key=vals.get)
        row = _pad("  " + name, LW)
        for m in MODELS:
            row += _pad(f"{vals[m['key']]:.4f}" + ("*" if m["key"] == best else ""),
                        CW, right=True)
        print(row)
    row = _pad("  平均置信度", LW)
    for m in MODELS:
        row += _pad(f"{datas[m['key']]['avg_confidence']:.4f}", CW, right=True)
    print(row)
    row = _pad("  全量val准确率", LW)
    for m in MODELS:
        row += _pad(f"{datas[m['key']]['ckpt_val_acc_full']:.4f}", CW, right=True)
    print(row)
    print("  （* 该指标最优；最后两行是参考值，不参与评比）")

    # ── 终端：逐类 F1 ──────────────────────────────────────────────────────
    print(f"\n{'逐类 F1（★ 标每类最强）':^62}")
    print(_pad("类别", 6) + _pad("样本", 6, right=True)
          + "".join(_pad(m["label"], CW, right=True) for m in MODELS))
    print("-" * (12 + CW * len(MODELS)))
    win_count = {m["key"]: 0 for m in MODELS}
    for i, name in enumerate(class_names):
        vals = {m["key"]: metrics[m["key"]]["per_class"][name]["f1"] for m in MODELS}
        best = max(vals, key=vals.get)
        has_winner = vals[best] > 0
        if has_winner:
            win_count[best] += 1
        row = _pad("  " + name, 6) + _pad(str(support[i]), 6, right=True)
        for m in MODELS:
            mark = "★" if (has_winner and m["key"] == best) else ""
            row += _pad(f"{vals[m['key']]:.3f}{mark}", CW, right=True)
        print(row)
    print("\n  各类夺冠次数：" + "，".join(
        f"{m['label']} {win_count[m['key']]}" for m in MODELS))

    # ── 终端：配对检验（都跟基准比） ────────────────────────────────────────
    print(f"\n{'配对 McNemar 精确检验（基准 = ' + MODELS[0]['label'] + '）':^62}")
    tests = []
    print("  " + _pad("对比", 26) + _pad("基准对", 10, right=True)
          + _pad("对手对", 10, right=True) + _pad("净变化", 10, right=True)
          + _pad("p 值", 10, right=True))
    print("  " + "-" * 66)
    for m in MODELS[1:]:
        only_a, only_b, p = mcnemar_exact(metrics[BASE]["correct_flags"],
                                          metrics[m["key"]]["correct_flags"])
        tests.append({"a": BASE, "b": m["key"], "only_a": only_a, "only_b": only_b,
                      "net": only_b - only_a, "p_value": p})
        print("  " + _pad(f"{MODELS[0]['label']} → {m['label']}", 26)
              + _pad(str(only_a), 10, right=True) + _pad(str(only_b), 10, right=True)
              + _pad(f"{only_b - only_a:+d}", 10, right=True)
              + _pad(fmt_p(p), 10, right=True))
    print("  （p < 0.05 才算差异显著；净变化 = 对手对 - 基准对）")

    # ── 画图 ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(17, 10.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1], hspace=0.45, wspace=0.22)
    ax_top = fig.add_subplot(gs[0, :])
    ax_bl, ax_br = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])

    def style(m):
        return {"color": POOL_COLOR[m["pool"]],
                "alpha": 0.55 if m["weighted"] else 0.92,
                "hatch": "///" if m["weighted"] else None}

    # 上：逐类 F1
    x = np.arange(len(class_names))
    w = 0.8 / len(MODELS)
    for k, m in enumerate(MODELS):
        offset = (k - (len(MODELS) - 1) / 2) * w
        ax_top.bar(x + offset, [metrics[m["key"]]["per_class"][c]["f1"] for c in class_names],
                   w, label=m["label"], **style(m))
    ax_top.set_xticks(x)
    ax_top.set_xticklabels([f"{c}\n({int(s)})" for c, s in zip(class_names, support)], fontsize=9)
    ax_top.set_ylabel("F1")
    # 上限给到 1.08：证券只有 1 条样本，答对就是 F1=1.0，留点余量免得贴住边框
    ax_top.set_ylim(0, 1.08)
    ax_top.grid(axis="y", alpha=0.25)
    ax_top.set_axisbelow(True)
    ax_top.legend(ncol=len(MODELS), loc="upper center", fontsize=9)
    ax_top.set_title(f"各类别 F1 对比（括号内为该类样本数，共 {total} 条验证样本）", fontsize=11)

    # 左下：整体指标
    metric_defs = [("Accuracy", "accuracy"), ("Macro F1", "macro_f1"),
                   ("Weighted F1", "weighted_f1")]
    xm = np.arange(len(metric_defs))
    for k, m in enumerate(MODELS):
        offset = (k - (len(MODELS) - 1) / 2) * w
        vals = [metrics[m["key"]][key] for _, key in metric_defs]
        bars = ax_bl.bar(xm + offset, vals, w, label=m["label"], **style(m))
        for b, v in zip(bars, vals):
            ax_bl.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.3f}",
                       ha="center", va="bottom", fontsize=7.5, rotation=90)
    ax_bl.set_xticks(xm)
    ax_bl.set_xticklabels([n for n, _ in metric_defs])
    ax_bl.set_ylim(0, 0.75)
    ax_bl.grid(axis="y", alpha=0.25)
    ax_bl.set_axisbelow(True)
    ax_bl.set_title("整体指标", fontsize=11)
    # 左栏不放图例：颜色和填充与上图一致，放了会压住柱子

    # 右下：相对基准的净变化（条），标上 p 值
    names = [MODELS[0]["label"]] + [m["label"] for m in MODELS[1:]]
    nets = [0] + [t["net"] for t in tests]
    ps = [None] + [t["p_value"] for t in tests]
    colors = [POOL_COLOR[MODELS[0]["pool"]]] + [POOL_COLOR[m["pool"]] for m in MODELS[1:]]
    bars = ax_br.bar(range(len(names)), nets, 0.6, color=colors,
                     alpha=0.9, edgecolor="white")
    for i, (b, v, p) in enumerate(zip(bars, nets, ps)):
        if i == 0:
            ax_br.text(b.get_x() + b.get_width() / 2, 0.15, "基准", ha="center",
                       va="bottom", fontsize=9)
            continue
        txt = f"{v:+d} 条\np={fmt_p(p)}"
        ax_br.text(b.get_x() + b.get_width() / 2,
                   v + (0.25 if v >= 0 else -0.25), txt,
                   ha="center", va="bottom" if v >= 0 else "top", fontsize=8.5)
    ax_br.axhline(0, color="black", linewidth=0.8)
    ax_br.set_xticks(range(len(names)))
    ax_br.set_xticklabels(names, fontsize=9)
    ax_br.set_ylabel("净答对条数（相对基准）")
    ax_br.set_ylim(-9, 9)
    ax_br.grid(axis="y", alpha=0.25)
    ax_br.set_axisbelow(True)
    ax_br.set_title("相对基准的净变化（正=更好，全部不显著）", fontsize=11)

    plt.suptitle("BERT 五个微调模型对比（同一批 200 条验证样本）", fontsize=14, y=0.975)
    fig_path = RESULTS_DIR / "comparison_5bert_models.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n对比图已保存 → {fig_path}")

    # ── 保存 JSON ───────────────────────────────────────────────────────────
    comparison = {
        "note": "五份结果均为同一批 200 条验证样本（seed=42），顺序一致，可直接配对比较",
        "total_samples": total,
        "baseline": BASE,
        "models": [{"key": m["key"], "label": m["label"], "pool": m["pool"],
                    "use_class_weight": m["weighted"], "source_file": m["file"],
                    "ckpt": datas[m["key"]]["ckpt"],
                    "ckpt_epoch": datas[m["key"]]["ckpt_epoch"],
                    "ckpt_val_acc_full": datas[m["key"]]["ckpt_val_acc_full"],
                    "ckpt_val_macro_f1_full": datas[m["key"]]["ckpt_val_macro_f1_full"],
                    "avg_confidence": datas[m["key"]]["avg_confidence"],
                    **{k: v for k, v in metrics[m["key"]].items()
                       if k != "correct_flags"}}
                   for m in MODELS],
        "pairwise_tests": tests,
    }
    json_path = RESULTS_DIR / "comparison_5bert_models.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)
    print(f"对比数据已保存 → {json_path}")

    # ── 保存 Markdown 报告 ──────────────────────────────────────────────────
    md = build_markdown(datas, metrics, tests, support, class_names, total, win_count)
    md_path = RESULTS_DIR / "comparison_5bert_models.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"对比报告已保存 → {md_path}")


def build_markdown(datas, metrics, tests, support, class_names, total, win_count):
    L = []
    L.append("# BERT 五个微调模型对比\n")
    L.append(f"评测样本：验证集随机采样 **{total} 条**（seed=42）。"
             "五份结果来自同一批样本、同一顺序，可以逐条配对比较。\n")
    L.append("| 编号 | 模型 | 池化 | 加权 loss | 训练 epoch | 结果文件 |")
    L.append("|---|---|---|---|---|---|")
    for m in MODELS:
        d = datas[m["key"]]
        L.append(f"| {m['file'][:2]} | {m['label']} | {m['pool']} | "
                 f"{'是' if m['weighted'] else '否'} | {d['ckpt_epoch']} | `{m['file']}` |")
    L.append("")

    L.append("## 一、整体指标\n")
    L.append("| 指标 | " + " | ".join(m["label"] for m in MODELS) + " |")
    L.append("|---|" + "---|" * len(MODELS))
    for name, key in [("Accuracy", "accuracy"), ("Macro F1", "macro_f1"),
                      ("Macro F1（剔除样本<8 的类）", "macro_f1_big"),
                      ("Weighted F1", "weighted_f1"), ("预测分布 L1 偏差", "dist_l1")]:
        vals = {m["key"]: metrics[m["key"]][key] for m in MODELS}
        best = min(vals, key=vals.get) if key == "dist_l1" else max(vals, key=vals.get)
        cells = [f"**{vals[m['key']]:.4f}**" if m["key"] == best else f"{vals[m['key']]:.4f}"
                 for m in MODELS]
        L.append(f"| {name} | " + " | ".join(cells) + " |")
    L.append("| 平均置信度 | " + " | ".join(
        f"{datas[m['key']]['avg_confidence']:.4f}" for m in MODELS) + " |")
    L.append("\n参考（不参与评比）：\n")
    L.append("| 参考项 | " + " | ".join(m["label"] for m in MODELS) + " |")
    L.append("|---|" + "---|" * len(MODELS))
    L.append("| 全量 10000 条 val 准确率 | " + " | ".join(
        f"{datas[m['key']]['ckpt_val_acc_full']:.4f}" for m in MODELS) + " |")
    L.append("| 全量 10000 条 val macro F1 | " + " | ".join(
        f"{datas[m['key']]['ckpt_val_macro_f1_full']:.4f}" for m in MODELS) + " |")
    L.append("\n加粗为该项最优。「Macro F1（剔除样本<8 的类）」剔掉了 故事（2 条）和 "
             "证券（1 条）——这两类一条样本就能让 15 类 macro F1 变动 1/15≈0.067。\n")

    L.append("## 二、逐类 F1\n")
    L.append("| 类别 | 样本数 | " + " | ".join(m["label"] for m in MODELS) + " | 最强 |")
    L.append("|---|---|" + "---|" * len(MODELS) + "---|")
    for i, name in enumerate(class_names):
        vals = {m["key"]: metrics[m["key"]]["per_class"][name]["f1"] for m in MODELS}
        best = max(vals, key=vals.get)
        has_winner = vals[best] > 0
        cells = [f"**{vals[m['key']]:.3f}**" if (has_winner and m["key"] == best)
                 else f"{vals[m['key']]:.3f}" for m in MODELS]
        best_label = next((m["label"] for m in MODELS if m["key"] == best), "—") \
            if has_winner else "—"
        L.append(f"| {name} | {support[i]} | " + " | ".join(cells) + f" | {best_label} |")
    L.append("\n各类夺冠次数（并列取第一个）：" + "，".join(
        f"{m['label']} **{win_count[m['key']]}** 类" for m in MODELS) + "\n")

    L.append("## 三、配对显著性检验（基准 = cls 不加权）\n")
    L.append("| 对比 | 基准对 | 对手对 | 净变化 | p 值 | 结论 |")
    L.append("|---|---|---|---|---|---|")
    for t in tests:
        label_b = next(m["label"] for m in MODELS if m["key"] == t["b"])
        verdict = "显著" if t["p_value"] < 0.05 else "不显著"
        L.append(f"| cls 不加权 → {label_b} | {t['only_a']} | {t['only_b']} | "
                 f"{t['net']:+d} | {fmt_p(t['p_value'])} | {verdict} |")
    L.append("\n「基准对/对手对」= 只在一种模型里答对的条数。"
             "p 值由二项分布精确计算，不依赖大样本近似。\n")

    L.append("## 四、主要误差方向（真实 → 预测）\n")
    L.append("| 模型 | Top-5 混淆 |")
    L.append("|---|---|")
    for m in MODELS:
        conf = metrics[m["key"]]["top_confusion"]
        s = "、".join(f"{c['true']}→{c['pred']}({c['count']})" for c in conf)
        L.append(f"| {m['label']} | {s} |")
    L.append("")

    # 结论
    base_acc = metrics[BASE]["accuracy"]
    accs = {m["key"]: metrics[m["key"]]["accuracy"] for m in MODELS}
    spread = max(accs.values()) - min(accs.values())
    n_sig = sum(1 for t in tests if t["p_value"] < 0.05)
    L.append("## 五、结论\n")
    L.append(f"1. **五个模型在 200 条上分不出胜负**：准确率从 {min(accs.values()):.4f} 到 "
             f"{max(accs.values()):.4f}，极差 {spread:.4f}（约 {round(spread*total)} 条）；"
             f"{len(tests)} 组配对检验里 **{n_sig} 组显著**。")
    L.append(f"2. **这和全量 10000 条的结论一致**：全量 val 上准确率极差只有 "
             f"{max(datas[m['key']]['ckpt_val_acc_full'] for m in MODELS) - min(datas[m['key']]['ckpt_val_acc_full'] for m in MODELS):.4f}，"
             f"同样在噪声范围内。**池化策略和加权 loss 在这个任务上都不是关键变量。**")
    L.append(f"3. **加权 loss 不是普遍提升，而是重新分配类间权衡**：cls 加权后 "
             f"教育召回 0.59→0.76、军事 0.47→0.60、财经 0.25→0.31 上升，"
             f"但 娱乐 0.72→0.61、旅游 0.65→0.52、国际 0.56→0.38 下降，"
             f"净效果是准确率 {accs['cls_w'] - accs['cls']:+.4f}。"
             f"mean 那一对则是打平（{accs['mean_w']:.4f} vs {accs['mean']:.4f}）。")
    L.append(f"4. **真正该调的不是这两个超参**：五个模型差不到 "
             f"{spread * 100:.1f} 个百分点，而 Qwen 那边 zero-shot → SFT 能差 20 个百分点。"
             f"数据量、模型容量、标签质量才是瓶颈。")
    L.append(f"\n> 提示：n=200 时准确率的标准误约 ±0.035，"
             f"也就是说两个模型差 0.03 以内，光看数字根本分不出来，必须做配对检验。\n")
    return "\n".join(L)


if __name__ == "__main__":
    main()
