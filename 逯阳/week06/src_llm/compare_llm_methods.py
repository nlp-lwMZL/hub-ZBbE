"""
四种方案的结果对比：zero-shot / few-shot / SFT(LoRA) / BERT 微调

只读已保存的评测结果，不加载模型、不重新推理，所以几秒就能跑完。
四份结果都是同一批 200 条验证样本（seed=42 抽样），顺序一致，因此可以做配对检验。

输入（都在 outputs/results/ 下，文件名自带方法名）：
  01_zeroshot_qwen2-0.5b_result.json    基础模型直接问
  02_fewshot_qwen2-0.5b_result.json     每类 2 个示例
  03_sft_lora_qwen2-0.5b_result.json    LoRA 微调后
  04_bert_cls_unweighted_result.json     BERT 微调后（判别式，cls 不加权）

输出（同样在 outputs/results/ 下）：
  comparison_4methods_zeroshot_fewshot_sft_bert.md    给人看的对比报告
  comparison_4methods_zeroshot_fewshot_sft_bert.json  给程序读的对比数据
  comparison_4methods_zeroshot_fewshot_sft_bert.png   对比图

使用方式：
  python src_llm/compare_llm_methods.py
"""

import json
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
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

# 样本数低于这个值的类别不参与 macro F1 稳健性检查（F1 靠运气，一条样本就能拉动 1/15）
MIN_SUPPORT = 8

# 四种方法固定用这四个颜色，前三个和池化对比图保持同一套视觉习惯
METHODS = [
    {
        "key":   "zero-shot",
        "label": "zero-shot",
        "desc":  "基础模型直接问",
        "path":  RESULTS_DIR / "01_zeroshot_qwen2-0.5b_result.json",
        "color": "#4C72B0",
    },
    {
        "key":   "few-shot",
        "label": "few-shot (2 例/类)",
        "desc":  "prompt 里塞 30 组示例",
        "path":  RESULTS_DIR / "02_fewshot_qwen2-0.5b_result.json",
        "color": "#DD8452",
    },
    {
        "key":   "sft",
        "label": "SFT (LoRA)",
        "desc":  "Qwen2-0.5B 微调 5000 条",
        "path":  RESULTS_DIR / "03_sft_lora_qwen2-0.5b_result.json",
        "color": "#55A868",
    },
    {
        "key":   "bert",
        "label": "BERT 微调 (cls)",
        "desc":  "全量 53K 条训练数据",
        "path":  RESULTS_DIR / "04_bert_cls_unweighted_result.json",
        "color": "#8172B2",
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# 指标计算
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(data):
    """把一份评测结果算成各种指标。无法解析的预测按 -1 处理，算错。"""
    rows = data["results"]
    y = [LABEL_NAMES.index(r["true_label"]) for r in rows]
    # pred_label 为 None 表示模型输出了 15 个类别以外的词，直接算错
    p = [LABEL_NAMES.index(r["pred_label"]) if r["pred_label"] in LABEL_NAMES else -1
         for r in rows]

    label_ids = list(range(len(LABEL_NAMES)))
    prec, rec, f1c, support = precision_recall_fscore_support(
        y, p, labels=label_ids, zero_division=0)

    # 稳健性检查：样本数太少的类（故事 2 条、证券 1 条）F1 完全靠运气，
    # 一条样本就能让 macro F1 波动 1/15≈0.067，所以再算一个剔除它们的版本
    big_ids = [i for i in label_ids if support[i] >= MIN_SUPPORT]
    macro_f1_big = f1_score(y, p, average="macro", labels=big_ids, zero_division=0)

    # 预测分布和真实分布的 L1 距离：0 表示分布完全一致，越大越偏
    pred_cnt = Counter(p)
    true_cnt = Counter(y)
    dist_l1 = sum(abs(pred_cnt.get(i, 0) - true_cnt.get(i, 0))
                  for i in label_ids) / len(y)

    # 错得最多的方向（真实 → 预测），用来解释误差来源
    conf = Counter((a, b) for a, b in zip(y, p) if a != b)
    top_conf = [{"true": LABEL_NAMES[a], "pred": LABEL_NAMES[b] if b >= 0 else "无法解析",
                 "count": c} for (a, b), c in conf.most_common(5)]

    unparseable = sum(1 for x in p if x < 0)

    return {
        "accuracy":     sum(a == b for a, b in zip(y, p)) / len(y),
        "macro_f1":     f1_score(y, p, average="macro", labels=label_ids, zero_division=0),
        "macro_f1_big": macro_f1_big,
        "weighted_f1":  f1_score(y, p, average="weighted", labels=label_ids, zero_division=0),
        "unparseable":  unparseable,
        "parse_rate":   1 - unparseable / len(y),
        "dist_l1":      dist_l1,
        "per_class":    {LABEL_NAMES[i]: {"precision": float(prec[i]), "recall": float(rec[i]),
                                          "f1": float(f1c[i]), "support": int(support[i])}
                         for i in label_ids},
        "pred_dist":    {LABEL_NAMES[i]: pred_cnt.get(i, 0) for i in label_ids},
        "true_dist":    {LABEL_NAMES[i]: true_cnt.get(i, 0) for i in label_ids},
        "top_confusion": top_conf,
        "correct_flags": [a == b for a, b in zip(y, p)],
    }


def mcnemar_exact(flags_a, flags_b):
    """配对样本的 McNemar 精确检验。

    返回 (仅 a 对, 仅 b 对, p)：
      仅 a 对 = a 答对而 b 答错
      仅 b 对 = b 答对而 a 答错
    净变化 = 仅 b 对 - 仅 a 对，正数表示 b 比 a 强。
    """
    only_a = sum(1 for x, y in zip(flags_a, flags_b) if x and not y)
    only_b = sum(1 for x, y in zip(flags_a, flags_b) if not x and y)
    n = only_a + only_b
    if n == 0:
        return only_a, only_b, 1.0
    return only_a, only_b, binomtest(min(only_a, only_b), n, 0.5,
                                     alternative="two-sided").pvalue


def fmt_p(p):
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def fmt_p_eq(p):
    """用在句子里，带上 "p =" 前缀。"""
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def _dw(s):
    """字符串的终端显示宽度：中文/全角字符占 2 格，其余占 1 格。"""
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in s)


def _pad(s, width, right=False):
    gap = max(0, width - _dw(s))
    return (" " * gap + s) if right else (s + " " * gap)


# ─────────────────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # ── 读结果 ──────────────────────────────────────────────────────────────
    datas, metrics = {}, {}
    for m in METHODS:
        if not m["path"].exists():
            print(f"[缺少] {m['path']}")
            print(f"       请先运行对应的评测脚本生成这份结果")
            return
        with open(m["path"], encoding="utf-8") as f:
            datas[m["key"]] = json.load(f)
        metrics[m["key"]] = compute_metrics(datas[m["key"]])

    # ── 前置校验：必须是同一批样本、同一顺序，否则配对检验无效 ────────────────
    ref = [r["text"] for r in datas["zero-shot"]["results"]]
    for m in METHODS[1:]:
        cur = [r["text"] for r in datas[m["key"]]["results"]]
        if len(cur) != len(ref):
            print(f"[错误] {m['label']} 是 {len(cur)} 条，zero-shot 是 {len(ref)} 条，样本规模不同")
            return
        if cur != ref:
            print(f"[错误] {m['label']} 的样本文本/顺序与 zero-shot 不一致，不能做配对检验")
            return
    total = len(ref)
    print(f"样本校验通过：{len(METHODS)} 份结果都是同一批 {total} 条验证样本，顺序一致 ✓")

    class_names = LABEL_NAMES
    support = [metrics["zero-shot"]["per_class"][c]["support"] for c in class_names]

    # ── 终端：整体指标 ──────────────────────────────────────────────────────
    LW = 17   # 指标名列的显示宽度
    CW = 20   # 每个方法列的显示宽度（"few-shot (2 例/类)" 正好 18 格，留 2 格间隙）
    print(f"\n{'='*74}")
    print(f"{'整体指标（同一批 ' + str(total) + ' 条验证样本）':^60}")
    print(f"{'='*74}")
    print(_pad("指标", LW) + "".join(_pad(m["label"], CW, right=True) for m in METHODS))
    print("-" * (LW + CW * len(METHODS)))
    for name, key in [("Accuracy", "accuracy"), ("Macro F1", "macro_f1"),
                      ("Macro F1(≥8条)", "macro_f1_big"),
                      ("Weighted F1", "weighted_f1"), ("可解析率", "parse_rate")]:
        vals = {m["key"]: metrics[m["key"]][key] for m in METHODS}
        best = max(vals, key=vals.get)
        row = _pad("  " + name, LW)
        for m in METHODS:
            row += _pad(f"{vals[m['key']]:.4f}" + ("*" if m["key"] == best else ""), CW, right=True)
        print(row)
    row = _pad("  无法解析条数", LW)
    for m in METHODS:
        row += _pad(str(metrics[m["key"]]["unparseable"]), CW, right=True)
    print(row)
    print("  （* 表示该指标最优）")

    # ── 终端：逐类 F1 ──────────────────────────────────────────────────────
    print(f"\n{'逐类 F1（★ 标每类最强）':^60}")
    print(_pad("类别", 6) + _pad("样本", 6, right=True)
          + "".join(_pad(m["label"], CW, right=True) for m in METHODS))
    print("-" * (12 + CW * len(METHODS)))
    best_count = {m["key"]: 0 for m in METHODS}
    for i, name in enumerate(class_names):
        vals = {m["key"]: metrics[m["key"]]["per_class"][name]["f1"] for m in METHODS}
        best = max(vals, key=vals.get)
        # 三个方法都没答对（比如只有 1 条样本的证券），标"最强"没有意义
        has_winner = vals[best] > 0
        if has_winner:
            best_count[best] += 1
        row = _pad("  " + name, 6) + _pad(str(support[i]), 6, right=True)
        for m in METHODS:
            mark = "★" if (has_winner and m["key"] == best) else ""
            row += _pad(f"{vals[m['key']]:.3f}{mark}", CW, right=True)
        print(row)
    print("\n  各类夺冠次数：" + "，".join(
        f"{m['label']} {best_count[m['key']]} 类" for m in METHODS))

    # ── 终端：配对显著性检验 ────────────────────────────────────────────────
    print(f"\n{'配对 McNemar 精确检验（同一批样本，逐条比对）':^60}")
    pairs = [("zero-shot", "few-shot"), ("zero-shot", "sft"), ("few-shot", "sft"),
             ("zero-shot", "bert"), ("sft", "bert")]
    tests = []
    print("  " + _pad("对比", 26) + _pad("仅前者对", 12, right=True)
          + _pad("仅后者对", 12, right=True) + _pad("净变化", 10, right=True)
          + _pad("p 值", 10, right=True))
    print("  " + "-" * 70)
    for a, b in pairs:
        only_a, only_b, p = mcnemar_exact(metrics[a]["correct_flags"],
                                          metrics[b]["correct_flags"])
        net = only_b - only_a
        tests.append({"a": a, "b": b, "only_a": only_a, "only_b": only_b,
                      "net": net, "p_value": p})
        print("  " + _pad(f"{a} → {b}", 26) + _pad(str(only_a), 12, right=True)
              + _pad(str(only_b), 12, right=True) + _pad(f"{net:+d}", 10, right=True)
              + _pad(fmt_p(p), 10, right=True))
    print("  （p < 0.05 才算差异显著；净变化 = 仅后者对 - 仅前者对，正数表示后者更强）")

    # ── 画图 ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 10.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1], hspace=0.45, wspace=0.22)
    ax_top, ax_bl, ax_br = fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])

    # 上：逐类 F1，四色并排
    x = np.arange(len(class_names))
    w = 0.8 / len(METHODS)
    for k, m in enumerate(METHODS):
        offset = (k - (len(METHODS) - 1) / 2) * w
        ax_top.bar(x + offset, [metrics[m["key"]]["per_class"][c]["f1"] for c in class_names],
                   w, label=m["label"], color=m["color"], alpha=0.9)
    ax_top.set_xticks(x)
    ax_top.set_xticklabels([f"{c}\n({int(s)})" for c, s in zip(class_names, support)], fontsize=9)
    ax_top.set_ylabel("F1")
    # 上限给到 1.0：证券只有 1 条样本，BERT 恰好答对，F1 = 1.0，不能被截断
    ax_top.set_ylim(0, 1.0)
    ax_top.grid(axis="y", alpha=0.25)
    ax_top.set_axisbelow(True)
    ax_top.legend(title="方法", ncol=len(METHODS), loc="upper center", fontsize=9)
    ax_top.set_title(f"各类别 F1 对比（括号内为该类样本数，共 {total} 条验证样本）", fontsize=11)

    # 左下：整体指标
    metric_defs = [("Accuracy", "accuracy"), ("Macro F1", "macro_f1"), ("Weighted F1", "weighted_f1")]
    xm = np.arange(len(metric_defs))
    for k, m in enumerate(METHODS):
        offset = (k - (len(METHODS) - 1) / 2) * w
        vals = [metrics[m["key"]][key] for _, key in metric_defs]
        bars = ax_bl.bar(xm + offset, vals, w, label=m["label"], color=m["color"], alpha=0.9)
        for b, v in zip(bars, vals):
            ax_bl.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.3f}",
                       ha="center", va="bottom", fontsize=8, rotation=90)
    ax_bl.set_xticks(xm)
    ax_bl.set_xticklabels([n for n, _ in metric_defs])
    ax_bl.set_ylim(0, 0.75)
    ax_bl.grid(axis="y", alpha=0.25)
    ax_bl.set_axisbelow(True)
    ax_bl.set_title("整体指标", fontsize=11)
    # 左栏不放图例，颜色与上图一致，放了会压住柱子

    # 右下：格式问题（无法解析率）+ 类别分布偏差
    diag_defs = [("无法解析率", lambda mt: mt["unparseable"] / total),
                 ("预测分布 L1 偏差", lambda mt: mt["dist_l1"])]
    xd = np.arange(len(diag_defs))
    for k, m in enumerate(METHODS):
        offset = (k - (len(METHODS) - 1) / 2) * w
        vals = [fn(metrics[m["key"]]) for _, fn in diag_defs]
        bars = ax_br.bar(xd + offset, vals, w, color=m["color"], alpha=0.9)
        for b, v in zip(bars, vals):
            ax_br.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.3f}",
                       ha="center", va="bottom", fontsize=8)
    ax_br.set_xticks(xd)
    ax_br.set_xticklabels([n for n, _ in diag_defs])
    ax_br.set_ylim(0, 0.72)
    ax_br.grid(axis="y", alpha=0.25)
    ax_br.set_axisbelow(True)
    ax_br.set_title("格式遵循与类别分布偏差（越低越好）", fontsize=11)

    plt.suptitle("四种分类方案对比：zero-shot / few-shot / SFT-LoRA / BERT 微调",
                 fontsize=14, y=0.975)
    fig_path = RESULTS_DIR / "comparison_4methods_zeroshot_fewshot_sft_bert.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n对比图已保存 → {fig_path}")

    # ── 保存 JSON ───────────────────────────────────────────────────────────
    comparison = {
        "note": "四份结果均为同一批验证样本（seed=42 抽样），顺序一致，可直接做配对比较",
        "total_samples": total,
        "methods": [{"key": m["key"], "label": m["label"], "desc": m["desc"],
                     "source_file": m["path"].name,
                     **{k: v for k, v in metrics[m["key"]].items()
                        if k not in ("correct_flags",)}}
                    for m in METHODS],
        "pairwise_tests": tests,
    }
    json_path = RESULTS_DIR / "comparison_4methods_zeroshot_fewshot_sft_bert.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)
    print(f"对比数据已保存 → {json_path}")

    # ── 保存 Markdown 报告 ──────────────────────────────────────────────────
    md = build_markdown(metrics, tests, support, class_names, total)
    md_path = RESULTS_DIR / "comparison_4methods_zeroshot_fewshot_sft_bert.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"对比报告已保存 → {md_path}")


def build_markdown(metrics, tests, support, class_names, total):
    """生成给人看的 Markdown 对比报告。"""
    L = []
    L.append("# 四种分类方案对比：zero-shot / few-shot / SFT(LoRA) / BERT 微调\n")
    L.append(f"评测样本：验证集随机采样 **{total} 条**（seed=42）。"
             "四份结果来自同一批样本、同一顺序，因此可以逐条配对比较。\n")
    L.append("| 编号 | 方法 | 类型 | 做法 | 结果文件 |")
    L.append("|---|---|---|---|---|")
    kinds = {"zero-shot": "生成式", "few-shot": "生成式",
             "sft": "生成式", "bert": "判别式"}
    for i, m in enumerate(METHODS, 1):
        L.append(f"| {i} | {m['label']} | {kinds[m['key']]} | {m['desc']} "
                 f"| `{m['path'].name}` |")
    L.append("")

    # 整体
    L.append("## 一、整体指标\n")
    L.append("| 指标 | " + " | ".join(m["label"] for m in METHODS) + " |")
    L.append("|---|" + "---|" * len(METHODS))
    for name, key in [("Accuracy", "accuracy"), ("Macro F1", "macro_f1"),
                      ("Macro F1（剔除样本<8 的类）", "macro_f1_big"),
                      ("Weighted F1", "weighted_f1"), ("可解析率", "parse_rate"),
                      ("预测分布 L1 偏差", "dist_l1")]:
        vals = {m["key"]: metrics[m["key"]][key] for m in METHODS}
        best = max(vals, key=vals.get) if key != "dist_l1" else min(vals, key=vals.get)
        cells = [f"**{vals[m['key']]:.4f}**" if m["key"] == best else f"{vals[m['key']]:.4f}"
                 for m in METHODS]
        L.append(f"| {name} | " + " | ".join(cells) + " |")
    L.append("| 无法解析条数 | " + " | ".join(
        str(metrics[m["key"]]["unparseable"]) for m in METHODS) + " |")
    L.append("\n加粗为该项最优。两处需要解释：\n")
    L.append("- BERT 的「可解析率」恒为 1.0000 —— argmax 一定落在 15 类里，"
             "这是判别式模型的结构性优势，不是它更会「遵守格式」。")
    L.append("- 「Macro F1（剔除样本<8 的类）」剔掉了 故事（2 条）和 证券（1 条）。"
             "这两类的 F1 在 0 和 1 之间跳，**一条样本就能让 15 类 macro F1 变动 1/15≈0.067**，"
             "所以这一行比上面的 Macro F1 更可信。\n")

    # 逐类
    L.append("## 二、逐类 F1\n")
    L.append("| 类别 | 样本数 | " + " | ".join(m["label"] for m in METHODS) + " | 最强 |")
    L.append("|---|---|" + "---|" * len(METHODS) + "---|")
    # 自己数一遍，别复用 main 里那个（那个已经被终端表格加过了）
    win_count = {m["key"]: 0 for m in METHODS}
    for i, name in enumerate(class_names):
        vals = {m["key"]: metrics[m["key"]]["per_class"][name]["f1"] for m in METHODS}
        best = max(vals, key=vals.get)
        has_winner = vals[best] > 0
        if has_winner:
            win_count[best] += 1
        cells = [f"**{vals[m['key']]:.3f}**" if (has_winner and m["key"] == best)
                 else f"{vals[m['key']]:.3f}" for m in METHODS]
        # 三个方法都没答对（比如只有 1 条样本的证券），标"最强"没有意义
        best_label = next((m["label"] for m in METHODS if m["key"] == best), "—") \
            if has_winner else "—"
        L.append(f"| {name} | {support[i]} | " + " | ".join(cells) + f" | {best_label} |")
    L.append("\n各类夺冠次数（并列取第一个）：" + "，".join(
        f"{m['label']} **{win_count[m['key']]}** 类" for m in METHODS) + "\n")

    # 显著性
    L.append("## 三、配对显著性检验（McNemar 精确检验）\n")
    L.append("| 对比 | 仅前者对 | 仅后者对 | 净变化 | p 值 | 结论 |")
    L.append("|---|---|---|---|---|---|")
    for t in tests:
        verdict = "显著" if t["p_value"] < 0.05 else "不显著"
        L.append(f"| {t['a']} → {t['b']} | {t['only_a']} | {t['only_b']} | "
                 f"{t['net']:+d} | {fmt_p(t['p_value'])} | {verdict} |")
    L.append("\n「仅前者对/仅后者对」= 只在一种方法里答对的条数，两者之差就是净变化。"
             "p 值由二项分布精确计算，不依赖大样本近似。\n")

    # 误差方向
    L.append("## 四、主要误差方向（真实 → 预测）\n")
    L.append("| 方法 | Top-5 混淆 |")
    L.append("|---|---|")
    for m in METHODS:
        conf = metrics[m["key"]]["top_confusion"]
        s = "、".join(f"{c['true']}→{c['pred']}({c['count']})" for c in conf)
        L.append(f"| {m['label']} | {s} |")
    L.append("")

    # 预测分布
    L.append("## 五、预测类别分布\n")
    L.append("看模型是不是只在少数几个类上打转。真实分布作为参照。\n")
    L.append("| 类别 | 真实条数 | " + " | ".join(m["label"] for m in METHODS) + " |")
    L.append("|---|---|" + "---|" * len(METHODS))
    for name in class_names:
        cells = [str(metrics[m["key"]]["pred_dist"][name]) for m in METHODS]
        L.append(f"| {name} | {metrics['zero-shot']['true_dist'][name]} | "
                 + " | ".join(cells) + " |")
    L.append("")

    # 结论
    zs, fs, sft, bert = (metrics["zero-shot"], metrics["few-shot"],
                         metrics["sft"], metrics["bert"])
    p_fs = next(t["p_value"] for t in tests if t["a"] == "zero-shot" and t["b"] == "few-shot")
    p_sft = next(t["p_value"] for t in tests if t["a"] == "zero-shot" and t["b"] == "sft")
    p_sb = next(t["p_value"] for t in tests if t["a"] == "sft" and t["b"] == "bert")

    L.append("## 六、结论\n")
    L.append(f"1. **few-shot 基本没用**：准确率 {zs['accuracy']:.4f} → {fs['accuracy']:.4f}"
             f"（{fs['accuracy'] - zs['accuracy']:+.4f}），{fmt_p_eq(p_fs)}，不显著。"
             f"它只是把答案在类别间重新分配了一遍，不是真的学会了判别。")
    L.append(f"2. **SFT 是真提升**：{zs['accuracy']:.4f} → {sft['accuracy']:.4f}"
             f"（{sft['accuracy'] - zs['accuracy']:+.4f}），{fmt_p_eq(p_sft)}。"
             f"Macro F1 从 {zs['macro_f1']:.4f} 到 {sft['macro_f1']:.4f}。")
    L.append(f"3. **格式问题只有微调能解决**：无法解析 {zs['unparseable']} 条 → "
             f"{fs['unparseable']} 条（few-shot 一点没改善）→ {sft['unparseable']} 条。"
             f"提示词教不会 0.5B 模型「只能从 15 个词里选」，训练才教得会。"
             f"BERT 更彻底：**结构上不可能答非所问**。")
    L.append(f"4. **类别分布**：预测分布 L1 偏差 {zs['dist_l1']:.3f} → {fs['dist_l1']:.3f} → "
             f"{sft['dist_l1']:.3f} → {bert['dist_l1']:.3f}，越低说明越不会只盯着热门类回答。")
    L.append(f"5. **BERT 在这 200 条上更强，但没有显著拉开差距**：准确率 {bert['accuracy']:.4f} vs "
             f"SFT {sft['accuracy']:.4f}（{bert['accuracy'] - sft['accuracy']:+.4f}），"
             f"{fmt_p_eq(p_sb)}，**不显著**；逐类 F1 上 BERT 拿下 {win_count['bert']} 类。")
    L.append(f"   真正拉开的是小类公平性：macro F1 {bert['macro_f1']:.4f} vs "
             f"{sft['macro_f1']:.4f}（{bert['macro_f1'] - sft['macro_f1']:+.4f}）。"
             f"但这个差距有一半是假的 —— 剔除只有 1~2 条样本的 故事/证券 后，"
             f"BERT {bert['macro_f1_big']:.4f} vs SFT {sft['macro_f1_big']:.4f}，"
             f"只差 {bert['macro_f1_big'] - sft['macro_f1_big']:+.4f}。")

    L.append("\n### 关于 BERT 的两个准确率，别混着用\n")
    L.append(f"- 本报告的 **{bert['accuracy']:.4f}**：在这 200 条上和另外三个方法逐条对打，"
             f"是**配对比较**，有效。")
    L.append(f"- 训练日志里的 **0.5637**：BERT 在完整 10000 条验证集上的成绩。")
    L.append(f"- 两者差 {bert['accuracy'] - 0.5637:+.4f}，说明这 200 条样本对 BERT 略偏容易"
             f"（n=200 时准确率的标准误约 ±0.035，这个偏差在噪声范围内）。")
    L.append("- 所以：**跨报告比较绝对数字没有意义，只有同一批样本上的配对比较才有效。**\n")

    L.append("### 一句话总结\n")
    L.append(f"> 提示词工程（few-shot）在这个任务上几乎无效（{fs['accuracy'] - zs['accuracy']:+.4f}，不显著）；"
             f"微调才是分水岭（{sft['accuracy'] - zs['accuracy']:+.4f}）。"
             f"0.5B 的 LoRA 微调追到了 BERT 身后 {bert['accuracy'] - sft['accuracy']:.4f}，"
             f"配对检验还分不出胜负（{fmt_p_eq(p_sb)}）；"
             f"BERT 的优势在小类公平性和「不会答非所问」，生成式的优势在于同一个模型还能做别的任务。\n")
    return "\n".join(L)


if __name__ == "__main__":
    main()
