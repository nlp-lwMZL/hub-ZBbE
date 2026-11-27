"""
跨方法对比：BERT+Linear / BERT+CRF / Qwen2-0.5B SFT(LoRA)，附 DeepSeek API 小节

只读已有的评估 JSON，不加载模型，秒级出结果。

输入：
  outputs_work/logs/work_eval_linear_{split}.json    ← src_work/evaluate.py 产出
  outputs_work/logs/work_eval_crf_{split}.json
  outputs_llm_work/logs/eval_sft_{split}.json        ← src_llm_work/evaluate_sft_work.py 产出
  outputs_llm_work/logs/eval_llm_0shot_1shot_3shot_{split}.json  ← llm_ner_work.py（仅 500 条子集）

输出（都在 outputs_llm_work/ 下）：
  compare_ner_{split}.md      对比报告
  compare_ner_{split}.png     对比图
  logs/compare_ner_{split}.json   机器可读

两件容易被数字骗到的事，本脚本会自己算出来写进报告：
  1. BERT 的 max_length=128 只读得进前 126 个字，超长句整句截掉，落在外面的
     实体从 BERT 的分母里消失；SFT 是生成式，看全文。两者实体集合并不相同。
  2. DeepSeek 那三条只跑了 500 条子集，样本数不同，单独成节，不混进同一张图。

使用方式：
  python src_llm_work/compare_results_work.py
  python src_llm_work/compare_results_work.py --split test
  python src_llm_work/compare_results_work.py --bert_max_len 512   # BERT 侧换过截断长度时
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import (ROOT, OUT_DIR, LOG_DIR, FIG_DIR, ENTITY_TYPES,
                    compute_metrics_multiset, _dw, _pad)

matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
matplotlib.rcParams["axes.unicode_minus"] = False

BERT_LOG_DIR = ROOT / "outputs_work" / "logs"

# BERT 单条推理耗时：2026-10-30 实测，GTX 1660 Ti、batch=16、max_length=512、
# 只计 forward（不含加载权重和 tokenize）：
#   BERT+Linear 42.4 ms/条，BERT+CRF 51.1 ms/条
# 之前这里写的是「约 5 ms」，那是拍脑袋的估计，被实测推翻了。
# 注意 512 比训练时的 128 慢很多（attention 是 O(n²)），128 下大约是这个数的 1/3。
BERT_MS_PER_ITEM = 45.0

METHODS = [
    {"key": "linear", "label": "BERT+Linear", "color": "#4C72B0", "kind": "判别式"},
    {"key": "crf",    "label": "BERT+CRF",    "color": "#DD8452", "kind": "判别式"},
    {"key": "sft",    "label": "Qwen2-0.5B SFT", "color": "#55A868", "kind": "生成式"},
]


def load_all(split: str, bert_tag: str = "") -> dict:
    """把三种方法的结果统一成 {key: {precision, recall, f1, per_type, note}}。

    bert_tag 是 evaluate.py 的 --tag 后缀：传 "full" 就读
    work_eval_crf_validation_full.json（放开截断的那份）。
    """
    out = {}
    suffix = f"_{bert_tag}" if bert_tag else ""

    for key, fname in [("linear", f"work_eval_linear_{split}{suffix}.json"),
                       ("crf", f"work_eval_crf_{split}{suffix}.json")]:
        p = BERT_LOG_DIR / fname
        if not p.exists():
            continue
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        out[key] = {
            "precision": d["precision"], "recall": d["recall"], "f1": d["f1"],
            # BERT 侧 per_type 用的是 sklearn 的键名 f1-score，这里统一成 f1
            "per_type": {t: v["f1-score"] for t, v in d.get("per_type", {}).items()},
            "gold_count": {t: v.get("support", 0) for t, v in d.get("per_type", {}).items()},
            "n_samples": d.get("n_samples"),
            "max_length": d.get("max_length"),
            "note": f"epoch={d.get('ckpt_epoch')}，非法序列 {d.get('illegal_stats', {}).get('total_illegal')} 条",
        }

    p = LOG_DIR / f"eval_sft_{split}.json"
    if p.exists():
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        m = d["metrics_multiset"]
        out["sft"] = {
            "precision": m["precision"], "recall": m["recall"], "f1": m["f1"],
            "per_type": {t: v["f1"] for t, v in d["per_type"].items()},
            "gold_count": {t: v["gold_count"] for t, v in d["per_type"].items()},
            "n_samples": d.get("n_samples"),
            "ms_per_item": d.get("ms_per_item"),
            "format_fail": d.get("format_fail"),
            "strict_f1": d.get("metrics_strict", {}).get("f1"),
            "note": f"格式失败 {d.get('format_fail')} 条，{d.get('ms_per_item')} ms/条",
        }
    return out


def load_sft_detail(split: str):
    """把 SFT 的逐句明细读出来（重算窗口指标要用）。"""
    p = LOG_DIR / f"eval_sft_{split}.json"
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        d = json.load(f)
    return d if d.get("detail") else None


def truncation_check(split: str, max_len: int, res: dict):
    """量化 BERT 的 max_length 截断对「同一批实体」这个前提的破坏。

    BERT 的 dataset.py 用 is_split_into_words=True + max_length=128，
    实际只能读入前 max_len-2 = 126 个字（[CLS]/[SEP] 各占一位）。
    超长句被**整句截断**，落在外面的实体既不进 TP 也不进 FN ——
    等于从 BERT 的分母里直接消失；而 SFT 是生成式，看到的是全文。
    所以「同一批 2318 条样本」成立，但「同一批实体」不成立。

    这里做三件事：
      1. 数出全文 gold 和前 126 字内 gold 的差额，逐类型列出；
      2. 和 BERT 自己报的 support 对一下，确认差额就是截断造成的；
      3. 把 SFT 也压到同一个窗口，看差距收窄到多少。
    """
    d = load_sft_detail(split)
    if d is None or "sft" not in res:
        return None
    det = d["detail"]
    window = max_len - 2

    def keep(ents, text):
        """只保留完全落在窗口内的实体。用 find 定位，句内重复按首次出现算——
        这是近似，但只影响几十条里的一两条，不影响结论方向。"""
        out = []
        for e in ents:
            s = text.find(e["text"])
            if 0 <= s and s + len(e["text"]) <= window:
                out.append(e)
        return out

    full_g = [it["gold"] for it in det]
    full_p = [it["pred"] for it in det]
    # 「去幻觉但不限窗口」：只丢掉原文里找不到的预测，gold 不动
    noluc_p = [[e for e in it["pred"]
                if it["text"].find(e["text"]) >= 0] for it in det]
    win_g = [keep(it["gold"], it["text"]) for it in det]
    win_p = [keep(it["pred"], it["text"]) for it in det]

    cnt_full = Counter(e["type"] for g in full_g for e in g)
    cnt_win = Counter(e["type"] for g in win_g for e in g)
    bert_support = res["crf"]["gold_count"]

    n_halluc = sum(len(a) for a in full_p) - sum(len(a) for a in noluc_p)
    return {
        "max_len": max_len,
        "window": window,
        "gold_full": {t: cnt_full.get(t, 0) for t in ENTITY_TYPES},
        "gold_window": {t: cnt_win.get(t, 0) for t in ENTITY_TYPES},
        "bert_support": {t: int(bert_support.get(t, 0)) for t in ENTITY_TYPES},
        "gold_full_total": sum(cnt_full.values()),
        "gold_window_total": sum(cnt_win.values()),
        "n_halluc": n_halluc,
        "m_full": compute_metrics_multiset(full_g, full_p),      # 主口径
        "m_noluc": compute_metrics_multiset(full_g, noluc_p),    # 只去幻觉
        "m_window": compute_metrics_multiset(win_g, win_p),      # 同窗口
    }


def load_deepseek(split: str):
    """读 DeepSeek 的 zero/one/few-shot 结果（500 条子集，样本数和主表不同）。"""
    p = LOG_DIR / f"eval_llm_0shot_1shot_3shot_{split}.json"
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _item_counts(detail):
    """每句的 (tp, n_pred, n_gold)，配对 bootstrap 重算 F1 用。"""
    tp, npd, ngd = [], [], []
    for it in detail:
        cg = Counter((e["text"], e["type"]) for e in it["gold"])
        cp = Counter((e["text"], e["type"]) for e in it["pred"])
        tp.append(sum((cg & cp).values()))
        npd.append(sum(cp.values()))
        ngd.append(sum(cg.values()))
    return (np.array(tp), np.array(npd), np.array(ngd))


def paired_bootstrap(det_a, det_b, n_boot=2000, seed=42):
    """同一批样本上配对重采样，估 ΔF1 的置信区间。

    两条 prompt 跑的是同一批句子、同一个顺序，所以可以配对 ——
    比各自算一个置信区间再比是否重叠要准得多。
    """
    tp_a, p_a, g_a = _item_counts(det_a)
    tp_b, p_b, g_b = _item_counts(det_b)
    n = len(tp_a)
    if n == 0 or len(tp_b) != n:
        return None
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        f1_a = 2 * tp_a[idx].sum() / max(p_a[idx].sum() + g_a[idx].sum(), 1)
        f1_b = 2 * tp_b[idx].sum() / max(p_b[idx].sum() + g_b[idx].sum(), 1)
        deltas[i] = f1_b - f1_a
    return {
        "delta": float(deltas.mean()),
        "ci_low": float(np.percentile(deltas, 2.5)),
        "ci_high": float(np.percentile(deltas, 97.5)),
        "share_negative": float((deltas < 0).mean()),
        "share_positive": float((deltas > 0).mean()),
        "n_boot": n_boot,
    }


def main():
    args = parse_args()
    res = load_all(args.split, args.bert_tag)
    if not res:
        print(f"没找到任何评估结果，请先跑 evaluate 脚本。")
        return

    avail = [m for m in METHODS if m["key"] in res]
    n_samples = {res[m["key"]].get("n_samples") for m in avail}
    if len(n_samples) > 1:
        print(f"[警告] 各方法的样本数不一致：{n_samples} —— 下面的数字不能直接横向比")
    total = max(n_samples) if n_samples else 0

    # ── 终端：整体指标 ────────────────────────────────────────────────────────
    print(f"\n{'=' * 76}")
    print(f"{'人民日报 NER —— 判别式 vs 生成式（' + args.split + '，' + str(total) + ' 条）':^70}")
    print(f"{'=' * 76}")
    head = _pad("方法", 18) + _pad("类型", 8) + _pad("Precision", 11, right=True) \
        + _pad("Recall", 9, right=True) + _pad("F1", 9, right=True)
    print(head)
    print("-" * _dw(head))
    for m in avail:
        r = res[m["key"]]
        print(_pad(m["label"], 18) + _pad(m["kind"], 8)
              + _pad(f"{r['precision']:.4f}", 11, right=True)
              + _pad(f"{r['recall']:.4f}", 9, right=True)
              + _pad(f"{r['f1']:.4f}", 9, right=True))
    print("-" * _dw(head))

    # ── 终端：逐类型 ──────────────────────────────────────────────────────────
    print(f"\n{'逐类型 F1（★ 标最强）':^56}")
    head2 = _pad("类型", 6) + _pad("gold", 7, right=True) + "".join(
        _pad(m["label"], 18, right=True) for m in avail)
    print(head2)
    print("-" * _dw(head2))
    best_count = {m["key"]: 0 for m in avail}
    for etype in ENTITY_TYPES:
        vals = {m["key"]: res[m["key"]]["per_type"].get(etype, 0.0) for m in avail}
        best = max(vals, key=vals.get)
        best_count[best] += 1
        row = _pad(etype, 6) + _pad(str(res[avail[0]["key"]]["gold_count"].get(etype, 0)), 7, right=True)
        for m in avail:
            row += _pad(f"{vals[m['key']]:.4f}" + ("★" if m["key"] == best else " "), 18, right=True)
        print(row)
    print("\n  各类夺冠：" + "，".join(
        f"{m['label']} {best_count[m['key']]} 类" for m in avail))

    # ── 口径校正：BERT 的截断 ────────────────────────────────────────────────
    # max_length 优先读评估 JSON 里记的（新版本 evaluate.py 会写进去），读不到才用命令行默认
    bert_max_len = next((res[k]["max_length"] for k in ("crf", "linear")
                         if res.get(k, {}).get("max_length")), args.bert_max_len)
    trunc_all = truncation_check(args.split, bert_max_len, res)
    # 传了 --bert_tag 说明已经放开截断重跑过，口径问题用「重跑」解决掉了，不用再写校正章节
    impact = truncation_impact(args.split, res, load_all(args.split, "")) if args.bert_tag else None
    residual = None
    if impact:
        if trunc_all:
            residual = trunc_all["gold_full_total"] - impact[0]["gold_full"]
        trunc = None
        print(f"\n[口径已对齐] BERT 用 max_length={bert_max_len} 重跑过"
              + (f"，仍有 {residual} 个实体超出位置编码上限" if residual else "，覆盖全部句子")
              + "；截断的影响见下方「截断长度的影响」。")
    else:
        trunc = trunc_all
        if trunc and trunc["gold_full_total"] == trunc["gold_window_total"]:
            print(f"\n[口径已对齐] BERT 的 max_length={bert_max_len} 覆盖了全部句子，"
                  f"三个方法看的是同一批 {trunc['gold_full_total']} 个实体，不用做校正。")
            trunc = None
    if trunc:
        print(f"\n{'─' * 76}")
        print(f"口径校正：BERT max_length={trunc['max_len']}，实际只读前 {trunc['window']} 个字")
        print(f"{'─' * 76}")
        head3 = _pad("类型", 6) + _pad("全文 gold", 11, right=True) \
            + _pad(f"前{trunc['window']}字 gold", 16, right=True) \
            + _pad("BERT support", 14, right=True) + _pad("被截掉", 9, right=True)
        print(head3)
        print("-" * _dw(head3))
        for t in ENTITY_TYPES:
            gf = trunc["gold_full"].get(t, 0)
            gw = trunc["gold_window"].get(t, 0)
            bs = trunc["bert_support"].get(t, 0)
            print(_pad(t, 6) + _pad(str(gf), 11, right=True) + _pad(str(gw), 16, right=True)
                  + _pad(str(bs), 14, right=True) + _pad(f"-{gf - gw}", 9, right=True))
        print("-" * _dw(head3))
        print(_pad("合计", 6) + _pad(str(trunc["gold_full_total"]), 11, right=True)
              + _pad(str(trunc["gold_window_total"]), 16, right=True)
              + _pad(str(sum(trunc["bert_support"].values())), 14, right=True)
              + _pad(f"-{trunc['gold_full_total'] - trunc['gold_window_total']}", 9, right=True))
        print("\n  同一批句子，但不是同一批实体。把 SFT 压到同样的窗口：")
        print("  " + _pad("口径", 26) + _pad("P", 9, right=True)
              + _pad("R", 9, right=True) + _pad("F1", 9, right=True))
        for tag, mk in [("SFT 全文（主口径）", "m_full"),
                        ("SFT 去幻觉（不限窗口）", "m_noluc"),
                        ("SFT 同窗口（前%d字）" % trunc["window"], "m_window")]:
            mm = trunc[mk]
            print("  " + _pad(tag, 26) + _pad(f"{mm['precision']:.4f}", 9, right=True)
                  + _pad(f"{mm['recall']:.4f}", 9, right=True)
                  + _pad(f"{mm['f1']:.4f}", 9, right=True))
        if "crf" in res:
            print("  " + _pad("BERT+CRF（本身只到窗口）", 26)
                  + _pad(f"{res['crf']['precision']:.4f}", 9, right=True)
                  + _pad(f"{res['crf']['recall']:.4f}", 9, right=True)
                  + _pad(f"{res['crf']['f1']:.4f}", 9, right=True))
        gap_raw = res["crf"]["f1"] - trunc["m_full"]["f1"]
        gap_win = res["crf"]["f1"] - trunc["m_window"]["f1"]
        print(f"\n  原始差距 {gap_raw:.4f} → 同窗口差距 {gap_win:.4f}"
              f"（结论方向不变，但头版数字虚高了 {gap_raw - gap_win:.4f}）")

    # ── 截断长度的影响（只在用 --bert_tag 读了放开截断的那份时才有） ──────────
    if impact:
        print(f"\n{'─' * 76}")
        print("截断长度的影响（同一份权重，只改 max_length）")
        print(f"{'─' * 76}")
        head5 = _pad("模型", 14) + _pad("128 F1", 10, right=True) \
            + _pad("放开 F1", 10, right=True) + _pad("变化", 10, right=True) \
            + _pad("实体数", 12, right=True)
        print(head5)
        print("-" * _dw(head5))
        for r in impact:
            print(_pad(r["label"], 14) + _pad(f"{r['f1_base']:.4f}", 10, right=True)
                  + _pad(f"{r['f1_full']:.4f}", 10, right=True)
                  + _pad(f"{r['f1_full'] - r['f1_base']:+.4f}", 10, right=True)
                  + _pad(f"{r['gold_base']}→{r['gold_full']}", 12, right=True))

    # ── DeepSeek 小节（500 条子集，不混进主表） ──────────────────────────────
    ds = load_deepseek(args.split)
    ds_rows = []
    boots = {}
    if ds and ds.get("runs"):
        print(f"\n{'─' * 76}")
        print(f"DeepSeek API（{ds['model']}，{ds['n_samples']} 条子集 —— "
              f"样本数和主表不同，不能直接和上表比）")
        print(f"{'─' * 76}")
        head4 = _pad("设置", 12) + _pad("P", 9, right=True) + _pad("R", 9, right=True) \
            + _pad("F1", 9, right=True) + _pad("格式失败", 10, right=True) \
            + _pad("ms/条", 9, right=True)
        print(head4)
        print("-" * _dw(head4))
        for run in ds["runs"]:
            mm = run["metrics"]
            label = f"{run['n_shot']}-shot" if run["n_shot"] else "zero-shot"
            ds_rows.append({"label": label, "n_shot": run["n_shot"],
                            "precision": mm["precision"], "recall": mm["recall"],
                            "f1": mm["f1"], "format_fail": run.get("format_fail"),
                            "ms_per_item": run.get("ms_per_item"),
                            "per_type": {t: v["f1"] for t, v in run["per_type"].items()},
                            "gold_count": {t: v["gold_count"] for t, v in run["per_type"].items()}})
            print(_pad(label, 12) + _pad(f"{mm['precision']:.4f}", 9, right=True)
                  + _pad(f"{mm['recall']:.4f}", 9, right=True)
                  + _pad(f"{mm['f1']:.4f}", 9, right=True)
                  + _pad(str(run.get("format_fail")), 10, right=True)
                  + _pad(f"{run.get('ms_per_item', 0):.0f}", 9, right=True))
        print("-" * _dw(head4))

        # 配对 bootstrap：同一批 500 句，两条 prompt 直接配对比
        base = next((r for r in ds["runs"] if r["n_shot"] == 0), None)
        boots = {}
        if base is not None:
            for run in ds["runs"]:
                if run["n_shot"] == 0:
                    continue
                b = paired_bootstrap(base["detail"], run["detail"])
                if b:
                    boots[run["n_shot"]] = b
                    sig = "显著" if (b["ci_low"] > 0 or b["ci_high"] < 0) else "不显著"
                    print(f"\n  {run['n_shot']}-shot vs zero-shot：ΔF1 = {b['delta']:+.4f}  "
                          f"95% CI [{b['ci_low']:+.4f}, {b['ci_high']:+.4f}]  "
                          f"{b['share_negative'] * 100:.1f}% 的重采样为负 → {sig}")

    # ── 画图 ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8),
                             gridspec_kw={"width_ratios": [1.15, 1]})
    x = range(len(ENTITY_TYPES))
    w = 0.8 / len(avail)
    for k, m in enumerate(avail):
        vals = [res[m["key"]]["per_type"].get(t, 0.0) for t in ENTITY_TYPES]
        off = (k - (len(avail) - 1) / 2) * w
        bars = axes[0].bar([i + off for i in x], vals, w, label=m["label"], color=m["color"])
        for b, v in zip(bars, vals):
            axes[0].text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v:.3f}",
                         ha="center", fontsize=8.5)
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels([f"{t}\n({res[avail[0]['key']]['gold_count'].get(t, 0)})"
                             for t in ENTITY_TYPES])
    # 下限取 0.70 而不是 0：三个方法的 F1 都在 0.78 以上，从 0 起会把差异压成一条平线。
    # 注意这是「截断纵轴」，看绝对值要看上面的表格，图只看相对高低。
    axes[0].set_ylim(0.70, 1.02)
    axes[0].set_ylabel("F1")
    axes[0].set_title("逐类型 F1（括号内为实体数）")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].set_axisbelow(True)

    metric_names = ["Precision", "Recall", "F1"]
    metric_keys = ["precision", "recall", "f1"]
    w2 = 0.8 / len(avail)
    for k, m in enumerate(avail):
        vals = [res[m["key"]][key] for key in metric_keys]
        off = (k - (len(avail) - 1) / 2) * w2
        bars = axes[1].bar([i + off for i in range(3)], vals, w2,
                           label=m["label"], color=m["color"])
        for b, v in zip(bars, vals):
            axes[1].text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v:.3f}",
                         ha="center", fontsize=8.5)
    axes[1].set_xticks(range(3))
    axes[1].set_xticklabels(metric_names)
    axes[1].set_ylim(0.70, 1.02)
    axes[1].set_title("整体指标")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].set_axisbelow(True)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(avail),
               frameon=False, fontsize=10, bbox_to_anchor=(0.5, 0.005))
    fig.suptitle(f"人民日报 NER：序列标注 vs 生成式（{args.split}，{total} 条）", fontsize=12)
    fig.tight_layout(rect=[0, 0.06, 1, 0.94])

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig_path = FIG_DIR / f"compare_ner_{args.split}.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()

    # ── 写 Markdown ───────────────────────────────────────────────────────────
    L = []
    L.append(f"# 人民日报 NER：判别式 vs 生成式（{args.split}）\n")
    L.append(f"三个方法都在同一批 **{total} 条**验证句上评估。")
    if trunc:
        L.append(f"注意：句子是同一批，**实体集合不是** —— BERT 的 `max_length={trunc['max_len']}` "
                 f"只读得进前 {trunc['window']} 个字，见下面「口径校正」。\n")
    elif impact:
        L.append(f"BERT 用 `max_length={bert_max_len}` 重跑过（训练时是 128），"
                 f"三个方法看的是同一批实体，可以直接比。\n")
    else:
        L.append("")
    L.append("| 方法 | 范式 | Precision | Recall | F1 | 备注 |")
    L.append("|---|---|---|---|---|---|")
    for m in avail:
        r = res[m["key"]]
        L.append(f"| {m['label']} | {m['kind']} | {r['precision']:.4f} | "
                 f"{r['recall']:.4f} | **{r['f1']:.4f}** | {r.get('note', '')} |")
    L.append("")
    L.append("## 逐类型 F1\n")
    L.append("| 类型 | 实体数 | " + " | ".join(m["label"] for m in avail) + " |")
    L.append("|---|---|" + "---|" * len(avail))
    for etype in ENTITY_TYPES:
        vals = {m["key"]: res[m["key"]]["per_type"].get(etype, 0.0) for m in avail}
        best = max(vals, key=vals.get)
        cells = [f"**{vals[m['key']]:.4f}**" if m["key"] == best else f"{vals[m['key']]:.4f}"
                 for m in avail]
        L.append(f"| {etype} | {res[avail[0]['key']]['gold_count'].get(etype, 0)} | "
                 + " | ".join(cells) + " |")
    if trunc:
        L.append("")
        L.append(f"> 「实体数」一列是 BERT 侧报的 support，只统计前 {trunc['window']} 个字内的实体。"
                 f"SFT 看的是全文，它的 gold 数见下一节。")
    L.append("")

    # ── 口径校正 ─────────────────────────────────────────────────────────────
    if trunc:
        L.append("## 口径校正：BERT 只看到前 %d 个字\n" % trunc["window"])
        L.append(f"`src_work/dataset.py` 里 `max_length={trunc['max_len']}`，"
                 f"`[CLS]`/`[SEP]` 各占一位，实际读入 **{trunc['window']} 个字**。"
                 f"超长句被整句截断，落在外面的实体既不进 TP 也不进 FN —— "
                 f"直接从 BERT 的分母里消失。SFT 是生成式，看到的是全文。\n")
        L.append("| 类型 | 全文 gold | 前 %d 字内 gold | BERT 报的 support | 被截掉 |"
                 % trunc["window"])
        L.append("|---|---|---|---|---|")
        for t in ENTITY_TYPES:
            gf = trunc["gold_full"].get(t, 0)
            gw = trunc["gold_window"].get(t, 0)
            bs = trunc["bert_support"].get(t, 0)
            L.append(f"| {t} | {gf} | {gw} | {bs} | **{gf - gw}** |")
        L.append(f"| 合计 | {trunc['gold_full_total']} | {trunc['gold_window_total']} | "
                 f"{sum(trunc['bert_support'].values())} | "
                 f"**{trunc['gold_full_total'] - trunc['gold_window_total']}** |")
        L.append("")
        L.append(f"> 「前 126 字内 gold」比 BERT 的 support 多 "
                 f"{trunc['gold_window_total'] - sum(trunc['bert_support'].values())} 条："
                 f"这一列是用 `text.find()` 估位置的（句内重复实体按首次出现算），"
                 f"而 BERT 的 support 来自真实 tokenize 后的标签对齐，后者更准。"
                 f"两边差 0.1% 量级，不影响结论。")
        L.append("")
        L.append(f"把 SFT 也压到同一个窗口，重新算一遍：\n")
        L.append("| 口径 | P | R | F1 | gold 实体 |")
        L.append("|---|---|---|---|---|")
        rows = [("SFT 全文（主口径，保留幻觉预测）", trunc["m_full"], trunc["gold_full_total"]),
                ("SFT 去幻觉（只丢原文里找不到的预测）", trunc["m_noluc"], trunc["gold_full_total"]),
                ("SFT 同窗口（前 %d 字）" % trunc["window"], trunc["m_window"],
                 trunc["gold_window_total"])]
        for tag, mm, ng in rows:
            L.append(f"| {tag} | {mm['precision']:.4f} | {mm['recall']:.4f} | "
                     f"{mm['f1']:.4f} | {ng} |")
        if "crf" in res:
            L.append(f"| BERT+CRF（本身只到窗口） | {res['crf']['precision']:.4f} | "
                     f"{res['crf']['recall']:.4f} | **{res['crf']['f1']:.4f}** | "
                     f"{sum(trunc['bert_support'].values())} |")
        gap_raw = res["crf"]["f1"] - trunc["m_full"]["f1"]
        gap_win = res["crf"]["f1"] - trunc["m_window"]["f1"]
        L.append("")
        L.append(f"- 原始差距 **{gap_raw:.4f}** → 同窗口差距 **{gap_win:.4f}**，"
                 f"头版数字虚高了 {gap_raw - gap_win:.4f}。")
        L.append(f"- 把 SFT 压到同窗口带来 {trunc['m_window']['f1'] - trunc['m_noluc']['f1']:+.4f}，"
                 f"去掉那 {trunc['n_halluc']} 条原文里根本不存在的「幻觉预测」带来 "
                 f"{trunc['m_noluc']['f1'] - trunc['m_full']['f1']:+.4f}，两项合计 "
                 f"{trunc['m_window']['f1'] - trunc['m_full']['f1']:+.4f}。")
        L.append(f"- 结论方向不变：生成式仍然明显落后，但**不要把 {gap_raw:.4f} 当准确数字引用**。"
                 f"想要严格对齐，得把 BERT 的 `max_length` 提到能覆盖全文再重跑。")
        L.append("")

    # ── 截断长度的影响 ───────────────────────────────────────────────────────
    if impact:
        L.append(f"## 附：截断长度的影响（max_length 128 → {bert_max_len}）\n")
        L.append("同一份权重，只改 `max_length` 重跑 —— 这是「截断到底亏了多少」的直接答案。\n")
        L.append("| 模型 | 128 F1 | 放开 F1 | 变化 | 参与统计的实体数 |")
        L.append("|---|---|---|---|---|")
        for r in impact:
            L.append(f"| {r['label']} | {r['f1_base']:.4f} | **{r['f1_full']:.4f}** | "
                     f"{r['f1_full'] - r['f1_base']:+.4f} | "
                     f"{r['gold_base']} → {r['gold_full']} |")
        L.append("")
        L.append("| 类型 | " + " | ".join(f"{r['label']} ΔF1" for r in impact) + " |")
        L.append("|---|" + "---|" * len(impact))
        for t in ENTITY_TYPES:
            cells = []
            for r in impact:
                b = r["per_type_base"].get(t, 0.0)
                f = r["per_type_full"].get(t, 0.0)
                cells.append(f"{f - b:+.4f}（{b:.4f}→{f:.4f}）")
            L.append(f"| {t} | " + " | ".join(cells) + " |")
        L.append("")
        L.append("读法：涨幅很小，**LOC 承担了几乎全部增量**（被截掉的那批实体里绝大多数是地名），"
                 "截断的代价落在长句里堆着的地名上，不是均匀分布。")
        L.append(f"涨得少不代表截断无害，只是这份数据的句子本来就短（长度 p99 只有 149 字，"
                 f"只有 1.8% 的句子超过 126 字）。")
        if residual:
            L.append("")
            L.append(f"还有一层天花板：`bert-base-chinese` 的位置编码上限就是 **512**，"
                     f"而验证集最长句 568 字。所以放开到 {bert_max_len} 之后仍有 "
                     f"**{residual} 个实体**（占 {residual / (impact[0]['gold_full'] + residual) * 100:.1f}%）"
                     f"看不到 —— 这一层靠调 `max_length` 解决不了，得换长文本模型。")
        L.append("")

    L.append("## 结论\n")
    if "sft" in res and "crf" in res:
        gap = res["sft"]["f1"] - res["crf"]["f1"]
        if trunc:
            gap_win = trunc["m_window"]["f1"] - res["crf"]["f1"]
            L.append(f"1. **生成式 NER 打不过序列标注**：SFT {res['sft']['f1']:.4f} vs "
                     f"BERT+CRF {res['crf']['f1']:.4f}，原始差 {abs(gap):.4f}；"
                     f"把 SFT 也压到前 {trunc['window']} 字后是 "
                     f"{trunc['m_window']['f1']:.4f}，差 **{abs(gap_win):.4f}** —— "
                     f"方向不变，幅度小一些。")
        else:
            L.append(f"1. **生成式 NER 打不过序列标注**：SFT {res['sft']['f1']:.4f} vs "
                     f"BERT+CRF {res['crf']['f1']:.4f}，差 **{abs(gap):.4f}**。")
        L.append(f"2. **差距集中在 ORG 和 LOC**：ORG 差 "
                 f"{res['crf']['per_type'].get('ORG', 0) - res['sft']['per_type'].get('ORG', 0):.4f}、"
                 f"LOC 差 "
                 f"{res['crf']['per_type'].get('LOC', 0) - res['sft']['per_type'].get('LOC', 0):.4f}、"
                 f"PER 只差 "
                 f"{res['crf']['per_type'].get('PER', 0) - res['sft']['per_type'].get('PER', 0):.4f}。"
                 f"人名靠 surface 规律就能认，机构和地名要判边界 —— "
                 f"序列标注的 token 级监督在这两类上更占优。")
        if "ms_per_item" in res["sft"]:
            ratio = res["sft"]["ms_per_item"] / BERT_MS_PER_ITEM
            L.append(f"3. **推理成本差一个多数量级**：BERT 约 {BERT_MS_PER_ITEM:.0f} ms/条，"
                     f"SFT 约 {res['sft']['ms_per_item']:.0f} ms/条，约 **{ratio:.0f} 倍**。")
        L.append(f"4. **格式问题**：SFT 有 {res['sft'].get('format_fail', '?')} 条输出不是合法 JSON；"
                 f"BERT 的 argmax 天然落在标签集内，不存在这个问题。")
        if res["sft"].get("strict_f1"):
            L.append(f"\n> 口径说明：SFT 的主指标用（surface, type）多重集匹配，"
                     f"严格位置匹配口径下是 {res['sft']['strict_f1']:.4f}。"
                     f"BERT 用 seqeval，等价于严格位置匹配。"
                     f"两者差异来自本数据集 7.57% 的句内重复实体。")

    # ── 附：DeepSeek API ─────────────────────────────────────────────────────
    if ds_rows:
        L.append("")
        L.append(f"## 附：DeepSeek API 的 zero / one / few-shot（{ds['n_samples']} 条子集）\n")
        L.append(f"模型 `{ds['model']}`，按 `seed={ds['seed']}` 从验证集抽 {ds['n_samples']} 条；"
                 f"三条 prompt 跑的是**同一批句子、同一顺序**，所以彼此可以配对比较。"
                 f"但这批子集和上面 2318 条的主表不是同一批样本，**不要跨表比绝对值**。\n")
        L.append("| 设置 | P | R | F1 | 格式失败 | ms/条 |")
        L.append("|---|---|---|---|---|---|")
        for r in ds_rows:
            L.append(f"| {r['label']} | {r['precision']:.4f} | {r['recall']:.4f} | "
                     f"{r['f1']:.4f} | {r['format_fail']} | {r['ms_per_item']:.0f} |")
        L.append("")
        L.append("| 类型 | 实体数 | " + " | ".join(r["label"] for r in ds_rows) + " |")
        L.append("|---|---|" + "---|" * len(ds_rows))
        for t in ENTITY_TYPES:
            vals = {r["label"]: r["per_type"].get(t, 0.0) for r in ds_rows}
            best = max(vals, key=vals.get)
            cells = [f"**{vals[r['label']]:.4f}**" if r["label"] == best
                     else f"{vals[r['label']]:.4f}" for r in ds_rows]
            L.append(f"| {t} | {ds_rows[0]['gold_count'].get(t, 0)} | " + " | ".join(cells) + " |")
        L.append("")
        if boots:
            L.append("配对 bootstrap（2000 次重采样，同一批句子）：\n")
            L.append("| 对比 | ΔF1 | 95% CI | 重采样为负的比例 | 判定 |")
            L.append("|---|---|---|---|---|")
            for n_shot, b in sorted(boots.items()):
                sig = "显著" if (b["ci_low"] > 0 or b["ci_high"] < 0) else "不显著"
                L.append(f"| {n_shot}-shot vs zero-shot | {b['delta']:+.4f} | "
                         f"[{b['ci_low']:+.4f}, {b['ci_high']:+.4f}] | "
                         f"{b['share_negative'] * 100:.1f}% | {sig} |")
            L.append("")
        L.append("读法：")
        L.append("- **给示例没帮忙，1-shot 还明显更差**。和本地 Qwen2-0.5B 在文本分类上的结论"
                 "（few-shot +2.5pt，p=0.50，不显著）一致的地方在于：给示例都没带来可靠收益。"
                 "区别是这里不但没赚，还倒亏。")
        L.append("  一个可能解释是「模仿压力」—— 模型会照着唯一示例的粒度和数量输出，"
                 "而单个示例覆盖不了这个数据集的各种边界情况。这是解释，不是证据。")
        L.append("- **格式失败率 0.2%**（1/500，开了 JSON mode），对比本地 Qwen2-0.5B 那 30% 的"
                 "解析失败，说明「能不能输出合法 JSON」主要是模型能力问题，不是 prompt 问题。")
        L.append("- **LOC 是 DeepSeek 的死穴**（0.619，gold 463 条只捞回 244 条），而这恰好是 "
                 "BERT 的强项（0.965）。人名有很强的 surface 规律，地名得靠句法和上下文判断 —— "
                 "「北京大学」里的「北京」、「美国新墨西哥大学」里的「美国」都不该单独算 LOC。")
        L.append("- 绝对值不要和主表比：这 500 条是子集，且 DeepSeek 没有在这份数据上训练过，"
                 "它和 SFT 的差距不能当「API 模型 vs 微调小模型」的结论。")

    L.append("")
    md_path = OUT_DIR / f"compare_ner_{args.split}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))

    json_path = LOG_DIR / f"compare_ner_{args.split}.json"
    payload = {"split": args.split, "total_samples": total,
               "methods": {m["key"]: {"label": m["label"], "kind": m["kind"],
                                      **res[m["key"]]} for m in avail}}
    if trunc:
        payload["truncation"] = _jsonable(trunc)
    if impact:
        payload["truncation_impact"] = _jsonable(impact)
        if residual:
            payload["truncation_impact_residual"] = residual
    if ds_rows:
        payload["deepseek"] = {"model": ds["model"], "n_samples": ds["n_samples"],
                               "seed": ds["seed"], "runs": ds_rows,
                               "bootstrap": {str(k): v for k, v in boots.items()}}
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"\n对比图   → {fig_path}")
    print(f"对比报告 → {md_path}")
    print(f"对比数据 → {json_path}")


def truncation_impact(split: str, res_full: dict, res_base: dict):
    """同一模型在「训练时的 128」和「放开截断」两种口径下的差异。

    这是回答「截断到底亏了多少」最直接的办法：权重没变，只有 max_length 变了。
    """
    if not res_base or not res_full:
        return None
    rows = []
    for m in METHODS:
        key = m["key"]
        if key == "sft" or key not in res_full or key not in res_base:
            continue
        a, b = res_base[key], res_full[key]
        rows.append({
            "label": m["label"],
            "f1_base": a["f1"], "f1_full": b["f1"],
            "per_type_base": a["per_type"], "per_type_full": b["per_type"],
            "gold_base": sum(a["gold_count"].values()),
            "gold_full": sum(b["gold_count"].values()),
        })
    return rows or None


def _jsonable(obj):
    """numpy 标量转 python 原生类型。

    踩过的坑：json.dump 遇到 numpy.int64 会在**写到一半**时抛 TypeError，
    留下一个截断的 JSON 文件，下次读取时报 `Expecting value: line 21`，
    排查方向会被带偏到文件损坏而不是类型问题上。
    """
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def parse_args():
    parser = argparse.ArgumentParser(description="跨方法 NER 对比")
    parser.add_argument("--split", default="validation", choices=["validation", "test"])
    parser.add_argument("--bert_max_len", type=int, default=128,
                        help="BERT 评估用的 max_length（默认 128 → 前 126 字）。"
                             "评估 JSON 里有 max_length 字段时以文件为准")
    parser.add_argument("--bert_tag", default="",
                        help="读 work_eval_*_{split}{tag}.json，如 --bert_tag full "
                             "用放开截断的那份结果")
    return parser.parse_args()


if __name__ == "__main__":
    main()
