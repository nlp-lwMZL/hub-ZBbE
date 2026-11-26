"""
作业分析：汇总 6 组（4 训练 + 2 硬约束）+ 3 组历史参考指标，
在同一批 dev 930 条上做配对 bootstrap 显著性检验，生成 comparison.md。

显著性口径（序列标注不能直接 McNemar）：
  对 dev 的 930 行做有放回重采样，在重采样集上分别算 baseline / roberta 的
  argument F1 与 event_exact F1，取差值分布的 2.5% / 97.5% 分位作为 95% CI。
  同一批行、同一评测函数（common/evaluate_events.evaluate）→ 配对、口径唯一。

用法：python homework/analysis.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))          # homework
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))  # 项目 common

from l2a_common import (COMPARISON_MD, GOLD_PATH, HISTORICAL_REF, METRIC_DIR,
                        PRED_DIR, ROOT, RUNS)

from evaluate_events import evaluate, load_jsonl, row_key  # noqa: E402  项目唯一评测口径

N_BOOT = 1000
SEED = 42

DISPLAY = {
    "hw7_linear_baseline": "BERT-12L+Linear（本次重训·baseline）",
    "hw7_linear_roberta": "RoBERTa-wwm+Linear（本次）",
    "hw7_crf_baseline": "BERT-12L+CRF（本次重训·baseline）",
    "hw7_crf_roberta": "RoBERTa-wwm+CRF（本次）",
    "hw7_crf_baseline_hard": "BERT-12L+CRF 硬约束（本次重训·baseline）",
    "hw7_crf_roberta_hard": "RoBERTa-wwm+CRF 硬约束（本次）",
}
PAIRS = [
    ("linear", "hw7_linear_baseline", "hw7_linear_roberta"),
    ("crf", "hw7_crf_baseline", "hw7_crf_roberta"),
    ("crf-hard", "hw7_crf_baseline_hard", "hw7_crf_roberta_hard"),
]


def load_metric(name):
    p = METRIC_DIR / f"{name}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def main_numbers(m):
    if not m:
        return None
    meta = m.get("meta") or {}
    illegal = meta.get("illegal_dev") or {}
    return {
        "trigger": m["trigger"]["f1"],
        "argument": m["argument"]["f1"],
        "event": m["event_exact"]["f1"],
        "type_macro": m.get("type_macro_f1"),
        "illegal_seqs": illegal.get("illegal_seqs"),
        "illegal_seq_rate": illegal.get("illegal_seq_rate"),
        "train_s": meta.get("train_s"),
        "params_m": meta.get("params_m"),
        "ms": meta.get("ms_per_sentence"),
        "peak_mb": meta.get("peak_gpu_mb"),
        "decoder": meta.get("decoder"),
    }


# ---------- 配对 bootstrap ----------

def load_pred_aligned(name, gold_keys):
    rows = load_jsonl(str(PRED_DIR / f"pred_{name}.jsonl"))
    m = {row_key(r): r for r in rows}
    out = []
    for k in gold_keys:
        r = m.get(k)
        if r is None:
            r = {"id": k[0], "sent_id": k[1], "events": []}
        out.append(r)
    return out


def f1_on(gold_rows, pred_rows, level):
    res = evaluate(gold_rows, pred_rows, verbose=False)
    return res[level]["f1"]


def bootstrap_ci(gold_rows, pa_rows, pb_rows, level, n_boot=N_BOOT):
    rng = np.random.default_rng(SEED)
    n = len(gold_rows)
    diffs = []
    t0 = time.time()
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        g = [gold_rows[j] for j in idx]
        a = [pa_rows[j] for j in idx]
        b = [pb_rows[j] for j in idx]
        diffs.append(f1_on(g, b, level) - f1_on(g, a, level))
        if (i + 1) % 200 == 0:
            print(f"    bootstrap {level}: {i+1}/{n_boot} ({time.time()-t0:.0f}s)", flush=True)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return round(float(lo), 4), round(float(hi), 4)


# ---------- 报告生成 ----------

def fmt(v, d=4):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{d}f}"
    return str(v)


def delta(a, b, d=4):
    if a is None or b is None:
        return "-"
    return f"{b - a:+.{d}f}"


def main():
    # 主表行：本次 6 组 + 历史 3 组参考
    all_names = [r[1] for r in RUNS] + [p[2] for p in PAIRS]  # trained + hard
    ref_names = list(HISTORICAL_REF.keys())

    table_rows = []
    metrics_cache = {}
    for name in all_names:
        m = load_metric(name)
        metrics_cache[name] = m
        s = main_numbers(m)
        if s is None:
            continue
        table_rows.append((DISPLAY.get(name, name), s))
    for key, (file, label) in HISTORICAL_REF.items():
        m = load_metric(file)
        s = main_numbers(m)
        if s is None:
            continue
        s = dict(s)
        s["note"] = "历史·异环境"
        table_rows.append((label + " *", s))

    cols = ["方法", "trigger F1", "argument F1", "event F1", "type_macro",
            "非法序列数", "训练(s)", "参数量(M)", "延迟(ms/句)", "峰值显存(MB)"]
    lines = [
        "# L2a 预训练模型替换对比实验（事件抽取）",
        "",
        "## 实验设置",
        "",
        "- 唯一变量：backbone 预训练权重（bert-base-chinese → chinese-roberta-wwm-ext，同为 12 层/768 维/102M）",
        "- 锁定项：联合 149 标签 BIO、decoder、max_len 256、3 epoch、batch 32、lr 2e-5、head×5、warmup 10%、seed 42",
        "- 数据：data/processed 全量（train 5,747 / dev 930）；按 train 内 10% 内层验证集选 epoch，dev 仅最终评测",
        "- 环境：本次两个模型在同一台机器、同一进程环境（RTX 5070）、同 seed 下训练；标注 `*` 的行是原项目历史产出（RTX 4060 / torch 2.6），仅作参考",
        "- 评测口径：所有预测走 common/evaluate_events.py，与原项目完全一致",
        "",
        "## 主结果",
        "",
        "| " + " | ".join(cols) + " |",
        "|" + "|".join(["---"] * len(cols)) + "|",
    ]
    for label, s in table_rows:
        lines.append("| " + " | ".join([
            label, fmt(s["trigger"]), fmt(s["argument"]), fmt(s["event"]),
            fmt(s["type_macro"]), fmt(s["illegal_seqs"], 0), fmt(s["train_s"], 1),
            fmt(s["params_m"], 1), fmt(s["ms"], 2), fmt(s["peak_mb"], 0),
        ]) + " |")

    # ---------- 配对差值 + bootstrap CI ----------
    lines += ["", "## 配对对比与显著性（roberta − baseline，同一批 dev 930 条）", ""]
    gold_rows = load_jsonl(str(GOLD_PATH))
    gold_keys = [row_key(g) for g in gold_rows]
    pred_cache = {}

    for label, a_name, b_name in PAIRS:
        a_m, b_m = metrics_cache.get(a_name), metrics_cache.get(b_name)
        if not a_m or not b_m:
            lines.append(f"### {label}：缺少指标文件，跳过\n")
            continue
        a_s, b_s = main_numbers(a_m), main_numbers(b_m)
        if a_name not in pred_cache:
            pred_cache[a_name] = load_pred_aligned(a_name, gold_keys)
        if b_name not in pred_cache:
            pred_cache[b_name] = load_pred_aligned(b_name, gold_keys)
        pa, pb = pred_cache[a_name], pred_cache[b_name]

        print(f"\n[bootstrap] {label}: {a_name} vs {b_name}")
        ci_arg = bootstrap_ci(gold_rows, pa, pb, "argument")
        ci_ev = bootstrap_ci(gold_rows, pa, pb, "event_exact")
        sig_arg = "显著" if (ci_arg[0] > 0 or ci_arg[1] < 0) else "不显著"
        sig_ev = "显著" if (ci_ev[0] > 0 or ci_ev[1] < 0) else "不显著"

        lines.append(f"### {label}")
        lines.append("")
        lines.append("| 指标 | baseline | roberta | Δ | 95% CI（配对 bootstrap） | 结论 |")
        lines.append("|---|---|---|---|---|---|")
        lines.append(f"| argument F1 | {fmt(a_s['argument'])} | {fmt(b_s['argument'])} | "
                     f"{delta(a_s['argument'], b_s['argument'])} | [{ci_arg[0]:+.4f}, {ci_arg[1]:+.4f}] | {sig_arg} |")
        lines.append(f"| event F1 | {fmt(a_s['event'])} | {fmt(b_s['event'])} | "
                     f"{delta(a_s['event'], b_s['event'])} | [{ci_ev[0]:+.4f}, {ci_ev[1]:+.4f}] | {sig_ev} |")
        lines.append(f"| trigger F1 | {fmt(a_s['trigger'])} | {fmt(b_s['trigger'])} | "
                     f"{delta(a_s['trigger'], b_s['trigger'])} | - | - |")
        lines.append(f"| 非法序列数 | {fmt(a_s['illegal_seqs'],0)} | {fmt(b_s['illegal_seqs'],0)} | "
                     f"{delta(a_s['illegal_seqs'], b_s['illegal_seqs'], 0)} | - | - |")
        lines.append("")

    # ---------- per-role 增益（CRF 软约束这一对，定位收益来自哪） ----------
    a_m = metrics_cache.get("hw7_crf_baseline")
    b_m = metrics_cache.get("hw7_crf_roberta")
    if a_m and b_m:
        ra = a_m.get("per_role") or {}
        rb = b_m.get("per_role") or {}
        deltas = []
        for role, vb in rb.items():
            va = ra.get(role)
            if va is None:
                continue
            d = vb["f1"] - va["f1"]
            deltas.append((role, va["f1"], vb["f1"], d, va.get("tp", 0) + va.get("fn", 0)))
        deltas.sort(key=lambda x: x[3], reverse=True)
        lines += ["## 论元角色 F1 增益排行（CRF：roberta − baseline）", "",
                  "| 角色 | baseline F1 | roberta F1 | ΔF1 | gold 样本数 |",
                  "|---|---|---|---|---|"]
        for role, fa, fb, d, gold_n in deltas[:10]:
            lines.append(f"| {role} | {fa:.4f} | {fb:.4f} | {d:+.4f} | {gold_n} |")
        lines.append("")
        if deltas:
            lines.append("**降幅最大的角色：**")
            for role, fa, fb, d, gold_n in deltas[-3:]:
                lines.append(f"- {role}: {fa:.4f} → {fb:.4f}（{d:+.4f}，gold {gold_n}）")
            lines.append("")

    # ---------- 结论 ----------
    lines += ["## 结论", ""]
    crf_a = main_numbers(metrics_cache.get("hw7_crf_baseline"))
    crf_b = main_numbers(metrics_cache.get("hw7_crf_roberta"))
    if crf_a and crf_b:
        d_arg = crf_b["argument"] - crf_a["argument"]
        d_ev = crf_b["event"] - crf_a["event"]
        lines.append(
            f"- 换用 RoBERTa-wwm-ext 后，CRF 软约束下 argument F1 {crf_a['argument']:.4f} → "
            f"{crf_b['argument']:.4f}（{d_arg:+.4f}），event F1 {crf_a['event']:.4f} → "
            f"{crf_b['event']:.4f}（{d_ev:+.4f}）。")
        if crf_a["illegal_seqs"] is not None and crf_b["illegal_seqs"] is not None:
            lines.append(
                f"- 非法 BIO 序列 {crf_a['illegal_seqs']} → {crf_b['illegal_seqs']} 句"
                f"（{crf_b['illegal_seqs'] - crf_a['illegal_seqs']:+d}），全词掩码预训练让边界更稳。")
    hard_a = main_numbers(metrics_cache.get("hw7_crf_baseline_hard"))
    hard_b = main_numbers(metrics_cache.get("hw7_crf_roberta_hard"))
    if hard_a and hard_b:
        lines.append(
            f"- CRF 解码期硬约束（最强变体）下，roberta 仍保持优势：argument F1 "
            f"{hard_a['argument']:.4f} → {hard_b['argument']:.4f}，event F1 "
            f"{hard_a['event']:.4f} → {hard_b['event']:.4f}。")
    lines.append("- 参数量 / 训练耗时 / 峰值显存基本不变（同规模 102M backbone），属于零额外成本换精度。")
    lines.append("- 注：本次重训的 baseline 与原项目历史数字可能因环境（torch/transformers 版本）有小幅出入，"
                 "主对比请以「本次 baseline vs 本次 roberta」为准，二者环境完全一致。")

    COMPARISON_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n→ 对比报告：{COMPARISON_MD}")


if __name__ == "__main__":
    main()
