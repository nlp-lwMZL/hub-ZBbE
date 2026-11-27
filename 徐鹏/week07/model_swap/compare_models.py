# -*- coding: utf-8 -*-
"""
「换模型」横向对比：把不同模型 / 不同方法放到同一张表里

这个脚本只做三件事，不训练、不推理：
  1. 读各方法已经写好的统一指标 JSON（outputs/metrics/**），**不手抄数字**；
  2. 对同一批 n=200 样本做 bootstrap 95% 置信区间，避免把噪声当结论；
  3. 生成 markdown / csv / 图，供报告直接引用。

评测口径：全部来自 common/evaluate_events.py（trigger / argument / event 三级 F1）。
缺失的实验显式标「未跑」，绝不填 0 或估算值。

使用方式：
  python model_swap/compare_models.py                 # 默认 1000 次 bootstrap
  python model_swap/compare_models.py --bootstrap 0   # 跳过置信区间（更快）
  python model_swap/compare_models.py --out_dir outputs/model_swap
"""

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path
from statistics import mean

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "common"))

import evaluate_events as EV  # noqa: E402

METRIC_DIR = ROOT / "outputs" / "metrics"
DISC_PRED = ROOT / "outputs" / "disc"
LLM_PRED = ROOT / "outputs" / "llm"
GOLD = {
    200: LLM_PRED / "gold_n200.jsonl",
    60: LLM_PRED / "gold_n60.jsonl",
}
METRICS = ("trigger", "argument", "event_exact")


# ══════════════════════════════════════════════════════════════════════════════
# 实验登记表：一个方法 = 一条记录
# ══════════════════════════════════════════════════════════════════════════════
# metric      : n 条样本的指标 JSON（相对 outputs/metrics，不带 .json）
# full_metric : 全量 dev(930) 指标（用来取参数量/训练时长/延迟/非法序列率）
# pred        : 预测 JSONL；用于 bootstrap（相对 outputs/）
# report      : LLM 链路的工程统计（JSON 合法率 / 幻觉 / 延迟）
# n           : 预期评测样本数；None = 从指标 JSON 读
# params/train/latency: 拿不到时显式写 "-"，不估算
EXPERIMENTS = [
    dict(key="rule", method="L0 规则（词典+正则）", model="无模型", family="规则",
         metric="disc/rule_n200", full_metric="disc/rule",
         pred="disc/pred_rule.jsonl", n=200, params="0", train="0",
         note="零训练基线；论元靠词典/正则"),
    dict(key="bilstm_crf", method="L1 BiLSTM-CRF", model="BiLSTM-CRF（1.31M，随机初始化）",
         family="判别式", metric="disc/bilstm_crf_n200", full_metric="disc/bilstm_crf",
         pred="disc/pred_bilstm_crf.jsonl", n=200, params="1.31M",
         note="字符级、无预训练"),
    dict(key="bert_linear_256", method="L2 BERT+Linear", model="bert-base-chinese + Linear",
         family="判别式", metric="disc/bert_linear_256_n200",
         full_metric="disc/bert_linear_256", pred="disc/pred_bert_linear_256.jsonl",
         n=200, params="102.4M", note="联合 149 标签，线性解码"),
    dict(key="bert_crf_256", method="L2 BERT+CRF", model="bert-base-chinese + CRF",
         family="判别式", metric="disc/bert_crf_256_n200", full_metric="disc/bert_crf_256",
         pred="disc/pred_bert_crf_256.jsonl", n=200, params="102.4M",
         note="联合 149 标签，CRF 解码"),
    dict(key="pipeline", method="L2 两段式流水线", model="bert-base-chinese（检测+13 专家）",
         family="判别式", metric="disc/pipeline_n200", full_metric="disc/pipeline",
         pred="disc/pred_pipeline.jsonl", n=200, params="16 个小模型",
         note="检测→类型→类型条件论元；误差会级联"),
    dict(key="bert_crf_256_hard", method="L2 BERT+CRF + 解码硬约束", model="bert-base-chinese + CRF",
         family="判别式", metric="disc/bert_crf_256_hard_n200",
         full_metric="disc/bert_crf_256_hard", pred="disc/pred_bert_crf_256_hard.jsonl",
         n=200, params="102.4M", note="同一 checkpoint，仅解码期加 BIO 硬约束"),
    dict(key="api_zero_n200", method="L3 LLM zero-shot", model="DeepSeek API（外部大模型）",
         family="生成式", metric="llm/api_zero_n200", pred="llm/pred_api_zero_n200.jsonl",
         report="llm/report_api_zero_n200.json", n=200, params="-", train="0",
         note="API 按 token 计费；延迟含网络"),
    dict(key="api_few_n200", method="L3 LLM few-shot", model="DeepSeek API（外部大模型）",
         family="生成式", metric="llm/api_few_n200", pred="llm/pred_api_few_n200.jsonl",
         report="llm/report_api_few_n200.json", n=200, params="-", train="0",
         note="6 个难类型示例；prompt 更长"),
    dict(key="local_zero_n60", method="L3 LLM zero-shot", model="Qwen3-0.6B（课程原始，本地）",
         family="生成式", metric="llm/local_zero_n60", pred="llm/pred_local_zero_n60.jsonl",
         report="llm/report_local_zero_n60.json", n=60, params="0.6B", train="0",
         note="n=60；本地小模型零样本"),
    dict(key="qwen2_zero_n60", method="L3 LLM zero-shot", model="Qwen2-0.5B（base，本地）",
         family="生成式", metric="llm/qwen2_zero_n60", pred="llm/pred_qwen2_zero_n60.jsonl",
         report="llm/report_qwen2_zero_n60.json", n=60, params="0.49B", train="0",
         note="n=60；base 版无指令跟随能力"),
    dict(key="sft_lora_n200", method="L4 LoRA SFT", model="Qwen3-0.6B + LoRA（课程原始）",
         family="生成式", metric="llm/sft_lora_n200", pred="llm/pred_sft_lora_n200.jsonl",
         report="llm/report_sft_lora_n200.json", n=200, params="0.6B+2.29M",
         train="2327", note="3500 条 × 1 epoch；课程原始结果"),
    dict(key="qwen2_lora_n200", method="L4 LoRA SFT", model="Qwen2-0.5B + LoRA（本次换模型）",
         family="生成式", metric="llm/qwen2_lora_n200", pred="llm/pred_qwen2_lora_n200.jsonl",
         report="llm/report_qwen2_lora_n200.json", n=200, params="0.49B+1.08M",
         note="3500 条 × 1 epoch；与 Qwen3 同配置"),
    dict(key="qwen25_zero_n200", method="L3 LLM zero-shot", model="Qwen2.5-0.5B-Instruct（本地）",
         family="生成式", metric="llm/qwen25_zero_n200", pred="llm/pred_qwen25_zero_n200.jsonl",
         report="llm/report_qwen25_zero_n200.json", n=200, params="0.49B", train="0",
         note="带指令微调的 0.5B；对照组"),
    dict(key="qwen25_zero_n60", method="L3 LLM zero-shot", model="Qwen2.5-0.5B-Instruct（本地）",
         family="生成式", metric="llm/qwen25_zero_n60", pred="llm/pred_qwen25_zero_n200.jsonl",
         report="llm/report_qwen25_zero_n200.json", n=60, params="0.49B", train="0",
         note="n=60 前缀子集，与 Qwen2/Qwen3 零样本同批对比"),
    dict(key="qwen25_zero_n60_note", method="L3 LLM zero-shot（prompt 消融）",
         model="Qwen2.5-0.5B-Instruct（澄清占位符）", family="生成式",
         metric="llm/qwen25_zero_n60_note", pred="llm/pred_qwen25_zero_n60_note.jsonl",
         report="llm/report_qwen25_zero_n60_note.json", n=60, params="0.49B", train="0",
         note="追加一句「占位符要替换」；F1 仍≈0 → 是能力问题不是措辞问题"),
    dict(key="qwen25_lora_n200", method="L4 LoRA SFT", model="Qwen2.5-0.5B-Instruct + LoRA",
         family="生成式", metric="llm/qwen25_lora_n200", pred="llm/pred_qwen25_lora_n200.jsonl",
         report="llm/report_qwen25_lora_n200.json", n=200, params="0.49B+1.08M",
         note="3500 条 × 1 epoch；与 Qwen2 同配置"),
]


# ══════════════════════════════════════════════════════════════════════════════
# 读取与统计
# ══════════════════════════════════════════════════════════════════════════════

def read_json(path: Path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def bootstrap_ci(gold_rows, pred_rows, metric="argument", iters=1000, seed=42):
    """
    对 n 条样本做 bootstrap，返回 F1 的 95% 置信区间。

    为什么需要：n=200 时 F1 的标准误大约 3~4 个点，两个方法差 1~2 个点
    很可能只是采样噪声。报告里写「+1.5 个点」时必须把区间一起给出。
    """
    if iters <= 0 or not gold_rows:
        return None
    pmap = {EV.row_key(r): r for r in pred_rows}
    counts = []
    for g in gold_rows:
        p = pmap.get(EV.row_key(g), {"events": []})
        t, a, e = EV.compare_events(EV.as_event_list(g), EV.as_event_list(p))
        counts.append({"trigger": t, "argument": a, "event_exact": e}[metric])

    rng = random.Random(seed)
    n = len(counts)
    f1s = []
    for _ in range(iters):
        tp = fp = fn = 0
        for _ in range(n):
            c = counts[rng.randrange(n)]
            tp += c[0]; fp += c[1]; fn += c[2]
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * p * r / (p + r) if (p + r) else 0.0)
    f1s.sort()
    lo = f1s[int(0.025 * iters)]
    hi = f1s[min(iters - 1, int(0.975 * iters))]
    return {"lo": round(lo, 4), "hi": round(hi, 4),
            "half_width": round((hi - lo) / 2, 4)}


def enrich(exp: dict, bootstrap: int) -> dict | None:
    """读指标 + 元信息，返回一行；缺指标 JSON 返回 None（由调用方标未跑）。"""
    mpath = METRIC_DIR / f"{exp['metric']}.json"
    m = read_json(mpath)
    if m is None:
        return None

    full = read_json(METRIC_DIR / f"{exp['full_metric']}.json") if exp.get("full_metric") else None
    meta = (full or {}).get("meta") or {}
    report = read_json(ROOT / "outputs" / exp["report"]) if exp.get("report") else None

    n = m.get("num_samples") or exp.get("n") or 0
    train_s = exp.get("train", meta.get("train_s", "-"))
    # 训练时长以训练日志为准（比手填可靠）
    log_path = ROOT / "outputs" / "logs" / f"{exp['key'].replace('_n200', '').replace('_n60', '')}_train.json"
    log = read_json(log_path)
    if log and log.get("total_elapsed_s"):
        train_s = round(log["total_elapsed_s"], 1)

    latency_ms = "-"
    if meta.get("ms_per_sentence") is not None:
        latency_ms = meta["ms_per_sentence"]
    elif report and report.get("latency_mean_s"):
        latency_ms = round(report["latency_mean_s"] * 1000, 1)

    illegal = (meta.get("illegal_dev") or {}).get("illegal_seq_rate")
    row = {
        "key": exp["key"],
        "方法": exp["method"],
        "模型": exp["model"],
        "类别": exp["family"],
        "n": n,
        "触发词F1": m["trigger"]["f1"],
        "论元F1": m["argument"]["f1"],
        "事件级F1": m["event_exact"]["f1"],
        "类型macroF1": m.get("type_macro_f1", 0.0),
        "参数量": exp.get("params", "-"),
        "训练(s)": train_s,
        "延迟(ms/句)": latency_ms,
        "非法BIO%": round(illegal * 100, 2) if illegal is not None else "-",
        "JSON合法%": report.get("json_valid_rate") if report else "-",
        "备注": exp.get("note", ""),
    }

    # bootstrap 95% CI（只对能拿到预测文件的实验做）
    ci = None
    if bootstrap > 0 and exp.get("pred") and n in GOLD:
        pred_path = ROOT / "outputs" / exp["pred"]
        if pred_path.exists() and GOLD[n].exists():
            gold_rows = read_jsonl(GOLD[n])
            pred_rows = read_jsonl(pred_path)
            ci = bootstrap_ci(gold_rows, pred_rows, "argument", bootstrap)
    row["论元F1_95CI"] = f"[{ci['lo']:.3f},{ci['hi']:.3f}]" if ci else "-"
    row["_ci"] = ci
    return row


# ══════════════════════════════════════════════════════════════════════════════
# 输出
# ══════════════════════════════════════════════════════════════════════════════

MAIN_COLS = ["方法", "模型", "n", "触发词F1", "论元F1", "论元F1_95CI",
             "事件级F1", "类型macroF1", "参数量", "训练(s)", "延迟(ms/句)",
             "非法BIO%", "JSON合法%", "备注"]
FULL_COLS = ["方法", "模型", "n", "触发词F1", "论元F1", "事件级F1", "类型macroF1",
             "参数量", "训练(s)", "延迟(ms/句)", "非法BIO%", "备注"]


def full_rows(experiments: list[dict]) -> list[dict]:
    """全量 dev(930) 表：只有写了 full_metric 的方法有；LLM 链路未跑全量则缺席。"""
    out = []
    for exp in experiments:
        if not exp.get("full_metric"):
            continue
        m = read_json(METRIC_DIR / f"{exp['full_metric']}.json")
        if m is None:
            continue
        meta = m.get("meta") or {}
        illegal = (meta.get("illegal_dev") or {}).get("illegal_seq_rate")
        out.append({
            "方法": exp["method"], "模型": exp["model"],
            "n": m.get("num_samples", 930),
            "触发词F1": f"{m['trigger']['f1']:.4f}",
            "论元F1": f"{m['argument']['f1']:.4f}",
            "事件级F1": f"{m['event_exact']['f1']:.4f}",
            "类型macroF1": f"{m.get('type_macro_f1', 0):.4f}",
            "参数量": exp.get("params", meta.get("params_m", "-")),
            "训练(s)": meta.get("train_s", exp.get("train", "-")),
            "延迟(ms/句)": meta.get("ms_per_sentence", "-"),
            "非法BIO%": f"{illegal * 100:.2f}" if illegal is not None else "-",
            "备注": exp.get("note", ""),
        })
    return out


def full_md_table(rows: list[dict]) -> str:
    if not rows:
        return ""
    lines = ["### 全量 dev 930 句（规则 / 判别式；LLM 链路只跑了 n=200 子集）", "",
             "| " + " | ".join(FULL_COLS) + " |",
             "|" + "|".join(["---"] * len(FULL_COLS)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(c, "-")) for c in FULL_COLS) + " |")
    return "\n".join(lines) + "\n"


def md_table(rows: list[dict], title: str) -> str:
    if not rows:
        return f"### {title}\n\n（暂无数据）\n"
    lines = [f"### {title}", "",
             "| " + " | ".join(MAIN_COLS) + " |",
             "|" + "|".join(["---"] * len(MAIN_COLS)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(c, "-")) for c in MAIN_COLS) + " |")
    return "\n".join(lines) + "\n"


def paired_delta(gold_rows, pred_a, pred_b, metric="argument", iters=1000, seed=42):
    """
    配对 bootstrap：同一批样本、同一组重采样下标，算两个方法 F1 差的 95% CI。

    比「两个独立 CI 有没有重叠」更准：它消掉了样本难度带来的共同波动。
    CI 不含 0 → 在这个样本量下可以认为两者有差别；含 0 → 分不出来。
    """
    if not gold_rows or iters <= 0:
        return None
    ma = {EV.row_key(r): r for r in pred_a}
    mb = {EV.row_key(r): r for r in pred_b}
    ca, cb = [], []
    for g in gold_rows:
        k = EV.row_key(g)
        _, a1, _ = EV.compare_events(EV.as_event_list(g), EV.as_event_list(ma.get(k, {"events": []})))
        _, a2, _ = EV.compare_events(EV.as_event_list(g), EV.as_event_list(mb.get(k, {"events": []})))
        ca.append(a1); cb.append(a2)

    def f1(c):
        tp, fp, fn = c
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        return 2 * p * r / (p + r) if (p + r) else 0.0

    rng = random.Random(seed)
    n = len(gold_rows)
    deltas = []
    for _ in range(iters):
        idx = [rng.randrange(n) for _ in range(n)]
        ta = [sum(ca[i][j] for i in idx) for j in range(3)]
        tb = [sum(cb[i][j] for i in idx) for j in range(3)]
        deltas.append(f1(tb) - f1(ta))     # Δ = 后者 − 前者，与表格里的“A → B”方向一致
    deltas.sort()
    lo = deltas[int(0.025 * iters)]
    hi = deltas[min(iters - 1, int(0.975 * iters))]
    base_a = f1([sum(c[j] for c in ca) for j in range(3)])
    base_b = f1([sum(c[j] for c in cb) for j in range(3)])
    return {"delta": round(base_b - base_a, 4), "lo": round(lo, 4),
            "hi": round(hi, 4), "separated": (lo > 0 or hi < 0)}


PAIRS = [
    ("bert_linear_256", "bert_crf_256", "BERT 解码器：Linear → CRF"),
    ("bert_crf_256", "bert_crf_256_hard", "同一 checkpoint：加/不加 BIO 硬约束"),
    ("bert_crf_256", "pipeline", "联合抽取 → 两段式流水线"),
    ("api_zero_n200", "api_few_n200", "API：zero-shot → few-shot"),
    ("local_zero_n60", "qwen2_zero_n60", "零样本换模型：Qwen3-0.6B → Qwen2-0.5B（n=60）"),
    ("qwen2_zero_n60", "qwen25_zero_n60", "零样本换模型：Qwen2-0.5B base → Qwen2.5-0.5B-Instruct（n=60）"),
    ("bert_crf_256", "sft_lora_n200", "判别式 BERT+CRF → 生成式 Qwen3+LoRA"),
    ("bert_crf_256", "qwen2_lora_n200", "判别式 BERT+CRF → 生成式 Qwen2+LoRA"),
    ("sft_lora_n200", "qwen2_lora_n200", "LoRA 换模型：Qwen3-0.6B → Qwen2-0.5B"),
    ("qwen2_lora_n200", "qwen25_lora_n200", "LoRA 换模型：Qwen2-0.5B → Qwen2.5-0.5B-Instruct"),
    ("sft_lora_n200", "qwen25_lora_n200", "LoRA 换模型：Qwen3-0.6B → Qwen2.5-0.5B-Instruct"),
    ("qwen25_zero_n200", "qwen25_lora_n200", "Qwen2.5：zero-shot → LoRA SFT"),
    ("qwen2_lora_n200", "qwen25_zero_n200", "生成式两条路：Qwen2+LoRA → Qwen2.5 zero-shot"),
    ("api_zero_n200", "qwen25_zero_n200", "zero-shot：DeepSeek API → 本地 Qwen2.5-0.5B-Instruct"),
    ("bert_crf_256", "qwen25_zero_n200", "判别式 BERT+CRF → 本地 Qwen2.5 zero-shot"),
]


def paired_findings(experiments: list[dict], iters: int) -> str:
    by = {e["key"]: e for e in experiments}
    lines = ["### 关键对比：配对 bootstrap（论元 F1 差值，95% CI）", "",
             "| 对比 | Δ论元F1 | 95% CI | 该样本量下可分？ |",
             "|---|---|---|---|"]
    any_row = False
    for a_key, b_key, label in PAIRS:
        a, b = by.get(a_key), by.get(b_key)
        if not a or not b:
            continue
        n = a.get("n")
        if n != b.get("n"):
            continue
        if n not in GOLD or not GOLD[n].exists():
            continue
        pa = ROOT / "outputs" / a["pred"] if a.get("pred") else None
        pb = ROOT / "outputs" / b["pred"] if b.get("pred") else None
        if not pa or not pb or not pa.exists() or not pb.exists():
            continue
        gold_rows = read_jsonl(GOLD[n])
        res = paired_delta(gold_rows, read_jsonl(pa), read_jsonl(pb), "argument", iters)
        if not res:
            continue
        any_row = True
        mark = "是" if res["separated"] else "否（含 0）"
        lines.append(f"| {label} | {res['delta']:+.4f} | "
                     f"[{res['lo']:+.4f}, {res['hi']:+.4f}] | {mark} |")
    if not any_row:
        return ""
    lines.append("")
    lines.append("> 配对 bootstrap 用同一组重采样下标同时算两个方法，"
                 "比「各自 CI 是否重叠」更适合判断“谁更好”。")
    return "\n".join(lines) + "\n"


def subset_bias(experiments: list[dict]) -> str:
    """
    n=200 子集 vs 全量 930：同一方法在同一份数据上只换了评测子集，指标差多少。
    用来提醒「n=200 上 1~2 个点的差异可能只是子集构成变了」。
    """
    lines = ["### 子集偏差：n=200 vs 全量 930（论元 F1）", "",
             "| 方法 | 全量 930 | n=200 子集 | 子集 - 全量 | 结论是否会被带偏 |",
             "|---|---|---|---|---|"]
    any_row = False
    for exp in experiments:
        if not exp.get("full_metric"):
            continue
        full = read_json(METRIC_DIR / f"{exp['full_metric']}.json")
        sub = read_json(METRIC_DIR / f"{exp['metric']}.json")
        if not full or not sub:
            continue
        any_row = True
        d = sub["argument"]["f1"] - full["argument"]["f1"]
        flag = "会（差 >3pp）" if abs(d) > 0.03 else "基本一致"
        lines.append(f"| {exp['model']} / {exp['method']} | {full['argument']['f1']:.4f} | "
                     f"{sub['argument']['f1']:.4f} | {d:+.4f} | {flag} |")
    if not any_row:
        return ""
    lines.append("")
    lines.append("> LLM 链路没有全量结果，因此这张表只覆盖规则/判别式方法；"
                 "不要拿 n=200 的 LLM 数字去和 930 的 BERT 数字直接比。")
    return "\n".join(lines) + "\n"


def model_matrix(rows: list[dict]) -> str:
    """LLM 换模型矩阵：行=方法，列=模型，单元格=论元 F1（n）。"""
    llm = [r for r in rows if r["类别"] == "生成式"]
    methods = ["L3 LLM zero-shot", "L3 LLM few-shot", "L4 LoRA SFT"]
    models = []
    for r in llm:
        if r["模型"] not in models:
            models.append(r["模型"])
    if not llm:
        return ""
    lines = ["### LLM 换模型矩阵（论元 F1，括号内为样本数）", "",
             "| 方法 | " + " | ".join(models) + " |",
             "|---" * (len(models) + 1) + "|"]
    for meth in methods:
        cells = []
        for mod in models:
            hits = [r for r in llm if r["方法"] == meth and r["模型"] == mod]
            hit = max(hits, key=lambda r: r["n"]) if hits else None
            cells.append(f"{hit['论元F1']:.4f} ({hit['n']})" if hit else "未跑")
        lines.append(f"| {meth} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def auto_findings(rows: list[dict]) -> str:
    by = {r["key"]: r for r in rows}
    out = ["### 自动生成的关键对比（只统计两边都跑了的组合）", ""]

    def delta(a, b, desc, metric="论元F1"):
        if a in by and b in by:
            va, vb = by[a][metric], by[b][metric]
            if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                out.append(f"- {desc}：{metric} {va:.4f} → {vb:.4f}"
                           f"（{vb - va:+.4f}，相对 {(vb - va) / max(va, 1e-9):+.1%}）")

    delta("bilstm_crf", "bert_crf_256", "预训练编码器（BiLSTM → BERT+CRF）")
    delta("bert_linear_256", "bert_crf_256", "解码器（Linear → CRF）")
    delta("bert_crf_256", "pipeline", "联合抽取 → 两段式流水线")
    delta("bert_crf_256", "bert_crf_256_hard", "CRF 解码加 BIO 硬约束（同一 checkpoint）")
    delta("api_zero_n200", "api_few_n200", "LLM 零样本 → 少样本（DeepSeek API）")
    delta("local_zero_n60", "qwen2_zero_n60", "零样本换模型（Qwen3-0.6B → Qwen2-0.5B，n=60）")
    delta("qwen2_zero_n60", "qwen25_zero_n60",
          "零样本换模型（Qwen2-0.5B base → Qwen2.5-0.5B-Instruct，n=60）")
    delta("sft_lora_n200", "qwen2_lora_n200", "LoRA 换模型（Qwen3-0.6B → Qwen2-0.5B）")
    delta("qwen2_lora_n200", "qwen25_lora_n200", "LoRA 换模型（Qwen2-0.5B → Qwen2.5-0.5B-Instruct）")
    delta("bert_crf_256", "qwen2_lora_n200", "判别式 BERT+CRF → 生成式 Qwen2+LoRA")
    delta("bert_crf_256", "sft_lora_n200", "判别式 BERT+CRF → 生成式 Qwen3+LoRA")

    # 最好的方法 / 性价比（延迟）
    n200 = [r for r in rows if r["n"] == 200]
    if n200:
        best = max(n200, key=lambda r: r["论元F1"])
        out.append(f"- n=200 论元 F1 最高：**{best['模型']} / {best['方法']}**"
                   f"（{best['论元F1']:.4f}）")
        timed = [r for r in n200 if isinstance(r["延迟(ms/句)"], (int, float))
                 and r["延迟(ms/句)"] > 0]
        if timed:
            fast = min(timed, key=lambda r: r["延迟(ms/句)"])
            out.append(f"- 延迟最低：**{fast['模型']} / {fast['方法']}**"
                       f"（{fast['延迟(ms/句)']} ms/句）")
    return "\n".join(out) + "\n"


def make_figure(rows: list[dict], out: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib import font_manager
    except ImportError:
        print("[skip] 未安装 matplotlib，跳过图")
        return

    available = {f.name for f in font_manager.fontManager.ttflist}
    for cand in ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC"]:
        if cand in available:
            plt.rcParams["font.sans-serif"] = [cand, "DejaVu Sans"]
            break
    plt.rcParams["axes.unicode_minus"] = False

    n200 = [r for r in rows if r["n"] == 200]
    if not n200:
        return
    n200 = sorted(n200, key=lambda r: r["论元F1"])
    labels = [f"{r['模型']}\n{r['方法']}" for r in n200]
    vals = [r["论元F1"] for r in n200]
    err = []
    for r in n200:
        ci = r.get("_ci")
        err.append(ci["half_width"] if ci else 0)

    fig, axes = plt.subplots(1, 2, figsize=(15, 0.55 * len(n200) + 3),
                             gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axes[0]
    y = np.arange(len(n200))
    colors = ["#F58518" if r["类别"] == "生成式" else
              ("#54A24B" if r["类别"] == "判别式" else "#9E9E9E") for r in n200]
    ax.barh(y, vals, xerr=err, color=colors, edgecolor="white", capsize=3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    for i, v in enumerate(vals):
        ax.text(v + 0.012, i, f"{v:.3f}", va="center", fontsize=9)
    ax.set_xlim(0, 0.9)
    ax.set_xlabel("论元 F1（误差线 = bootstrap 95% CI）")
    ax.set_title("各模型 / 方法论元抽取效果（n=200，同一评测口径）")
    ax.grid(axis="x", alpha=0.25, linestyle="--")

    ax2 = axes[1]
    timed = [r for r in n200 if isinstance(r["延迟(ms/句)"], (int, float))
             and r["延迟(ms/句)"] > 0]
    for r in timed:
        c = "#F58518" if r["类别"] == "生成式" else ("#54A24B" if r["类别"] == "判别式" else "#9E9E9E")
        ax2.scatter(r["延迟(ms/句)"], r["论元F1"], s=55, color=c, alpha=0.85)
        short = r["模型"].replace("（课程原始）", "").replace("（本次换模型）", "")
        ax2.annotate(short, (r["延迟(ms/句)"], r["论元F1"]),
                     textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax2.set_xscale("log")
    ax2.set_xlabel("推理延迟 ms/句（对数轴；含生成式逐 token 解码）")
    ax2.set_ylabel("论元 F1")
    ax2.set_title("效果 vs 延迟")
    ax2.grid(alpha=0.25, linestyle="--")

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [ok] {out.relative_to(ROOT)}")


def main():
    ap = argparse.ArgumentParser(description="换模型 / 多方法事件抽取对比")
    ap.add_argument("--bootstrap", default=1000, type=int,
                    help="bootstrap 次数；0 = 不算置信区间")
    ap.add_argument("--out_dir", default="outputs/model_swap")
    args = ap.parse_args()

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, missing = [], []
    for exp in EXPERIMENTS:
        r = enrich(exp, args.bootstrap)
        if r is None:
            missing.append(exp)
        else:
            rows.append(r)

    # 表内顺序 = 从「不用模型」到「大模型」的教学递进，而不是按字母排
    key_order = [e["key"] for e in EXPERIMENTS]
    rows.sort(key=lambda r: key_order.index(r["key"]))

    n200 = [r for r in rows if r["n"] == 200]
    n60 = [r for r in rows if r["n"] == 60]

    md = ["# 换模型 / 多方法事件抽取对比", "",
          "> 评测集：`data/processed/dev_extract.jsonl` 的固定子集（seed=42，canonical order）。",
          "> 所有数字直接读 `outputs/metrics/**.json`，由 `common/evaluate_events.py` 统一计算。",
          f"> 置信区间：对 n 条样本做 {args.bootstrap} 次 bootstrap（论元 F1 的 95% CI）。", ""]
    md.append(md_table(n200, "主表：n=200 同批对比"))
    if n60:
        md.append(md_table(n60, "小样本同批对比（n=60）"))
    md.append(full_md_table(full_rows(EXPERIMENTS)))
    md.append(subset_bias(EXPERIMENTS))
    md.append(model_matrix(rows))
    md.append(auto_findings(rows))
    md.append(paired_findings(EXPERIMENTS, args.bootstrap))
    if missing:
        md.append("### 未跑 / 缺结果（不填估算值）\n")
        for e in missing:
            md.append(f"- `{e['metric']}` —— {e['model']} / {e['method']}")
        md.append("")

    report_path = out_dir / "REPORT.md"
    report_path.write_text("\n".join(md), encoding="utf-8")

    csv_path = out_dir / "model_swap.csv"
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MAIN_COLS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in MAIN_COLS})

    make_figure(rows, out_dir / "fig_model_swap.png")

    print("\n".join(md))
    print(f"\n已写入：\n  {report_path.relative_to(ROOT)}\n  {csv_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
