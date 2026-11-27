"""
汇总所有方法的结果，生成 PPT / NOTES 可直接引用的对比表

为什么要单独一个汇总脚本：
  PPT 上的主表如果靠人手抄各个 JSON，「某一行是旧数字」这类错误几乎必然发生
  （CLAUDE.md 的踩坑表里就记过「图注写 3 epoch、实际跑了 5 epoch」）。
  所以这里**只从 outputs/metrics/disc/*.json 读**，并同时输出：
    - 一列「溯源」信息（训练时长 / 参数量 / 延迟 / epoch / max_len），
      让表里每个数字都能反查到对应的实验配置
    - 缺失的方法显式标「未跑」，绝不补一个估算值

使用方式：
  python eval/compare_methods.py                 # 汇总全部已有结果
  python eval/compare_methods.py --with_n200     # 额外带 n=200 子集列（跨链路同批对比）
  python eval/compare_methods.py --out outputs/metrics/disc/summary.md
"""

import os
import sys
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # Windows: torch 与 numpy 各带 libiomp5md，不设会 OMP Error #15

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]   # 项目根目录（方法目录的上一级）
sys.path.insert(0, str(ROOT / "common"))     # 公共模块：event_data / crf / eval_utils / models / log_utils

METRIC_DIR = ROOT / "outputs" / "metrics" / "disc"

# 主表顺序 = 教学递进顺序（L0 规则 → L1 无预训练 → L2 BERT → L2 两段式 → 上界/消融）
MAIN_ORDER = [
    ("rule", "L0 规则（词典+正则）"),
    ("bilstm_linear", "L1 BiLSTM + Linear"),
    ("bilstm_crf", "L1 BiLSTM-CRF（无预训练）"),
    ("bert_linear_256", "L2 BERT+Linear（联合，149 标签）"),
    ("bert_crf_256", "L2 BERT+CRF（联合，149 标签）"),
    ("pipeline", "L2 两段式流水线（检测→类型→论元）"),
    ("pipeline_oracle_type", "L2 两段式·类型给金标（上界）"),
]
ABLATION_ORDER = [
    ("bert_linear_128", "消融：max_len 128"),
    # 注意命名：4 层消融跑的是 **linear** 解码器（与主线 bert_linear_256 对照才干净），
    # 所以文件名是 bert4L_linear_256 而不是 bert4L_crf_256 —— 写错名字会让这一整行
    # 在汇总表里静默消失（本项目真实踩过：表格少了一行 4 层消融，只能靠人工发现）。
    ("bert4L_linear_256", "消融：BERT 前 4 层"),
    ("bert_crf_256_hard", "消融：CRF 非法转移硬约束"),
]

def load(name: str) -> dict | None:
    p = METRIC_DIR / f"{name}.json"
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def row_of(name: str, label: str, with_n200: bool) -> dict | None:
    r = load(name)
    if r is None:
        return None
    meta = r.get("meta") or {}
    illegal = (meta.get("illegal_dev") or {})
    row = {
        "方法": label,
        "触发词F1": f"{r['trigger']['f1']:.4f}",
        "论元F1": f"{r['argument']['f1']:.4f}",
        "事件级F1": f"{r['event_exact']['f1']:.4f}",
        "类型macroF1": f"{r.get('type_macro_f1', 0):.4f}",
        "参数(M)": meta.get("params_m", "-"),
        "训练(s)": meta.get("train_s", "-"),
        "延迟(ms/句)": meta.get("ms_per_sentence", "-"),
        "非法序列占比": (f"{illegal.get('illegal_seq_rate', 0):.2%}"
                        if illegal else "-"),
        "epoch": meta.get("epochs", meta.get("arg_epochs", "-")),
        "max_len": meta.get("max_len", "-"),
    }
    if with_n200:
        n = r.get("n200_summary") or {}
        row["n200论元F1"] = f"{n.get('argument_f1', 0):.4f}" if n else "-"
        row["n200事件F1"] = f"{n.get('event_f1', 0):.4f}" if n else "-"
    return row

def to_markdown(rows: list[dict], title: str) -> str:
    if not rows:
        return f"### {title}\n\n（暂无数据）\n"
    cols = list(rows[0].keys())
    lines = [f"### {title}", "",
             "| " + " | ".join(cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(c, "-")) for c in cols) + " |")
    return "\n".join(lines) + "\n"

def main():
    ap = argparse.ArgumentParser(description="汇总各方法指标，生成对比表")
    ap.add_argument("--with_n200", action="store_true", help="附带 n=200 子集列")
    ap.add_argument("--out", default="", help="markdown 输出路径")
    args = ap.parse_args()

    main_rows = [r for name, label in MAIN_ORDER
                 if (r := row_of(name, label, args.with_n200))]
    missing = [label for name, label in MAIN_ORDER if load(name) is None]
    abl_rows = [r for name, label in ABLATION_ORDER
                if (r := row_of(name, label, args.with_n200))]

    md = ["# 事件抽取方法对比（dev_extract 930 句，统一评测口径 common/evaluate_events.py）", ""]
    md.append(to_markdown(main_rows, "主表：方法对比"))
    if missing:
        md.append("> ⚠ 以下方法尚未跑出结果（表中不出现，绝不填估算值）："
                  + "、".join(missing) + "\n")
    if abl_rows:
        md.append(to_markdown(abl_rows, "消融实验"))

    # 关键结论（自动生成，避免手写文字与实际数字脱节）
    if len(main_rows) >= 2:
        md.append("### 自动生成的关键对比")
        by_label = {r["方法"]: r for r in main_rows}
        pairs = [("L0 规则（词典+正则）", "L1 BiLSTM-CRF（无预训练）", "预训练/建模带来的提升"),
                 ("L1 BiLSTM-CRF（无预训练）", "L2 BERT+CRF（联合，149 标签）", "BERT 预训练带来的提升"),
                 ("L2 BERT+CRF（联合，149 标签）", "L2 两段式流水线（检测→类型→论元）",
                  "联合 → 两段式（标签空间变小）带来的提升")]
        for a, b, desc in pairs:
            if a in by_label and b in by_label:
                da = float(by_label[a]["论元F1"])
                db = float(by_label[b]["论元F1"])
                md.append(f"- {desc}：论元 F1 {da:.4f} → {db:.4f}"
                          f"（{db - da:+.4f}，相对 {(db - da) / max(da, 1e-9):+.1%}）")
        md.append("")

    text = "\n".join(md)
    out = Path(args.out) if args.out else METRIC_DIR / "summary.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"已写入 → {out.relative_to(ROOT)}")

if __name__ == "__main__":
    main()
