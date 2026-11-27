# -*- coding: utf-8 -*-
"""
按事件类型 / 论元角色拆开看：不同模型到底强在哪、弱在哪

输入：outputs/metrics/**/*.json（都带 per_type / per_role）
输出：outputs/model_swap/per_type.md、per_type.csv、per_role.md

用法：
  python model_swap/per_type_analysis.py
"""

import csv
import json
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parents[1]
METRIC_DIR = ROOT / "outputs" / "metrics"
OUT_DIR = ROOT / "outputs" / "model_swap"

# (显示名, 指标文件相对路径, 样本数)
METHODS = [
    ("规则", "disc/rule_n200", 200),
    ("BiLSTM-CRF", "disc/bilstm_crf_n200", 200),
    ("BERT+Linear", "disc/bert_linear_256_n200", 200),
    ("BERT+CRF", "disc/bert_crf_256_n200", 200),
    ("BERT+CRF+硬约束", "disc/bert_crf_256_hard_n200", 200),
    ("两段式", "disc/pipeline_n200", 200),
    ("API zero", "llm/api_zero_n200", 200),
    ("API few", "llm/api_few_n200", 200),
    ("Qwen3+LoRA", "llm/sft_lora_n200", 200),
    ("Qwen2+LoRA", "llm/qwen2_lora_n200", 200),
    ("Qwen2.5 zero", "llm/qwen25_zero_n200", 200),
    ("Qwen2.5+LoRA", "llm/qwen25_lora_n200", 200),
]


def load(rel: str):
    p = METRIC_DIR / f"{rel}.json"
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def md_table(cols, rows) -> str:
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(lines)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    loaded = [(name, load(rel), n) for name, rel, n in METHODS]
    loaded = [(name, m, n) for name, m, n in loaded if m is not None]

    # ── 事件类型 ──────────────────────────────────────────────────────────────
    type_rows = []
    all_types = []
    for name, m, _ in loaded:
        for t in (m.get("per_type") or {}):
            if t not in all_types:
                all_types.append(t)

    def support(m, t):
        d = (m.get("per_type") or {}).get(t) or {}
        return d.get("tp", 0) + d.get("fn", 0)

    # 用第一个带 per_type 的方法统计支持度（金标一致）
    base_m = next((m for _, m, _ in loaded if m.get("per_type")), {})
    all_types.sort(key=lambda t: -support(base_m, t))
    header = ["事件类型", "金标数"] + [n for n, _, _ in loaded] + ["最好-最差"]
    for t in all_types:
        vals = []
        for _, m, _ in loaded:
            d = (m.get("per_type") or {}).get(t)
            vals.append(round(d["f1"], 4) if d else None)
        present = [v for v in vals if v is not None]
        spread = round(max(present) - min(present), 4) if len(present) >= 2 else "-"
        type_rows.append([t, support(base_m, t)] + [v if v is not None else "未跑" for v in vals] + [spread])

    # ── 论元角色（取金标支持度最高的 15 个）────────────────────────────────────
    role_support = {}
    for name, m, _ in loaded:
        for r, d in (m.get("per_role") or {}).items():
            role_support[r] = max(role_support.get(r, 0), d.get("tp", 0) + d.get("fn", 0))
    top_roles = sorted(role_support, key=lambda r: -role_support[r])[:15]

    chosen = [("BERT+CRF", "disc/bert_crf_256_n200"),
              ("BERT+CRF+硬约束", "disc/bert_crf_256_hard_n200"),
              ("API zero", "llm/api_zero_n200"),
              ("Qwen3+LoRA", "llm/sft_lora_n200"),
              ("Qwen2+LoRA", "llm/qwen2_lora_n200"),
              ("Qwen2.5+LoRA", "llm/qwen25_lora_n200")]
    chosen = [(n, load(r)) for n, r in chosen]
    chosen = [(n, m) for n, m in chosen if m]
    role_header = ["角色", "金标数"] + [n for n, _ in chosen]
    role_rows = []
    for r in top_roles:
        vals = []
        for _, m in chosen:
            d = (m.get("per_role") or {}).get(r)
            vals.append(round(d["f1"], 4) if d else None)
        role_rows.append([r, role_support[r]] + [v if v is not None else "未跑" for v in vals])

    md = ["# 分类型 / 分角色拆解（n=200 子集）", "",
          "> 事件类型指标 = 该类型事件的 event-level F1；角色指标 = 该角色论元的 F1。",
          "> 样本少的长尾类型 F1 抖动很大，看趋势不看小数点后三位。", "",
          "## 事件类型", "", md_table(header, type_rows), "",
          "## 论元角色（按金标支持度取前 15）", "", md_table(role_header, role_rows), ""]
    (OUT_DIR / "per_type.md").write_text("\n".join(md), encoding="utf-8")

    with open(OUT_DIR / "per_type.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(type_rows)

    print("\n".join(md))
    print(f"\n已写入：{OUT_DIR / 'per_type.md'}")


if __name__ == "__main__":
    main()
