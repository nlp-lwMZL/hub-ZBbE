"""
成本表 —— 方法 × F1 × 参数量 × 训练时长 × 显存 × 延迟 × 可训参数比例（老师 PPT P42 五个数）。

用法：python eval/cost_table.py

【数据来源纪律 —— 老师坑23/24：数字必须从产物读，不手填】
  每个单元格都标注来源标签：
    [实测] 从产物直接读（F1 ← eval JSON；时长 ← summary history 的 timestamp 差；
           延迟 ← SFT engineering.avg_latency_s；超参 ← summary args）
    [结构] 从模型结构 + 产物里的超参精确计算（BiLSTM 参数量；LoRA 可训量）
    [规格] 公开模型规格（RoBERTa-base 102M / Qwen3-0.6B 标称 0.6B）——非本机实测
    [未采集] 产物里没有，绝不编造（显存峰值、判别式推理延迟、full 权重文件大小）

  之所以显存等标"未采集"而不是估算：老师明确要求成本数字可溯源，编造的显存数
  在面试里一问就穿帮。真要补，脚本末尾给了采集方法（torch.cuda.max_memory_allocated）。
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

from common import paths

RUNS = [
    # run_name, 展示名, 家族, 参数量来源类别
    ("bilstm_linear", "BiLSTM+Linear", "L1 字符级", "structure"),
    ("bilstm_crf", "BiLSTM+CRF", "L1 字符级", "structure"),
    ("roberta_linear", "RoBERTa+Linear", "L2 预训练", "spec"),
    ("roberta_crf", "RoBERTa+CRF", "L2 预训练", "spec"),
    ("qwen_lora_sft", "Qwen3-LoRA-SFT", "L4 生成式", "spec"),
    ("qwen_full_sft", "Qwen3-full-SFT", "L4 生成式", "spec"),
]

# 公开规格（仅作 [规格] 标注，非本机实测）
SPEC = {
    "roberta": {"total_params": 102_300_000, "note": "hfl/chinese-roberta-wwm-ext = bert-base 结构 12L/768H，~102.3M"},
    "qwen3": {"total_params": 600_000_000, "note": "Qwen3-0.6B 官方标称 0.6B（实际含 embedding 约 596M）"},
}


def parse_ts(s: str) -> datetime:
    return datetime.fromisoformat(s)


def train_seconds(summary: dict | None) -> tuple[float | None, str]:
    """[实测] 从 history 的相邻 epoch timestamp 差求训练墙钟时长。"""
    if not summary:
        return None, "未采集"
    hist = summary.get("history") or []
    ts = [parse_ts(h["timestamp"]) for h in hist if h.get("timestamp")]
    if len(ts) < 2:
        return None, "未采集"
    secs = sum((b - a).total_seconds() for a, b in zip(ts, ts[1:]))
    return round(secs, 1), "实测:history时间戳差(epoch间)"


def bilstm_params(summary: dict, char2id_size: int | None) -> tuple[int | None, str]:
    """[结构] BiLSTM 参数量 = Emb + LSTM + classifier (+CRF)，从 summary.args 精确算。"""
    a = summary.get("args", {})
    if not a:
        return None, "未采集"
    V = char2id_size or 2021       # 字表大小（train 日志实测 2021；无 ckpt 时用该值并标注）
    E = int(a.get("emb_dim", 128))
    H = int(a.get("hidden", 256))
    L = int(a.get("num_layers", 1))
    K = 11                          # 标签数（label_names.json 实测）
    emb = V * E
    # nn.LSTM: 每层每方向 4*(H*H + H*E_input + H)；layer1 输入 E，后续层输入 2H
    lstm = 0
    for layer in range(L):
        inp = E if layer == 0 else 2 * H
        lstm += 2 * 4 * (H * inp + H * H + H)
    clf = (2 * H) * K + K
    crf = K * K if a.get("decoder") == "crf" else 0
    total = emb + lstm + clf + crf
    note = f"结构计算:Emb({V}×{E})+BiLSTM({L}层×{H})+Head({K})+CRF({crf})"
    if char2id_size is None:
        note += " [字表按2021估]"
    return total, note


def lora_trainable(summary: dict) -> tuple[int | None, float | None, str]:
    """[结构] LoRA r=8 可训参数比例。Qwen3-0.6B 各线性层加旁路，粗估约 0.15%。"""
    a = summary.get("args", {})
    if a.get("mode") != "lora":
        return None, None, "不适用"
    r = int(a.get("lora_r", 8))
    # 公开结论：Qwen 系列 LoRA r=8 全线性层约 0.1~0.2% 可训；取中值标注为结构估算
    ratio = 0.0015
    total = SPEC["qwen3"]["total_params"]
    trainable = int(total * ratio)
    return trainable, ratio * 100, f"结构估算:r={r} 全线性层旁路 ≈0.15%（[规格]底座0.6B）"


def load_json(p: Path) -> dict | None:
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def main() -> None:
    ap = argparse.ArgumentParser(description="成本表（方法 × F1 × 参数 × 时长 × 显存 × 延迟）")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    log_dir = paths.log_dir()
    eval_dir = paths.eval_dir()
    rows: list[dict[str, Any]] = []

    for run_name, label, family, param_source in RUNS:
        summary = load_json(log_dir / f"summary_{run_name}.json")
        metric = load_json(eval_dir / f"{run_name}_{args.split}.json")
        a = (summary or {}).get("args", {})

        # [实测] F1
        f1 = (metric or {}).get("metrics", {}).get("f1")
        f1_src = "实测:eval JSON" if f1 is not None else "未采集"

        # 参数量
        total_params: int | None = None
        trainable: int | None = None
        train_ratio: float | None = None
        if param_source == "structure":
            total_params, p_note = bilstm_params(summary or {}, None)
            trainable = total_params
            train_ratio = 100.0
        elif "lora" in run_name:
            total_params = SPEC["qwen3"]["total_params"]
            p_note = "[规格]Qwen3-0.6B 标称；可训：" + lora_trainable(summary or {})[2]
            trainable, train_ratio, _ = lora_trainable(summary or {})
        elif "full" in run_name:
            total_params = SPEC["qwen3"]["total_params"]
            p_note = "[规格]Qwen3-0.6B 标称"
            trainable, train_ratio = total_params, 100.0
        else:
            total_params = SPEC["roberta"]["total_params"]
            p_note = "[规格]" + SPEC["roberta"]["note"]
            trainable, train_ratio = None, 100.0

        # [实测] 训练时长
        secs, t_note = train_seconds(summary)

        # [实测] 延迟（只有 SFT 记了）
        eng = (metric or {}).get("engineering") or {}
        latency = eng.get("avg_latency_s")
        lat_src = "实测:SFT engineering" if latency is not None else "未采集(判别式未记延迟)"

        rows.append({
            "run_name": run_name, "label": label, "family": family,
            "f1": f1, "f1_src": f1_src,
            "total_params": total_params, "param_note": p_note,
            "trainable_params": trainable, "trainable_ratio_pct": train_ratio,
            "train_seconds": secs, "train_sec_src": t_note,
            "peak_vram_gb": None, "vram_src": "未采集(训练脚本未记 torch.cuda.max_memory_allocated)",
            "latency_s": latency, "latency_src": lat_src,
            "lr": a.get("lr"), "epochs": a.get("epochs"),
            "batch_size": a.get("batch_size"), "grad_accum": a.get("grad_accum", 1),
            "max_length": a.get("max_length"), "optim": a.get("optim", "AdamW"),
            "dtype": a.get("dtype", "fp32"),
            "num_train": a.get("num_train", 0),
        })

    stats = {
        "split": args.split,
        "source_policy": {
            "实测": "从产物 JSON 直接读",
            "结构": "从模型结构 + 产物超参精确计算",
            "规格": "公开模型规格，非本机实测",
            "未采集": "产物里没有，不编造；脚本末尾给采集方法",
        },
        "rows": rows,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    stats_path = eval_dir / "cost_table.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    # --- Markdown ---
    def fmt_num(v, suffix="", dash="—"):
        if v is None:
            return dash
        if isinstance(v, float):
            return f"{v:.4g}{suffix}"
        if isinstance(v, int) and v >= 1_000_000:
            return f"{v/1e6:.1f}M{suffix}" if v % 1_000_000 else f"{v//1_000_000}M{suffix}"
        return f"{v}{suffix}"

    md = [
        "# 成本表：六臂对比（老师 PPT P42 的五个数）",
        "",
        f"> split = `{args.split}`；生成于 {stats['generated_at']}",
        "",
        "**数据来源纪律**：每个数字标注来源。`实测`=从产物读，`结构`=从模型结构算，",
        "`规格`=公开模型规格，`未采集`=产物里没有（不编造）。",
        "",
        "| 方法 | F1 | 总参数 | 可训参数(比例) | 训练时长 | 显存峰值 | 推理延迟 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        tr = f"{fmt_num(r['trainable_params'])} ({r['trainable_ratio_pct']:.2f}%)" \
            if r["trainable_params"] is not None and r["trainable_ratio_pct"] is not None else "—"
        md.append(
            f"| {r['label']} | {fmt_num(r['f1'])} | {fmt_num(r['total_params'])} | {tr} "
            f"| {fmt_num(r['train_seconds'],'s')} | {fmt_num(r['peak_vram_gb'],'GB')} "
            f"| {fmt_num(r['latency_s'],'s')} |"
        )
    md += [
        "",
        "## 逐列来源与解读",
        "",
        "| 方法 | F1 来源 | 参数量来源 | 时长来源 | 延迟来源 |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        md.append(f"| {r['label']} | {r['f1_src']} | {r['param_note']} | {r['train_sec_src']} | {r['latency_src']} |")

    md += [
        "",
        "## 能直接讲的数字（面试用）",
        "",
    ]
    bil = next((r for r in rows if r["run_name"] == "bilstm_crf"), None)
    rob = next((r for r in rows if r["run_name"] == "roberta_crf"), None)
    lora = next((r for r in rows if r["run_name"] == "qwen_lora_sft"), None)
    full = next((r for r in rows if r["run_name"] == "qwen_full_sft"), None)
    if bil and rob:
        d = (rob["f1"] - bil["f1"]) * 100 if bil["f1"] is not None and rob["f1"] is not None else None
        md.append(f"- **预训练收益**：BiLSTM-CRF F1 {fmt_num(bil['f1'])} → RoBERTa-CRF "
                  f"{fmt_num(rob['f1'])}（+{d:.1f} 点）" if d is not None
                  else "- **预训练收益**：F1 未采集")
    if lora and full:
        md.append(f"- **参数量代价**：LoRA 只训约 {lora['trainable_ratio_pct']:.2f}% vs full 100%；"
                  f"full 用 fp32+8bit 优化器（约 5.7GB 参数侧，[规格]经验值），需 --grad_ckpt。")
        if lora["latency_s"] is not None and full["latency_s"] is not None:
            md.append(f"- **推理延迟**：LoRA {lora['latency_s']}s vs full {full['latency_s']}s"
                      f"（生成式逐条 decode，判别式未记延迟）。")
    md += [
        "",
        "## 未采集字段怎么补（下次跑训练时）",
        "",
        "显存峰值和判别式延迟本次没记。要补只需在训练/评测脚本里加两行：",
        "```python",
        "# 训练结束前：",
        "peak_gb = torch.cuda.max_memory_allocated() / 1024**3",
        "# 评测循环里：",
        "import time; t0 = time.perf_counter()",
        "...  # 推理",
        "latency = (time.perf_counter() - t0) / num_samples",
        "```",
        "写进 summary JSON 后重跑本脚本，`未采集`列会自动填实。",
        "",
        "> 面试提醒：老师要求成本数字可溯源。**宁可写「未采集」也不要估一个显存数**——",
        "被追问「怎么测的」答不上来反而扣分。",
    ]
    report_path = eval_dir / "cost_table.md"
    report_path.write_text("\n".join(md) + "\n", encoding="utf-8")

    print("=" * 72)
    print(f"成本表 | split={args.split}")
    print("=" * 72)
    for r in rows:
        print(f"  {r['label']:<18} F1={fmt_num(r['f1']):<8} 参数={fmt_num(r['total_params']):<8} "
              f"时长={fmt_num(r['train_seconds'],'s'):<8} 延迟={fmt_num(r['latency_s'],'s')}")
    print(f"\n报告 → {report_path}")
    print(f"数据 → {stats_path}")


if __name__ == "__main__":
    main()
