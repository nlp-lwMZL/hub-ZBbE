# -*- coding: utf-8 -*-
"""
加载 SFT checkpoint（LoRA / 全量）评测事件抽取，并调用统一评测脚本出指标

教学重点：
  1. **自动识别 checkpoint 类型**：目录里有 adapter_config.json → LoRA（先加载基座再 merge），
     否则 → 全量微调 checkpoint。教学项目里两种模型共用一套评估代码，避免"两套代码两套口径"。
  2. **评测必须用同一批样本**：这里复用 event_data.sample_rows（seed=42 + 排序后 shuffle），
     和 llm_extract.py / 判别式模型得到的是同一批 dev 句子。
  3. **指标不自己算**：生成预测 JSONL 后直接 subprocess 调 common/evaluate_events.py，
     保证和 BERT/BiLSTM/规则等方案是同一把尺子（trigger/argument/event 三级 + type_macro）。
  4. **解析失败不删行**：预测里写 events=[] 保留在原文件，分母不变（删了 F1 会虚高）。

使用方式：
  python llm_lora/evaluate_sft.py                                # 评 LoRA（默认 200 条）
  python llm_lora/evaluate_sft.py --n 930                        # 全量 dev
  python llm_lora/evaluate_sft.py --ckpt_dir outputs/llm/ckpt/sft_full_ckpt  # 评全量微调

依赖：pip install torch transformers peft
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]   # 项目根目录
sys.path.insert(0, str(ROOT / "common"))        # 公共模块
sys.path.insert(0, str(ROOT / "llm_zero_shot"))  # 生成式链路共用：event_data / event_prompt / local_generate
sys.path.insert(0, str(Path(__file__).resolve().parent))  # llm_lora 自己的 model_paths.py

from log_utils import EvalLogger  # noqa: E402
from event_data import (  # noqa: E402
    OUTPUT_LLM_DIR, ROOT, SEED, load_jsonl, load_schema, make_pred_row,
    sample_rows, write_eval_ids, write_gold_subset,
)
from event_prompt import build_messages, parse_model_output  # noqa: E402
from local_generate import generate_batch, resolve_torch_dtype  # noqa: E402
from model_paths import QWEN_MODEL_DIR, resolve_model_path  # noqa: E402

try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

DEFAULT_CKPT = ROOT / "outputs" / "llm" / "ckpt" / "sft_adapter"

# ══════════════════════════════════════════════════════════════════════════════
# 模型加载
# ══════════════════════════════════════════════════════════════════════════════

def load_model(model_path: str, ckpt_dir: str, dtype: torch.dtype, device: torch.device):
    """加载 checkpoint，返回 (model, tokenizer, ckpt_type)。"""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ckpt_path = Path(ckpt_dir)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"checkpoint 不存在：{ckpt_path}\n"
            f"请先跑训练：python llm_lora/train_sft.py"
        )
    is_lora = (ckpt_path / "adapter_config.json").exists()
    if is_lora and not PEFT_AVAILABLE:
        raise ImportError("加载 LoRA adapter 需要 peft：pip install peft>=0.14.0")

    if is_lora:
        # LoRA 只存了"低秩增量"，必须连同基座一起加载再 merge 回一个普通模型，
        # 之后的生成代码就和全量微调完全一样（不需要特殊处理）
        print(f"[加载] LoRA adapter: {ckpt_path}\n        基座: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"   # 批量生成必须左 padding（见 local_generate.py 说明）
        base = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, trust_remote_code=True)
        model = PeftModel.from_pretrained(base, str(ckpt_path)).merge_and_unload()
        ckpt_type = "lora"
    else:
        print(f"[加载] 全量微调 checkpoint: {ckpt_path}")
        tokenizer = AutoTokenizer.from_pretrained(str(ckpt_path), trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(
            str(ckpt_path), torch_dtype=dtype, trust_remote_code=True)
        ckpt_type = "full_ft"

    model = model.to(device).eval()
    return model, tokenizer, ckpt_type

# ══════════════════════════════════════════════════════════════════════════════
# 统一评测 + 汇总表
# ══════════════════════════════════════════════════════════════════════════════

def run_shared_evaluation(gold_path: Path, pred_path: Path, metrics_path: Path) -> dict:
    """调用 common/evaluate_events.py（全项目唯一口径），返回其结果字典。"""
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, str(ROOT / "common" / "evaluate_events.py"),
                    "--gold", str(gold_path), "--pred", str(pred_path),
                    "--json", str(metrics_path), "--quiet"],
                   check=True, cwd=str(ROOT))
    with open(metrics_path, "r", encoding="utf-8") as f:
        return json.load(f)

def print_compare_table(skip_keywords=("probe", "smoke", "mock", "dry")):
    """扫描 outputs/metrics/llm/*.json 打印一张横向对比表（PPT 素材）。"""
    metrics_dir = ROOT / "outputs" / "metrics" / "llm"
    rows = []
    for path in sorted(metrics_dir.glob("*.json")):
        tag = path.stem
        if any(k in tag for k in skip_keywords):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                m = json.load(f)
            rows.append((tag, m["trigger"]["f1"], m["argument"]["f1"],
                         m["event_exact"]["f1"], m.get("type_macro_f1", 0)))
        except (json.JSONDecodeError, KeyError):
            continue
    if not rows:
        return
    print("\n横向对比（同一批 dev 样本，统一评测口径）")
    print(f"  {'方法':<26}{'trigger':>9}{'argument':>10}{'event':>8}{'type_macro':>11}")
    for tag, tr, ar, ev, tm in rows:
        print(f"  {tag:<26}{tr:>9.4f}{ar:>10.4f}{ev:>8.4f}{tm:>11.4f}")

def parse_args():
    p = argparse.ArgumentParser(description="SFT checkpoint 事件抽取评估")
    p.add_argument("--model_path", default=str(resolve_model_path(QWEN_MODEL_DIR)))
    p.add_argument("--ckpt_dir", default=str(DEFAULT_CKPT))
    p.add_argument("--data", default=None, help="金标 JSONL（默认 dev_extract.jsonl）")
    p.add_argument("--n", default=200, type=int, help="评测样本数（默认 200，与 API 实验同批）")
    p.add_argument("--seed", default=SEED, type=int)
    p.add_argument("--max_new_tokens", default=320, type=int)
    p.add_argument("--max_length", default=1280, type=int)
    p.add_argument("--gen_batch", default=8, type=int,
                   help="批量解码 batch（Windows 上逐条解码会被 kernel 启动开销拖死）")
    p.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    p.add_argument("--tag", default="", help="输出后缀，默认 sft_{lora|full}_n{n}")
    return p.parse_args()

def main():
    args = parse_args()
    # 纯评测方法用 EvalLogger：写 outputs/logs/llm_lora_eval_eval.json + _console.log
    eval_logger = EvalLogger(method="llm_lora_eval", args=args, root=ROOT,
                             notes="LoRA SFT 模型评测（生成式链路）")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = resolve_torch_dtype(device, args.dtype)
    print(f"设备: {device}（dtype={torch_dtype}）")

    schema = load_schema()
    gold_in = Path(args.data) if args.data else (ROOT / "data" / "processed" / "dev_extract.jsonl")
    all_rows = [json.loads(line) for line in open(gold_in, encoding="utf-8") if line.strip()]
    rows = sample_rows(all_rows, args.n, args.seed)

    model, tokenizer, ckpt_type = load_model(
        args.model_path, args.ckpt_dir, torch_dtype, device)
    tag = args.tag or f"sft_{'lora' if ckpt_type == 'lora' else 'full'}_n{len(rows)}"

    OUTPUT_LLM_DIR.mkdir(parents=True, exist_ok=True)
    gold_subset = OUTPUT_LLM_DIR / f"gold_n{len(rows)}.jsonl"
    pred_path = OUTPUT_LLM_DIR / f"pred_{tag}.jsonl"
    raw_path = OUTPUT_LLM_DIR / f"raw_{tag}.jsonl"
    report_path = OUTPUT_LLM_DIR / f"report_{tag}.json"
    metrics_path = ROOT / "outputs" / "metrics" / "llm" / f"{tag}.json"
    write_eval_ids(OUTPUT_LLM_DIR / "eval_ids.json", rows, args.seed)
    write_gold_subset(gold_subset, rows)
    print(f"评测样本 {len(rows)} 条（seed={args.seed}，与 API / 判别式同一批）\n")

    # ── 推理（批量解码，摊薄 Windows 上的 kernel 启动开销）────────────────────
    stat_keys = ("empty_reply", "json_ok", "repaired", "noise_stripped", "type_invalid",
                 "role_invalid", "trigger_not_in_source", "arg_not_in_source")
    stats = {k: 0 for k in stat_keys}
    latencies, n_ok = [], 0
    t0 = time.time()
    with open(pred_path, "w", encoding="utf-8") as f_pred, \
            open(raw_path, "w", encoding="utf-8") as f_raw:
        for start in range(0, len(rows), args.gen_batch):
            chunk = rows[start:start + args.gen_batch]
            chunk_msgs = [build_messages(r["text"], schema, few_shot=False) for r in chunk]
            outs = generate_batch(model, tokenizer, chunk_msgs, device,
                                  max_new_tokens=args.max_new_tokens,
                                  batch_size=args.gen_batch,
                                  max_length=args.max_length)
            for k, (row, out) in enumerate(zip(chunk, outs)):
                i = start + k + 1
                events, info = parse_model_output(out["text"], row["text"], schema)
                for key in stat_keys:
                    stats[key] += int(info.get(key, 0) or 0)
                latencies.append(out["latency_s"])
                n_ok += int(info["json_ok"])
                f_pred.write(json.dumps(make_pred_row(row, events), ensure_ascii=False) + "\n")
                f_raw.write(json.dumps({"id": row.get("id"), "sent_id": row.get("sent_id"),
                                        "raw": out["text"], "events": events},
                                       ensure_ascii=False) + "\n")
            if i % 10 < args.gen_batch or i == len(rows):
                print(f"  [{i:3d}/{len(rows)}] 合法 {n_ok} 条，"
                      f"平均 {(time.time() - t0) / i:.2f}s/条", flush=True)

    total_s = time.time() - t0
    lat_sorted = sorted(latencies)
    report = {
        "tag": tag, "backend": "sft", "ckpt_type": ckpt_type,
        "ckpt_dir": str(Path(args.ckpt_dir).resolve()),
        "model_path": str(Path(args.model_path).resolve()),
        "n": len(rows), "seed": args.seed,
        "json_valid_rate": round(100.0 * n_ok / len(rows), 2) if rows else 0,
        "parse_fail": len(rows) - n_ok,
        **{k: stats[k] for k in stat_keys if k != "json_ok"},
        "latency_mean_s": round(sum(latencies) / len(rows), 3) if rows else 0,
        "latency_p50_s": round(lat_sorted[len(lat_sorted) // 2], 3) if rows else 0,
        "total_s": round(total_s, 1),
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    res = run_shared_evaluation(gold_subset, pred_path, metrics_path)

    print("\n" + "=" * 78)
    print(f"结果：{tag}（{ckpt_type}，{len(rows)} 条）")
    print("=" * 78)
    for key, name in (("trigger", "trigger"), ("argument", "argument"), ("event_exact", "event")):
        m = res[key]
        print(f"  {name:<8} F1={m['f1']:.4f}  (P={m['precision']:.4f} R={m['recall']:.4f})")
    print(f"  type_macro_f1 = {res['type_macro_f1']:.4f}")
    print(f"  JSON 合法率: {report['json_valid_rate']}% | 解析失败 {report['parse_fail']} 条 | "
          f"修复 {report['repaired']} 条 | 空回复 {report['empty_reply']} 条")
    print(f"  幻觉论元 {report['arg_not_in_source']} 次 | "
          f"平均延迟 {report['latency_mean_s']}s/条 | 总耗时 {total_s:.0f}s")

    # 最难的 3 个事件类型（定位瓶颈，PPT 的 bad case 分析用）
    worst = sorted(res.get("per_type", {}).items(), key=lambda kv: kv[1]["f1"])[:3]
    if worst:
        print("  最难类型: " + " | ".join(
            f"{t}(F1={m['f1']:.3f}, 金标{m['tp'] + m['fn']}条)" for t, m in worst))

    print(f"\n预测 → {pred_path}\n统计 → {report_path}\n统一评测 → {metrics_path}")
    eval_logger.finalize(
        metrics={k: res[k]["f1"] for k in ("trigger", "argument", "event_exact")},
        type_macro_f1=res.get("type_macro_f1"), report=report,
        ckpt_dir=str(getattr(args, "ckpt_dir", "")))
    print_compare_table()

if __name__ == "__main__":
    main()
