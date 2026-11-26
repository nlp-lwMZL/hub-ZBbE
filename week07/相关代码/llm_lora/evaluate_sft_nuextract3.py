# -*- coding: utf-8 -*-
"""
NuExtract3 SFT checkpoint 事件抽取评测

与 evaluate_sft.py 的差异：
  1. 用 AutoProcessor + AutoModelForImageTextToText 加载（NuExtract3 是 VLM）。
  2. prompt 必须带 mode="structured" + template + enable_thinking=False，
     否则模型不会输出结构化 JSON。
  3. 生成后直接尝试 json.loads，失败时做轻量清洗（NuExtract3 偶尔会包裹
     在 markdown code fence 或额外说明文字中）。

使用方式：
  python llm_lora/evaluate_sft_nuextract3.py                     # 默认评 LoRA，200 条
  python llm_lora/evaluate_sft_nuextract3.py --n 930             # 全量 dev
  python llm_lora/evaluate_sft_nuextract3.py --ckpt_dir outputs/llm/ckpt/sft_full_ckpt_nuextract3
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "common"))
sys.path.insert(0, str(ROOT / "llm_zero_shot"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from log_utils import EvalLogger
from event_data import (
    OUTPUT_LLM_DIR, ROOT, SEED, load_jsonl, load_schema, make_pred_row,
    sample_rows, write_eval_ids, write_gold_subset,
)
from model_paths import NUEXTRACT_MODEL_DIR, resolve_model_path
from local_generate import resolve_torch_dtype

try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

DEFAULT_CKPT = ROOT / "outputs" / "llm" / "ckpt" / "sft_adapter_nuextract3"


# ══════════════════════════════════════════════════════════════════════════════
# 模板（必须与训练脚本保持一致）
# ══════════════════════════════════════════════════════════════════════════════

def build_template_str() -> str:
    """与 train_sft_nuextract3.py 里使用的 template 完全相同。"""
    template_obj = {
        "events": [
            {
                "type": "verbatim-string",
                "trigger": "verbatim-string",
                "arguments": [
                    {"role": "verbatim-string", "text": "verbatim-string"}
                ],
            }
        ]
    }
    return json.dumps(template_obj, ensure_ascii=False)


# ══════════════════════════════════════════════════════════════════════════════
# 模型加载
# ══════════════════════════════════════════════════════════════════════════════

def load_nuextract3(model_path: str, ckpt_dir: str, dtype: torch.dtype,
                    device: torch.device):
    """返回 (model, processor, ckpt_type)。"""
    from transformers import AutoModelForImageTextToText, AutoProcessor

    ckpt_path = Path(ckpt_dir)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"checkpoint 不存在：{ckpt_path}\n"
            f"请先训练：python llm_lora/train_sft_nuextract3.py"
        )
    is_lora = (ckpt_path / "adapter_config.json").exists()
    if is_lora and not PEFT_AVAILABLE:
        raise ImportError("加载 LoRA adapter 需要 peft：pip install peft>=0.14.0")

    if is_lora:
        print(f"[加载] NuExtract3 LoRA adapter: {ckpt_path}\n        基座: {model_path}")
        processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True)
        base = AutoModelForImageTextToText.from_pretrained(
            model_path, dtype=dtype, trust_remote_code=True)
        model = PeftModel.from_pretrained(base, str(ckpt_path)).merge_and_unload()
        ckpt_type = "lora"
    else:
        print(f"[加载] NuExtract3 全量微调 checkpoint: {ckpt_path}")
        processor = AutoProcessor.from_pretrained(
            str(ckpt_path), trust_remote_code=True)
        model = AutoModelForImageTextToText.from_pretrained(
            str(ckpt_path), dtype=dtype, trust_remote_code=True)
        ckpt_type = "full_ft"

    model = model.to(device).eval()
    return model, processor, ckpt_type


# ══════════════════════════════════════════════════════════════════════════════
# 单条推理
# ══════════════════════════════════════════════════════════════════════════════

def run_nuextract_single(model, processor, text: str, template_str: str,
                         device: torch.device, max_new_tokens: int = 320) -> dict:
    """
    对一条公告句做结构化抽取。
    NuExtract3 的 processor.apply_chat_template 直接返回模型输入，
    不需要先 encode 成 input_ids 再手动拼 label。
    """
    messages = [
        {"role": "user", "content": [{"type": "text", "text": f"句子：{text}"}]}
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        mode="structured",
        template=template_str,
        enable_thinking=False,
    ).to(device)

    t0 = time.time()
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False)
    latency = time.time() - t0

    # 只取新生成的部分
    generated_ids = generated_ids[:, inputs.input_ids.shape[1]:]
    raw = processor.batch_decode(
        generated_ids, skip_special_tokens=True,
        clean_up_tokenization_spaces=False)[0].strip()

    return {"text": raw, "latency_s": latency}


# ══════════════════════════════════════════════════════════════════════════════
# 输出解析（NuExtract3 可能返回纯 JSON，也可能包裹在 code fence 中）
# ══════════════════════════════════════════════════════════════════════════════

def parse_nuextract_output(raw: str) -> tuple[list, dict]:
    """
    返回 (events, info)。
    info 的键与 evaluate_sft.py 的 parse_model_output 保持兼容（至少含 json_ok）。
    """
    info = {"json_ok": 0, "empty_reply": 0}
    if not raw or not raw.strip():
        info["empty_reply"] = 1
        return [], info

    text = raw.strip()
    # 去掉 markdown code fence
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text).strip()
    # 去掉可能的 </think> 前缀（enable_thinking=False 时一般不会出现）
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()

    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        # 尝试提取第一个 {...} 块
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return [], info
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return [], info

    info["json_ok"] = 1
    events = obj.get("events", []) if isinstance(obj, dict) else []
    if not isinstance(events, list):
        events = []
    # 只保留结构合法的 event
    clean = []
    for ev in events:
        if isinstance(ev, dict) and "type" in ev and "trigger" in ev:
            clean.append({
                "type": ev.get("type", ""),
                "trigger": ev.get("trigger", ""),
                "arguments": ev.get("arguments", []) if isinstance(
                    ev.get("arguments"), list) else [],
            })
    return clean, info


# ══════════════════════════════════════════════════════════════════════════════
# 统一评测 + 对比表（直接复用 evaluate_sft.py 的逻辑）
# ══════════════════════════════════════════════════════════════════════════════

def run_shared_evaluation(gold_path: Path, pred_path: Path,
                          metrics_path: Path) -> dict:
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [sys.executable, str(ROOT / "common" / "evaluate_events.py"),
         "--gold", str(gold_path), "--pred", str(pred_path),
         "--json", str(metrics_path), "--quiet"],
        check=True, cwd=str(ROOT),
    )
    with open(metrics_path, "r", encoding="utf-8") as f:
        return json.load(f)


def print_compare_table(skip_keywords=("probe", "smoke", "mock", "dry")):
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


# ══════════════════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="NuExtract3 SFT 事件抽取评估")
    p.add_argument("--model_path",
                   default=str(resolve_model_path(NUEXTRACT_MODEL_DIR)))
    p.add_argument("--ckpt_dir", default=str(DEFAULT_CKPT))
    p.add_argument("--data", default=None,
                   help="金标 JSONL（默认 data/processed/dev_extract.jsonl）")
    p.add_argument("--n", default=200, type=int,
                   help="评测样本数（默认 200，与 API 实验同批）")
    p.add_argument("--seed", default=SEED, type=int)
    p.add_argument("--max_new_tokens", default=320, type=int)
    p.add_argument("--dtype", default="auto",
                   choices=["auto", "bf16", "fp16", "fp32"])
    p.add_argument("--tag", default="",
                   help="输出后缀，默认 sft_{lora|full}_nuextract3_n{n}")
    return p.parse_args()


def main():
    args = parse_args()
    eval_logger = EvalLogger(
        method="llm_lora_eval_nuextract3", args=args, root=ROOT,
        notes="NuExtract3 SFT 模型评测（VLM processor 链路）")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = resolve_torch_dtype(device, args.dtype)
    print(f"设备: {device}（dtype={torch_dtype}）")

    schema = load_schema()
    gold_in = Path(args.data) if args.data else (
        ROOT / "data" / "processed" / "dev_extract.jsonl")
    all_rows = [json.loads(line) for line in open(gold_in, encoding="utf-8")
                if line.strip()]
    rows = sample_rows(all_rows, args.n, args.seed)

    model, processor, ckpt_type = load_nuextract3(
        args.model_path, args.ckpt_dir, torch_dtype, device)
    tag = args.tag or f"sft_{ckpt_type}_nuextract3_n{len(rows)}"

    template_str = build_template_str()

    OUTPUT_LLM_DIR.mkdir(parents=True, exist_ok=True)
    gold_subset = OUTPUT_LLM_DIR / f"gold_n{len(rows)}.jsonl"
    pred_path = OUTPUT_LLM_DIR / f"pred_{tag}.jsonl"
    raw_path = OUTPUT_LLM_DIR / f"raw_{tag}.jsonl"
    report_path = OUTPUT_LLM_DIR / f"report_{tag}.json"
    metrics_path = ROOT / "outputs" / "metrics" / "llm" / f"{tag}.json"

    write_eval_ids(OUTPUT_LLM_DIR / "eval_ids.json", rows, args.seed)
    write_gold_subset(gold_subset, rows)
    print(f"评测样本 {len(rows)} 条（seed={args.seed}，与 API / 判别式同一批）\n")

    # ── 逐条推理（NuExtract3 的 processor 调用较重，不做批量解码）────────
    latencies, n_ok, n_empty = [], 0, 0
    t0 = time.time()
    with open(pred_path, "w", encoding="utf-8") as f_pred, \
            open(raw_path, "w", encoding="utf-8") as f_raw:
        for i, row in enumerate(rows, 1):
            out = run_nuextract_single(
                model, processor, row["text"], template_str, device,
                max_new_tokens=args.max_new_tokens)
            events, info = parse_nuextract_output(out["text"])
            latencies.append(out["latency_s"])
            n_ok += int(info["json_ok"])
            n_empty += int(info["empty_reply"])

            f_pred.write(json.dumps(make_pred_row(row, events),
                                    ensure_ascii=False) + "\n")
            f_raw.write(json.dumps({
                "id": row.get("id"), "sent_id": row.get("sent_id"),
                "raw": out["text"], "events": events,
            }, ensure_ascii=False) + "\n")

            if i % 10 == 0 or i == len(rows):
                print(f"  [{i:3d}/{len(rows)}] 合法 {n_ok} 条，"
                      f"平均 {(time.time() - t0) / i:.2f}s/条", flush=True)

    total_s = time.time() - t0
    lat_sorted = sorted(latencies)

    report = {
        "tag": tag, "backend": "sft", "base_model": "NuExtract3",
        "ckpt_type": ckpt_type,
        "ckpt_dir": str(Path(args.ckpt_dir).resolve()),
        "model_path": str(Path(args.model_path).resolve()),
        "n": len(rows), "seed": args.seed,
        "json_valid_rate": round(100.0 * n_ok / len(rows), 2) if rows else 0,
        "parse_fail": len(rows) - n_ok,
        "empty_reply": n_empty,
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
    for key, name in (("trigger", "trigger"), ("argument", "argument"),
                      ("event_exact", "event")):
        m = res[key]
        print(f"  {name:<8} F1={m['f1']:.4f}  "
              f"(P={m['precision']:.4f} R={m['recall']:.4f})")
    print(f"  type_macro_f1 = {res['type_macro_f1']:.4f}")
    print(f"  JSON 合法率: {report['json_valid_rate']}% | "
          f"解析失败 {report['parse_fail']} 条 | "
          f"空回复 {report['empty_reply']} 条")
    print(f"  平均延迟 {report['latency_mean_s']}s/条 | 总耗时 {total_s:.0f}s")

    worst = sorted(res.get("per_type", {}).items(),
                   key=lambda kv: kv[1]["f1"])[:3]
    if worst:
        print("  最难类型: " + " | ".join(
            f"{t}(F1={m['f1']:.3f}, 金标{m['tp'] + m['fn']}条)"
            for t, m in worst))

    print(f"\n预测 → {pred_path}\n统计 → {report_path}\n统一评测 → {metrics_path}")
    eval_logger.finalize(
        metrics={k: res[k]["f1"] for k in ("trigger", "argument", "event_exact")},
        type_macro_f1=res.get("type_macro_f1"), report=report,
        ckpt_dir=str(getattr(args, "ckpt_dir", "")))
    print_compare_table()


if __name__ == "__main__":
    main()