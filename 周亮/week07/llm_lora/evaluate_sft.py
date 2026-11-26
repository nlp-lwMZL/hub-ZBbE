"""
L4 评测：加载 LoRA adapter -> 批量生成 -> 容错解析 -> text.find 对齐 -> 统一评测。

【与判别式臂同口径】预测写成 common/metrics 的 pred_spans JSONL，喂同一个 evaluate()。
但生成式独有的工程指标必须一起报（比 F1 更能说明成本）：
  json_valid / repaired / noise_stripped / illegal_type / text_not_found / 平均延迟 / 生成 token 数。

【批量解码 + 左 padding（老师 PPT 的坑）】
decoder-only 模型必须左 padding + pad_token_id=eos_token_id，否则右 padding 会让模型
从 pad 位置续写。逐 token 解码在 T4 上每步要发 ~200 个 CUDA kernel，batch 8 能提速 4~6 倍。

用法：
  python llm_lora/evaluate_sft.py --adapter <adapter_dir> --split test
  python llm_lora/evaluate_sft.py --adapter <adapter_dir> --split validation --n 32  # 冒烟
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR / "llm_lora"))

from common import paths
from common.metrics import evaluate, write_predictions
from llm_utils import parse_entities, align_spans_to_text, VALID_TYPES


def parse_args():
    p = argparse.ArgumentParser(description="L4 Qwen3 SFT 评测")
    p.add_argument("--ckpt", default=None, help="SFT 权重目录：LoRA adapter 或 full 权重（自动识别）")
    p.add_argument("--adapter", default=None, help="[兼容旧名] 同 --ckpt")
    p.add_argument("--split", default="test", choices=["train", "validation", "test"])
    p.add_argument("--n", type=int, default=0, help="只评前 N 条；0=全量（冒烟用 32）")
    p.add_argument("--gen_batch", type=int, default=8, help="批量生成大小")
    p.add_argument("--max_new_tokens", type=int, default=192)
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="auto")
    return p.parse_args()


def resolve_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        try:
            torch.ones(1, device="mps"); return torch.device("mps")
        except (RuntimeError, AssertionError):
            pass
    return torch.device("cpu")


def pick_dtype(name, device):
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    if device.type == "cuda":
        try:
            if torch.cuda.is_bf16_supported(including_emulation=False):
                return torch.bfloat16
        except TypeError:
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
        return torch.float16
    return torch.float32


def build_prompt(tokenizer, text: str) -> str:
    from sft_data import SYSTEM_PROMPT
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user", "content": f"文本：{text}"}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )


def main():
    args = parse_args()
    device = resolve_device(args.device)
    dtype = pick_dtype(args.dtype, device)
    ckpt_arg = args.ckpt or args.adapter
    if not ckpt_arg:
        raise ValueError("需要 --ckpt（或旧名 --adapter）指定权重目录")
    ckpt = Path(ckpt_arg)
    if not ckpt.exists():
        raise FileNotFoundError(f"权重目录不存在：{ckpt}")

    # 自动识别：目录里有 adapter_config.json → LoRA adapter；否则 → full 整模型
    is_lora = (ckpt / "adapter_config.json").exists()
    mode = "LoRA" if is_lora else "full"
    print(f"加载 SFT 权重：{ckpt}（{mode} 模式）")

    tok = AutoTokenizer.from_pretrained(str(ckpt))
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"                      # decoder-only 必须左 padding

    if is_lora:
        from peft import PeftModel
        base_id = json.loads((ckpt / "adapter_config.json").read_text()).get("base_model_name_or_path")
        if not base_id:
            raise ValueError(f"adapter_config.json 里缺 base_model_name_or_path：{ckpt}")
        base = AutoModelForCausalLM.from_pretrained(base_id, dtype=dtype, trust_remote_code=False)
        model = PeftModel.from_pretrained(base, str(ckpt))
    else:
        # full 模式：权重目录本身就是完整模型，直接加载
        model = AutoModelForCausalLM.from_pretrained(str(ckpt), dtype=dtype, trust_remote_code=False)
    model.to(device); model.eval()
    model.config.pad_token_id = tok.pad_token_id

    records = json.loads((paths.get_data_dir() / f"{args.split}.json").read_text(encoding="utf-8"))
    if args.n > 0:
        records = records[:args.n]
    print(f"评测 {len(records)} 条 | device={device} dtype={dtype} | gen_batch={args.gen_batch}")

    stats: Counter = Counter()
    pred_by_id: dict[str, list[dict]] = {}
    total_gen_tokens = 0
    t0 = time.time()

    for start in range(0, len(records), args.gen_batch):
        chunk = records[start:start + args.gen_batch]
        prompts = [build_prompt(tok, r["text"]) for r in chunk]
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
                  max_length=640).to(device)
        with torch.no_grad():
            generated = model.generate(
                **enc,
                generation_config=GenerationConfig(
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False, num_beams=1,
                    pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
                ),
            )
        gen_only = generated[:, enc["input_ids"].shape[1]:]
        texts = tok.batch_decode(gen_only, skip_special_tokens=True)
        for rec, raw in zip(chunk, texts):
            total_gen_tokens += int(gen_only.shape[1])
            items = parse_entities(raw, stats)
            spans = align_spans_to_text(items, rec["text"], stats)
            pred_by_id[rec["id"]] = spans
        if (start // args.gen_batch) % 5 == 0:
            print(f"  {min(start + args.gen_batch, len(records))}/{len(records)}", flush=True)

    elapsed = time.time() - t0
    metrics = evaluate(records, pred_by_id)
    # run 名取权重目录的父目录名（.../checkpoints/<run>/adapter 或 .../<run>/full）
    run_name = ckpt.parent.name if ckpt.parent.name not in ("checkpoints", ".") else ckpt.name
    # 保证 full/lora 两臂产物不互相覆盖
    stem = f"{run_name}_{args.split}"
    pred_path = paths.pred_dir() / f"{stem}.jsonl"
    write_predictions(pred_path, pred_by_id)
    out_dir = paths.eval_dir()
    result = {
        "run_name": run_name,
        "method": f"Qwen3-0.6B {mode} SFT", "split": args.split,
        "num_samples": len(records), "metrics": metrics,
        "engineering": {
            "json_valid": stats["json_valid"],
            "repaired": stats["repaired"],
            "noise_stripped": stats["noise_stripped"],
            "illegal_type": stats["illegal_type"],
            "text_not_found": stats["text_not_found"],
            "empty_or_no_json": stats["empty_or_no_json"],
            "unrepairable": stats["unrepairable"],
            "avg_latency_s": round(elapsed / max(1, len(records)), 3),
            "avg_gen_tokens": round(total_gen_tokens / max(1, len(records)), 1),
        },
        "ckpt": str(ckpt), "mode": mode, "pred_path": str(pred_path),
    }
    res_path = out_dir / f"{stem}.json"
    res_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n实体 P={metrics['precision']} R={metrics['recall']} F1={metrics['f1']} "
          f"| type_macro={metrics['type_macro_f1']}")
    print(f"工程指标: {dict(stats)} | 平均延迟 {elapsed/max(1,len(records)):.2f}s/条")
    print(f"  预测 {pred_path}\n  指标 {res_path}")


if __name__ == "__main__":
    main()
