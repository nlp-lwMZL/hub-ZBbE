"""
Qwen3 + LoRA / 全量微调 推理评测：在测试集上算 Accuracy / Macro F1。

要点
----
1. 推理必须和训练用同一个 chat 模板 + 同一个 enable_thinking=False，
   否则格式错位，准确率会莫名掉一半。
2. 解析失败的条数要单独统计（pred=-1），它决定要不要加约束解码。

用法
----
python lora/eval_lora.py                       # 默认 adapter + 测试集
python lora/eval_lora.py --n 200 --split test
python lora/eval_lora.py --ckpt output/lora/full_ft_ckpt   # 全量微调权重
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from sklearn.metrics import classification_report, f1_score

CODE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.label import CLS_NUM_LABELS  # noqa: E402
from train_lora import LABELS, MODEL_DIR, OUT_DIR, SYSTEM, load_split, parse_label  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(OUT_DIR / "lora_adapter"),
                    help="LoRA adapter 目录，或全量微调的完整权重目录")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"   # 生成时必须左填充，否则新 token 位置错乱

    ckpt_dir = Path(args.ckpt)
    if (ckpt_dir / "adapter_config.json").exists():
        from peft import PeftModel
        base = AutoModelForCausalLM.from_pretrained(str(MODEL_DIR), torch_dtype=dtype)
        model = PeftModel.from_pretrained(base, str(ckpt_dir)).to(device).eval()
        kind = "lora"
    else:
        model = AutoModelForCausalLM.from_pretrained(str(ckpt_dir), torch_dtype=dtype).to(device).eval()
        kind = "full_ft"
    print(f"[eval] 加载方式：{kind}（{ckpt_dir}）", flush=True)

    samples = load_split(args.split, args.n)
    prompts = [
        tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": d["text"]}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        for d in samples
    ]

    preds, raws = [], []
    for i in range(0, len(prompts), args.batch_size):
        chunk = prompts[i: i + args.batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                 do_sample=False, pad_token_id=tok.pad_token_id)
        gen = out[:, enc["input_ids"].shape[1]:]
        for text in tok.batch_decode(gen, skip_special_tokens=True):
            raws.append(text)
            preds.append(parse_label(text))
        print(f"  {min(i + args.batch_size, len(prompts))}/{len(prompts)}", flush=True)

    y_true = [d["cls_label"] for d in samples]
    valid = [(t, p) for t, p in zip(y_true, preds) if p >= 0]
    acc = sum(t == p for t, p in valid) / max(1, len(valid))
    macro_f1 = f1_score([t for t, _ in valid], [p for _, p in valid],
                        average="macro", labels=list(range(CLS_NUM_LABELS)),
                        zero_division=0)
    report = classification_report(
        [t for t, _ in valid], [p for _, p in valid],
        labels=list(range(CLS_NUM_LABELS)), target_names=LABELS,
        digits=4, zero_division=0)

    metrics = {
        "method": f"qwen3_{kind}",
        "split": args.split,
        "n": len(samples),
        "parse_fail": len(samples) - len(valid),
        "acc": acc,
        "macro_f1": macro_f1,
    }
    print(f"[{kind}] n={len(samples)} 解析失败={metrics['parse_fail']} "
          f"acc={acc:.4f} macro_f1={macro_f1:.4f}")
    print(report)

    out_dir = OUT_DIR if args.out is None else Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{kind}_{args.split}"
    (out_dir / f"metrics_{tag}.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / f"report_{tag}.txt").write_text(report, encoding="utf-8")
    with (out_dir / f"preds_{tag}.jsonl").open("w", encoding="utf-8") as f:
        for d, p, r in zip(samples, preds, raws):
            f.write(json.dumps({"text": d["text"][:120], "true": d["cls_label"],
                                "pred": p, "raw": r}, ensure_ascii=False) + "\n")
    print(f"→ {out_dir}")


if __name__ == "__main__":
    main()
