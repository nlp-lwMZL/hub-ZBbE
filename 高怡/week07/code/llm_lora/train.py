"""Qwen3-0.6B + LoRA 事件抽取微调（生成式，无 think，输出 event JSON）。

与判别式版本（extract/roberta）的差异
------------------------------------
- 输入：整条公告文本 + system 提示（含事件类型/槽位清单）
- 输出：仅一个 JSON ``{"event_type","trigger","arguments":[{"role","text"}]}``
  **不含 start/end**；位置在解码后用 ``align.py`` 回填原文得到
- loss：system/user 段 label=-100，只在 assistant 的 JSON + <|im_end|> 上计算
- 无 think：prompt 里已包含空的 think 块（``enable_thinking=False``），生成直接是 JSON

所有超参可配：LoRA 秩/alpha/dropout/目标模块、lr、warmup、wd、累积步数、裁剪、
序列长度、生成长度、评估条数等。

示例
----
# 冒烟（CPU，几分钟内可跑完）
python -m extract.llm_lora.train --max-samples 32 --max-steps 4 --eval-limit 8 \
    --max-seq-len 1024 --batch-size 1 --grad-accum-steps 1

# 小规模真实训练
python -m extract.llm_lora.train --max-samples 2000 --epochs 1 --eval-limit 200
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.data import load_records  # noqa: E402
from common.label import TRIGGER_LABEL  # noqa: E402
from common.logging_utils import setup_logger  # noqa: E402
from common.metrics import (  # noqa: E402
    compute_event_exact_match,
    compute_metrics,
    compute_type_metrics,
)
from common.schema import EVENT_SCHEMA  # noqa: E402

try:
    from .align import (
        attach_spans,
        gold_spans,
        spans_of,
        to_metric_event,
        gold_event_to_metric,
    )
    from .data import IGNORE_INDEX, SFTDataset, collate_fn, encode_prompt, get_im_end_id, render_prompt
    from .prompt import parse_model_output, normalize_prediction
except ImportError:  # 直接 python extract/llm_lora/train.py
    from align import (  # type: ignore
        attach_spans,
        gold_spans,
        spans_of,
        to_metric_event,
        gold_event_to_metric,
    )
    from data import IGNORE_INDEX, SFTDataset, collate_fn, encode_prompt, get_im_end_id, render_prompt  # type: ignore
    from prompt import parse_model_output, normalize_prediction  # type: ignore

DEFAULT_MODEL_DIR = PROJECT_ROOT / "pretrain_models" / "Qwen3-0___6B"


# ------------------------------------------------------------------ 评估


def _strip_think(raw: str) -> str:
    import re
    return re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()


@torch.inference_mode()
def evaluate(model, tokenizer, records, args, device, logger) -> tuple[dict, list[dict]]:
    model.eval()
    model.config.use_cache = True
    gold_spans_list, pred_spans_list = [], []
    gold_types, pred_types = [], []
    gold_events, pred_events = [], []
    rows: list[dict] = []
    n_parse_fail = n_no_event = n_unmatched = 0

    t0 = time.time()
    for i, rec in enumerate(records):
        text = rec.get("text") or ""
        prompt_ids = encode_prompt(
            tokenizer, text, args.max_seq_len - args.gen_max_new_tokens
        )
        inputs = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=args.amp and device.type == "cpu"):
            out = model.generate(
                input_ids=inputs,
                attention_mask=torch.ones_like(inputs),
                max_new_tokens=args.gen_max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=get_im_end_id(tokenizer),
            )
        raw = tokenizer.decode(out[0, len(prompt_ids):], skip_special_tokens=True)
        raw = _strip_think(raw)

        obj = parse_model_output(raw)
        if obj is None:
            n_parse_fail += 1
            pred = {"type": "", "trigger": "", "arguments": []}
        else:
            pred = normalize_prediction(obj)
        if pred["type"] == "无":
            n_no_event += 1

        spanned = attach_spans(pred, text)
        n_unmatched += len(spanned["unmatched"])

        g_spans = gold_spans(rec.get("event"))
        p_spans = spans_of(spanned)
        gold_spans_list.append(g_spans)
        pred_spans_list.append(p_spans)
        gold_types.append((rec.get("event") or {}).get("type", ""))
        pred_types.append(spanned["type"])
        gold_events.append(gold_event_to_metric(rec.get("event")))
        pred_events.append(to_metric_event(spanned))
        rows.append({
            "source": rec.get("source", ""),
            "slice_id": rec.get("slice_id", ""),
            "gold": rec.get("event"),
            "pred": spanned,
            "raw": raw[:500],
        })

        if (i + 1) % args.log_every_eval == 0:
            speed = (i + 1) / max(1e-6, time.time() - t0)
            logger.info("    评估 %d/%d | %.2f 条/秒 | 解析失败 %d",
                        i + 1, len(records), speed, n_parse_fail)

    report = compute_metrics(gold_spans_list, pred_spans_list)
    report.event_type = compute_type_metrics(gold_types, pred_types)
    report.event_match = compute_event_exact_match(gold_events, pred_events)

    stats = {
        "n": len(records),
        "parse_fail": n_parse_fail,
        "no_event": n_no_event,
        "unmatched_spans": n_unmatched,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    model.train()
    model.config.use_cache = False
    return {"report": report, "stats": stats}, rows


def score_of(report, monitor: str) -> float:
    d = report.as_dict()
    if monitor == "event_exact":
        return (d.get("event_match") or {}).get("accuracy", 0.0)
    if monitor == "f1":
        return report.f1
    if monitor == "type_macro_f1":
        return (d.get("event_type") or {}).get("macro_f1", 0.0)
    raise ValueError(monitor)


# ------------------------------------------------------------------ 参数


def parse_args():
    p = argparse.ArgumentParser(description="Qwen3 + LoRA 事件抽取微调")
    p.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    p.add_argument("--train-file", default=str(PROJECT_ROOT / "data/dataset/split/train.jsonl"))
    p.add_argument("--val-file", default=str(PROJECT_ROOT / "data/dataset/split/val.jsonl"))
    p.add_argument("--max-seq-len", type=int, default=2048)
    p.add_argument("--max-samples", type=int, default=0, help="训练集只取前 N 条（--max-per-class 启用时忽略）")
    p.add_argument("--max-per-class", type=int, default=0,
                   help="训练集按事件类型分层截断，每类最多 N 条；稀有类不足 N 条则全量保留")
    p.add_argument("--eval-limit", type=int, default=200, help="评估集取前 N 条（--eval-per-class 启用时忽略）")
    p.add_argument("--eval-per-class", type=int, default=0,
                   help="评估集按事件类型分层截断，每类最多 N 条；保证稀有类进入评估")

    # LoRA
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--lora-target", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")

    # 训练
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum-steps", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--warmup-steps", type=int, default=0)
    p.add_argument("--lr-scheduler", default="linear", choices=["linear", "constant"])
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--grad-checkpoint", action="store_true", help="梯度检查点（省内存、更慢）")
    p.add_argument("--max-steps", type=int, default=0, help="每 epoch 最多 N 个 micro-step")
    p.add_argument("--monitor", default="event_exact",
                   choices=["event_exact", "f1", "type_macro_f1"])
    p.add_argument("--patience", type=int, default=2)

    # 生成 / 精度
    p.add_argument("--gen-max-new-tokens", type=int, default=256)
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    p.add_argument("--output-root", default=str(PROJECT_ROOT / "extract/output/llm_lora"))
    p.add_argument("--run-name", default="")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--log-every-eval", type=int, default=25)
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def stratified_sample(records: list[dict], per_class: int, seed: int):
    """按事件类型分层采样：每类最多 per_class 条（不足则全量保留）。

    返回 (采样结果, {类别: 条数})。这样稀有类不会被大类挤掉。
    """
    buckets: dict[str, list[dict]] = {}
    for r in records:
        key = (r.get("event") or {}).get("type") or "无"
        buckets.setdefault(key, []).append(r)
    rng = random.Random(seed)
    out: list[dict] = []
    dist: dict[str, int] = {}
    for key in sorted(buckets):
        items = list(buckets[key])
        rng.shuffle(items)
        take = items[:per_class] if per_class > 0 else items
        out.extend(take)
        dist[key] = len(take)
    rng.shuffle(out)
    return out, dist


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


EXPERIMENT_FIELDS = [
    "run_name", "start_time", "model", "lora_r", "lora_alpha", "lora_target",
    "max_seq_len", "epochs", "batch_size", "grad_accum", "lr", "warmup_steps",
    "n_train", "n_eval", "best_epoch", "event_exact", "val_f1", "trigger_f1",
    "arg_f1", "type_macro_f1", "parse_fail", "elapsed_min",
]


def append_experiment_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = not path.exists()
    with path.open("a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=EXPERIMENT_FIELDS)
        if header:
            w.writeheader()
        w.writerow(row)


# ------------------------------------------------------------------ 主流程


def main():
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    ts = time.strftime("%Y%m%d_%H%M%S")
    start_time = time.strftime("%Y-%m-%d %H:%M:%S")
    run_name = args.run_name or f"{ts}_qwen3lora_r{args.lora_r}_lr{args.lr:g}"
    run_dir = Path(args.output_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(run_dir / f"llm_lora_{run_name}.log")
    (run_dir / "config.json").write_text(
        json.dumps({**vars(args), "run_name": run_name, "start_time": start_time},
                   ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info("=" * 70)
    logger.info("Qwen3 + LoRA 事件抽取微调")
    logger.info("run: %s | 设备: %s | amp=%s", run_name, device, args.amp)
    logger.info("参数: %s", json.dumps(vars(args), ensure_ascii=False))

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(args.model_dir, dtype="float32")
    base.config.use_cache = False
    if args.grad_checkpoint:
        base.gradient_checkpointing_enable()

    from peft import LoraConfig, get_peft_model
    targets = [t.strip() for t in args.lora_target.split(",") if t.strip()]
    model = get_peft_model(base, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        bias="none", task_type="CAUSAL_LM", target_modules=targets,
    ))
    model.to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info("LoRA: r=%d alpha=%d dropout=%.3f targets=%s",
                args.lora_r, args.lora_alpha, args.lora_dropout, targets)
    logger.info("可训练参数: %d / %d (%.2f%%)", trainable, total, 100.0 * trainable / total)

    # ---------- 数据 ----------
    if args.max_per_class > 0:
        all_train = load_records(args.train_file)
        train_records, dist = stratified_sample(all_train, args.max_per_class, args.seed)
        logger.info("训练集分层采样（每类<=%d）：合计 %d 条 | 分布 %s",
                    args.max_per_class, len(train_records),
                    json.dumps(dist, ensure_ascii=False))
    else:
        train_records = load_records(args.train_file, limit=args.max_samples)

    if args.eval_per_class > 0:
        all_val = load_records(args.val_file)
        val_records, vdist = stratified_sample(all_val, args.eval_per_class, args.seed + 1)
        logger.info("评估集分层采样（每类<=%d）：合计 %d 条 | 分布 %s",
                    args.eval_per_class, len(val_records),
                    json.dumps(vdist, ensure_ascii=False))
    else:
        val_records = load_records(args.val_file, limit=args.eval_limit)

    train_ds = SFTDataset(train_records, tokenizer, args.max_seq_len)
    logger.info("训练 %d 条 | 评估 %d 条 | max_seq_len=%d",
                len(train_ds), len(val_records), args.max_seq_len)
    if device.type == "cpu" and val_records:
        est_h = len(val_records) * args.gen_max_new_tokens / 3.0 / 3600
        logger.info("CPU 评估预估：%d 条 × %d tokens ≈ %.1f 小时（实测约 3 token/s）",
                    len(val_records), args.gen_max_new_tokens, est_h)

    pad_id = tokenizer.pad_token_id
    loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0,
        collate_fn=lambda b: collate_fn(b, pad_id),
        generator=torch.Generator().manual_seed(args.seed),
    )

    # 诊断：确认 loss 只落在 assistant 段
    demo = collate_fn([train_ds[0]], pad_id)
    sup_ratio = float((demo["labels"] != IGNORE_INDEX).sum()) / demo["labels"].numel()
    logger.info("标签诊断: 有效 label 占比 %.1f%%（prompt 段应为 -100）", sup_ratio * 100)
    if not (0.0 < sup_ratio < 0.9):
        raise RuntimeError(f"标签掩码异常：有效 label 占比 {sup_ratio:.2%}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
    )
    steps_per_epoch = math.ceil(len(loader) / max(1, args.grad_accum_steps))
    if args.max_steps:
        steps_per_epoch = min(steps_per_epoch, math.ceil(args.max_steps / args.grad_accum_steps))
    total_updates = max(1, steps_per_epoch * args.epochs)
    warmup = args.warmup_steps if args.warmup_steps > 0 else int(total_updates * args.warmup_ratio)
    warmup = max(0, min(warmup, total_updates))

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(1, warmup)
        if args.lr_scheduler == "constant":
            return 1.0
        return max(0.0, (total_updates - step) / max(1, total_updates - warmup))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    logger.info("优化: AdamW lr=%.2e wd=%.3f | warmup %d / %d updates | %s",
                args.lr, args.weight_decay, warmup, total_updates, args.lr_scheduler)

    best_score, best_epoch, no_improve = -1.0, -1, 0
    adapter_dir = run_dir / "adapter"
    best_meta: dict = {}
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        model.config.use_cache = False
        t0 = time.time()
        running, n_micro, seen = 0.0, 0, 0
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(loader, start=1):
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=args.amp and device.type == "cpu"):
                loss = model(input_ids=input_ids, attention_mask=attn, labels=labels).loss
            (loss / args.grad_accum_steps).backward()
            running += float(loss.item())
            n_micro += 1
            seen += input_ids.size(0)

            if step % args.grad_accum_steps == 0 or step == len(loader):
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], args.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if n_micro % args.log_every == 0:
                logger.info("    step %d | loss=%.4f | lr=%.2e | %d 条 | %.0fs",
                            n_micro, running / n_micro,
                            optimizer.param_groups[0]["lr"], seen, time.time() - t0)
            if args.max_steps and n_micro >= args.max_steps:
                break

        train_loss = running / max(n_micro, 1)
        logger.info("Epoch %d/%d | train_loss=%.4f | %.1fs",
                    epoch, args.epochs, train_loss, time.time() - t0)

        if not val_records:
            continue
        result, rows = evaluate(model, tokenizer, val_records, args, device, logger)
        report, stats = result["report"], result["stats"]
        score = score_of(report, args.monitor)
        d = report.as_dict()
        em = d.get("event_match") or {}
        logger.info("    评估: %s | 解析失败 %d | 无事件 %d | %.0fs",
                    report.summary(), stats["parse_fail"], stats["no_event"],
                    stats["elapsed_sec"])
        et = d.get("event_type")
        if et:
            logger.info("    事件类型: acc=%.4f macroF1=%.4f | %s",
                        et["accuracy"], et["macro_f1"],
                        json.dumps(et["per_class_f1"], ensure_ascii=False))
        if em.get("per_type"):
            logger.info("    分类型 exact: %s", json.dumps(em["per_type"], ensure_ascii=False))

        if score > best_score:
            best_score, best_epoch, no_improve = score, epoch, 0
            model.save_pretrained(adapter_dir)
            (run_dir / "val_predictions.jsonl").write_text(
                "\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
            (run_dir / "val_report.json").write_text(json.dumps(
                {"args": vars(args), "best_epoch": epoch, "stats": stats,
                 "report": d}, ensure_ascii=False, indent=2), encoding="utf-8")
            best_meta = {"stats": stats, "report": d, "train_loss": train_loss}
            logger.info("    ★ 新最优 epoch=%d %s=%.4f -> %s",
                        epoch, args.monitor, score, adapter_dir)
        else:
            no_improve += 1
            logger.info("    %s 未提升 (%d/%d)", args.monitor, no_improve, args.patience)
            if no_improve >= args.patience:
                logger.info("    触发 early stopping")
                break

    elapsed_min = (time.time() - t_start) / 60
    if best_meta:
        d = best_meta["report"]
        em = d.get("event_match") or {}
        et = d.get("event_type") or {}
        append_experiment_row(Path(args.output_root) / "experiments.csv", {
            "run_name": run_name, "start_time": start_time,
            "model": Path(args.model_dir).name, "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha, "lora_target": args.lora_target,
            "max_seq_len": args.max_seq_len, "epochs": args.epochs,
            "batch_size": args.batch_size, "grad_accum": args.grad_accum_steps,
            "lr": args.lr, "warmup_steps": warmup,
            "n_train": len(train_ds), "n_eval": len(val_records),
            "best_epoch": best_epoch, "event_exact": round(em.get("accuracy", 0.0), 4),
            "val_f1": round((d.get("overall") or {}).get("f1", 0.0), 4),
            "trigger_f1": round((d.get("trigger") or {}).get("f1", 0.0), 4),
            "arg_f1": round((d.get("arg") or {}).get("f1", 0.0), 4),
            "type_macro_f1": round(et.get("macro_f1", 0.0), 4),
            "parse_fail": best_meta["stats"]["parse_fail"],
            "elapsed_min": round(elapsed_min, 1),
        })

    logger.info("=" * 70)
    logger.info("训练完成：最优 epoch=%d %s=%.4f，总耗时 %.1f 分钟",
                best_epoch, args.monitor, best_score, elapsed_min)
    logger.info("适配器: %s", adapter_dir)
    logger.info("验证报告: %s", run_dir / "val_report.json")


if __name__ == "__main__":
    main()
