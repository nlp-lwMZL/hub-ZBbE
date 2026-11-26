# -*- coding: utf-8 -*-
"""
NuExtract3 SFT（LoRA / 全量）—— 事件抽取

与 train_sft.py 的差异仅在「模型加载」和「chat template 构造」两处：
  1. NuExtract3 是 Qwen3.5-4B VLM，用 AutoModelForImageTextToText 加载，
     processor 负责 chat template；纯文本场景下 tokenizer 从 processor.tokenizer 取。
  2. 结构化抽取必须传入 mode="structured" + template + enable_thinking=False，
     否则模型不会进入 JSON 输出模式，训练目标与推理行为会错位。
  3. 4B 模型在 8GB 卡上 LoRA 可行，但默认开启梯度检查点，batch_size=1 起步。

依赖（NuExtract3 对 transformers 版本敏感）：
  pip install "transformers>=5.0.0" peft torch accelerate tqdm
  若你的环境仍是 4.x，可改用 AutoModelForMultimodalLM，但建议直接上 v5。

使用方式：
  python llm_lora/train_sft_nuextract3.py                      # LoRA，全量 train
  python llm_lora/train_sft_nuextract3.py --max_steps 20       # 冒烟
  python llm_lora/train_sft_nuextract3.py --full_ft --lr 1e-5  # 全量微调（8GB 大概率 OOM）
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForImageTextToText, AutoProcessor
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "common"))
sys.path.insert(0, str(ROOT / "llm_zero_shot"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from log_utils import TrainLogger, peak_gpu_mb
from event_data import (
    ROOT, SEED, canonical_order, load_jsonl, load_schema, to_target_json,
)
from model_paths import NUEXTRACT_MODEL_DIR, resolve_model_path

# 直接复用 train_sft.py 的 loss / collate，保证与 Qwen3-0.6B 链路同一套实现
from train_sft import collate_fn, compute_lm_loss

try:
    from peft import LoraConfig, TaskType, get_peft_model
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

DATA_DIR   = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "outputs" / "llm" / "ckpt"


# ══════════════════════════════════════════════════════════════════════════════
# Dataset：NuExtract3 结构化抽取模式
# ══════════════════════════════════════════════════════════════════════════════

class NuExtractSFTDataset(Dataset):
    """
    与 EventSFTDataset 结构相同，只改 prompt 的构造方式。

    关键：NuExtract3 的 processor.apply_chat_template 在结构化抽取模式下
    必须接收 mode="structured"、template 和 enable_thinking=False。
    template 是一个与输出同构的 JSON 模板字符串（叶子节点为类型占位符）。
    """

    def __init__(self, rows, processor, tokenizer, template_str: str,
                 max_length: int = 1024):
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        self.n_prompt_truncated = 0
        self.n_target_truncated = 0

        for row in tqdm(rows, desc="tokenize(nuextract3)", ncols=80):
            target = to_target_json(row["event"])

            # NuExtract3 结构化抽取：纯文本输入，不需要图像占位符
            prompt_text = processor.apply_chat_template(
                [{"role": "user", "content": f"句子：{row['text']}"}],
                tokenize=False,
                add_generation_prompt=True,
                mode="structured",
                template=template_str,
                enable_thinking=False,
            )

            prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
            # 目标 JSON + EOS；EOS 必须保留，否则推理时不会停
            response_ids = tokenizer.encode(target, add_special_tokens=False) + \
                [tokenizer.eos_token_id]

            input_ids = (prompt_ids + response_ids)[: max_length]
            labels = ([-100] * len(prompt_ids) + response_ids)[: max_length]

            if len(prompt_ids) >= max_length:
                self.n_prompt_truncated += 1
            elif len(prompt_ids) + len(response_ids) > max_length:
                self.n_target_truncated += 1
            if all(x == -100 for x in labels):
                continue

            self.samples.append({
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ══════════════════════════════════════════════════════════════════════════════
# 参数
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="NuExtract3 SFT 事件抽取")
    p.add_argument("--model_path", default=str(resolve_model_path(NUEXTRACT_MODEL_DIR)))
    p.add_argument("--data_dir", default=str(DATA_DIR))
    p.add_argument("--output_dir", default=str(OUTPUT_DIR))
    p.add_argument("--num_train", default=-1, type=int, help="-1 = 用全部 train")
    p.add_argument("--epochs", default=1, type=int)
    p.add_argument("--batch_size", default=1, type=int,
                   help="4B + 151k 词表，8GB 卡 batch=1；LoRA 可试 2")
    p.add_argument("--grad_accum", default=16, type=int)
    p.add_argument("--lr", default=1e-4, type=float,
                   help="NuExtract3 官方 LoRA 推荐 lr≈1e-4")
    p.add_argument("--max_length", default=1024, type=int,
                   help="4B 显存更紧；prompt+目标 p95≈1161，完整覆盖需 1280 + --grad_ckpt")
    p.add_argument("--full_ft", action="store_true",
                   help="全量微调（4B 全参数；8GB 卡大概率 OOM，仅作对比）")
    p.add_argument("--lora_r", default=8, type=int)
    p.add_argument("--lora_alpha", default=16, type=int)
    p.add_argument("--lora_dropout", default=0.05, type=float)
    p.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp32"])
    p.add_argument("--grad_ckpt", action="store_true", default=True,
                   help="4B 模型 8GB 卡默认开梯度检查点")
    p.add_argument("--no_grad_ckpt", action="store_false", dest="grad_ckpt")
    p.add_argument("--seed", default=SEED, type=int)
    p.add_argument("--run_name", default="nuextract3_sft")
    p.add_argument("--max_steps", default=0, type=int)
    p.add_argument("--val_size", default=200, type=int)
    p.add_argument("--std_logits", action="store_true",
                   help="退回标准 HF loss（慢、费显存，仅作对照）")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_dir, output_dir = Path(args.data_dir), Path(args.output_dir)
    ckpt_dir = output_dir / (
        "sft_full_ckpt_nuextract3" if args.full_ft else "sft_adapter_nuextract3"
    )
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device} | 模型: NuExtract3 | 模式: {'全量微调' if args.full_ft else 'LoRA'}")

    # ── 数据：train 里留 val，绝不用 dev ──────────────────────────────────────
    rows = load_jsonl(data_dir / "train_extract.jsonl")
    ordered = canonical_order(rows, args.seed)
    n_val = min(args.val_size, len(ordered) // 20)
    val_rows, train_rows = ordered[:n_val], ordered[n_val:]
    if args.num_train > 0:
        train_rows = train_rows[: args.num_train]
    print(f"训练 {len(train_rows)} 条 | 验证(取自 train 头部) {len(val_rows)} 条")

    # ── 构造 NuExtract3 的结构化模板 ─────────────────────────────────────────
    # template 的叶子节点用类型占位符，模型会按这个结构输出。
    # 建议先用 1 条真实 to_target_json 的输出作为 template 跑 --max_steps 5 冒烟，
    # 确认模型能正常进入 JSON 模式；下面的模板是事件抽取的通用占位版本。
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
    template_str = json.dumps(template_obj, ensure_ascii=False)

    # ── Processor / Tokenizer / Model ────────────────────────────────────────
    # NuExtract3 必须用 AutoProcessor 加载 processor；tokenizer 从 processor 取。
    processor = AutoProcessor.from_pretrained(
        args.model_path, trust_remote_code=True)
    tokenizer = processor.tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    torch_dtype = {
        "auto": torch.bfloat16 if device.type == "cuda" else torch.float32,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[args.dtype]
    print(f"加载基座: {args.model_path}（dtype={torch_dtype}）")

    # NuExtract3 是 VLM，用 AutoModelForImageTextToText 加载；
    # 纯文本训练时不需要真实图像输入，但模型架构仍是视觉语言模型。
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path, dtype=torch_dtype, trust_remote_code=True,
    )
    model.config.use_cache = False

    train_ds = NuExtractSFTDataset(
        train_rows, processor, tokenizer, template_str, args.max_length)
    val_ds = NuExtractSFTDataset(
        val_rows, processor, tokenizer, template_str, args.max_length)
    print(f"prompt 完全截断（丢弃）：{train_ds.n_prompt_truncated} 条 | "
          f"目标尾部截断：{train_ds.n_target_truncated} 条（max_length={args.max_length}）")
    if train_ds.n_prompt_truncated > 0:
        print("[警告] 有样本 prompt 超过 max_length，labels 全 -100 → 白跑。"
              "请调大 --max_length（或先用 --max_steps 5 检查截断数）")

    _collate = lambda b: collate_fn(b, tokenizer.pad_token_id)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=_collate, drop_last=False)
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=_collate)

    # ── LoRA / 全量 ──────────────────────────────────────────────────────────
    if args.full_ft:
        total = sum(p.numel() for p in model.parameters())
        print(f"trainable params: {total:,} || all params: {total:,} || trainable%: 100.0")
    else:
        if not PEFT_AVAILABLE:
            raise ImportError("LoRA 需要 peft：pip install peft>=0.14.0")
        # NuExtract3（Qwen3.5 系）的注意力投影层名同样是 q/k/v/o_proj
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r, lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=args.lora_dropout, bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    if args.grad_ckpt:
        model.gradient_checkpointing_enable()
    model = model.to(device)

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01)
    steps_per_epoch = max(1, len(train_loader) // args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    if args.max_steps > 0:
        total_steps = min(total_steps, args.max_steps)
    print(f"训练步数: {total_steps}（等效 batch={args.batch_size * args.grad_accum}，"
          f"lr={args.lr}）\n")

    logger = TrainLogger(
        method=args.run_name, args=args, root=ROOT,
        params=sum(p.numel() for p in model.parameters() if p.requires_grad),
        notes="NuExtract3 LoRA SFT（Qwen3.5-4B VLM，纯文本事件抽取）")

    history, global_step = [], 0
    best_val_loss = float("inf")
    t_train0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss, running_tokens = 0.0, 0
        optimizer.zero_grad()
        t0 = time.time()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", ncols=100)

        for step, batch in enumerate(pbar, 1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            if args.std_logits:
                out = model(input_ids=input_ids, attention_mask=attention_mask,
                            labels=labels)
                loss = out.loss
            else:
                # 复用 train_sft.py 的快路径：只对尾部目标段算 logits
                loss = compute_lm_loss(model, input_ids, attention_mask, labels)
                if loss is None:
                    optimizer.zero_grad()
                    continue

            if torch.isnan(loss) or torch.isinf(loss):
                print("[警告] loss=nan，跳过该 batch（检查 max_length）")
                optimizer.zero_grad()
                continue

            logger.log_step(epoch=epoch, step=global_step,
                            loss=float(loss.item()),
                            lr=optimizer.param_groups[0]["lr"])
            (loss / args.grad_accum).backward()

            if step % args.grad_accum == 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                if global_step % 20 == 0:
                    pbar.set_postfix(loss=f"{loss.item():.4f}", step=global_step)
                if args.max_steps > 0 and global_step >= args.max_steps:
                    break

            n_tokens = int((labels != -100).sum().item())
            running_loss += loss.item() * n_tokens
            running_tokens += n_tokens

        train_loss = running_loss / max(running_tokens, 1)
        elapsed = time.time() - t0

        # ── 验证 loss ────────────────────────────────────────────────────────
        model.eval()
        v_loss, v_tokens = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                if args.std_logits:
                    out = model(input_ids=input_ids, attention_mask=attention_mask,
                                labels=labels)
                    val_batch_loss = out.loss
                else:
                    val_batch_loss = compute_lm_loss(
                        model, input_ids, attention_mask, labels)
                n_tokens = int((labels != -100).sum().item())
                if val_batch_loss is None:
                    continue
                v_loss += val_batch_loss.item() * n_tokens
                v_tokens += n_tokens
        val_loss = v_loss / max(v_tokens, 1)

        print(f"Epoch {epoch}/{args.epochs} | train_loss={train_loss:.4f} "
              f"val_loss={val_loss:.4f} | {elapsed:.0f}s")
        logger.log_epoch(epoch=epoch, train_loss=train_loss, elapsed=elapsed,
                         val_metrics={"val_loss": round(val_loss, 4)},
                         extra={"optimizer_steps": global_step})
        history.append({
            "epoch": epoch, "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4), "elapsed_s": round(elapsed, 1),
            "optimizer_steps": global_step,
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_pretrained(str(ckpt_dir))
            tokenizer.save_pretrained(str(ckpt_dir))
            # processor 也保存：推理时 apply_chat_template 需要 mode/template 等 kwargs
            processor.save_pretrained(str(ckpt_dir))
            print(f"  ✓ checkpoint 已保存 → {ckpt_dir}")

        if args.max_steps > 0 and global_step >= args.max_steps:
            break

    total_s = time.time() - t_train0
    log = {
        "mode": "full_ft" if args.full_ft else "lora",
        "base_model": "NuExtract3",
        "args": {k: (str(v) if isinstance(v, Path) else v)
                 for k, v in vars(args).items()},
        "num_train": len(train_ds), "num_val": len(val_ds),
        "prompt_truncated": train_ds.n_prompt_truncated,
        "target_truncated": train_ds.n_target_truncated,
        "optimizer_steps": global_step, "history": history,
        "best_val_loss": round(best_val_loss, 4),
        "total_train_s": round(total_s, 1),
        "trainable_params": sum(p.numel() for p in model.parameters()
                                if p.requires_grad),
        "all_params": sum(p.numel() for p in model.parameters()),
        "gpu_peak_mem_gib": round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2)
        if torch.cuda.is_available() else 0,
    }
    log_path = output_dir / "train_log_nuextract3.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)

    logger.finalize(total_s=total_s, optimizer_steps=global_step,
                    best_val_loss=best_val_loss, peak_gpu_mb=peak_gpu_mb())
    print(f"\n训练完成：{total_s / 60:.1f} 分钟，"
          f"峰值显存 {log['gpu_peak_mem_gib']} GiB")
    print(f"日志 → {log_path}\ncheckpoint → {ckpt_dir}")
    print("\n下一步（评测需单独适配 processor 的 template 传递）："
          f"python llm_lora/evaluate_sft.py --ckpt_dir {ckpt_dir}")


if __name__ == "__main__":
    main()