"""
L4 生成式微调：Qwen3-0.6B + LoRA / 全量 SFT，学会「文本 -> 实体 span 列表 JSON」。

三类方法对比的第三臂：
  L1 BiLSTM+CRF（无预训练）  vs L2 RoBERTa+CRF（预训练判别式）  vs L4 SFT（生成式微调）
两种 SFT 模式（消融「训多少参数」）：
  --mode lora  只训 q/k/v/o 旁路（r=8, alpha=16），显存宽松，~0.2% 参数
  --mode full  全量微调，**分层学习率**（底层小、上层大、embed/head 单独设），显存吃紧

【已落地的工程要点】
  1. loss masking：只在 JSON 回复算 loss，prompt 全 -100（sft_data.py + collate_sft 的 pad -100）；
  2. 目标格式与推理/评测一致（llm_utils.parse_entities 解析同一格式）；
  3. enable_thinking=False，否则 <think> 污染 JSON；
  4. warmup + linear decay（get_linear_schedule_with_warmup）；
  5. 梯度剪裁（clip_grad_norm_）；
  6. 全量微调分层学习率（build_param_groups）；
  7. 长句滑动窗口由 common/sliding_window.py 在评测侧提供（训练侧靠 --max_length + 滑窗样本增强可选）。

【T4 显存】full 0.6B 用 fp32 主权重 + 8bit 优化器约 5.7 GiB（week6 实测账）；
  装不下就 --grad_ckpt --batch_size 1 --grad_accum 16。

用法（Colab）：
  python llm_lora/train_sft.py --mode lora   --run_name qwen_lora_sft --num_train 2000
  python llm_lora/train_sft.py --mode full   --run_name qwen_full_sft --num_train 2000 \
      --grad_ckpt --dtype fp32 --lr 2e-5 --optim adamw8bit
  python llm_lora/train_sft.py --max_steps 20 --num_train 40 --run_name smoke  # 冒烟
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR / "llm_lora"))

from common import paths
from common.model_paths import resolve_qwen
from sft_data import SFTDataset, collate_sft


def parse_args():
    p = argparse.ArgumentParser(description="L4 Qwen3-0.6B SFT for NER（LoRA / 全量）")
    p.add_argument("--mode", choices=["lora", "full"], default="lora",
                   help="lora=只训旁路；full=全量微调+分层学习率")
    p.add_argument("--model_path", default=None, help="本地 Qwen3-0.6B；不给则解析/在线加载")
    p.add_argument("--run_name", default=None)
    p.add_argument("--max_length", type=int, default=640, help="prompt+目标 JSON 的长度上限")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=None,
                   help="学习率；不传时按模式自动选：lora=2e-4，full=2e-5。"
                        "full 用 2e-4 会把 fp16 权重推到 inf → loss nan")
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    # LoRA 旋钮
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    # full 分层学习率旋钮
    p.add_argument("--layer_decay", type=float, default=0.95,
                   help="full 模式：每层 lr 相对上一层的衰减系数，1.0=不分层；底层=lr*decay^(L-1-layer)")
    p.add_argument("--embed_head_lr_mult", type=float, default=1.0,
                   help="full 模式：embed_tokens / lm_head / norm 的 lr 倍数（相对最上层 lr）")
    p.add_argument("--optim", default="adamw", choices=["adamw", "adamw8bit"],
                   help="full 模式建议 adamw8bit（bitsandbytes，省 ~2/3 优化器状态）")
    p.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    p.add_argument("--grad_ckpt", action="store_true", help="梯度检查点（full 模式强烈建议开）")
    p.add_argument("--num_train", type=int, default=0, help="只用前 N 条；0=全量")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_steps", type=int, default=0, help="冒烟：最多多少个优化器步")
    p.add_argument("--device", default="auto")
    return p.parse_args()


def set_seed(s: int) -> None:
    random.seed(s)
    import numpy as np
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def resolve_device(name: str) -> torch.device:
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


def pick_dtype(name: str, device: torch.device, mode: str = "lora"):
    """
    选择权重 dtype。关键（week6 踩过的坑，week7 full SFT 又踩了一次）：
      - full 模式在 T4 上**必须用 fp32 主权重**。T4 无原生 bf16，auto 若回退 fp16，
        0.6B 全量 + 词表 151936 的前向 logits / 梯度会超出 fp16 上限（65504），
        没有 GradScaler 兜底就渐进溢出 → loss 在 step 8 左右变 nan。
        0.6B fp32 主权重 + 8bit 优化器约 5.7 GiB，T4 14.6 GiB 装得下。
      - bf16（数值范围≈fp32）不受此影响，原生支持时优先用。
      - 显式 --dtype fp16 是用户强行选择，此时训练循环启用 GradScaler 缩放 loss 兜底。
    """
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    # auto
    if device.type == "cuda":
        try:
            if torch.cuda.is_bf16_supported(including_emulation=False):
                return torch.bfloat16       # 原生 bf16，安全
        except TypeError:
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
        # 无原生 bf16：full 用 fp32（安全），lora 才敢用 fp16（旁路小、不易溢出）
        return torch.float32 if mode == "full" else torch.float16
    return torch.float32


def build_param_groups(model, args, n_layers: int) -> list[dict]:
    """
    全量微调的分层学习率（layer-wise LR decay）。

    规则（Qwen3ForCausalLM 结构）：
      - model.layers.<i>     : lr_i = args.lr * layer_decay^(n_layers-1-i)
                               （最上层 i=n_layers-1 用 base lr，越往下越小）
      - model.embed_tokens   : lr = args.lr * embed_head_lr_mult
      - model.norm (final)   : lr = args.lr * embed_head_lr_mult
      - lm_head              : lr = args.lr * embed_head_lr_mult

    这样底层特征抽取稳、高层语义和分类头学得快——week6 判别式「encoder 小 lr、
    head 大 lr」同款思路在全量 LM 上的推广。LoRA 模式不走这里。
    """
    import re
    decay = args.layer_decay
    mult = args.embed_head_lr_mult
    groups: dict[str, dict] = {}

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # 1) 先按 transformer block 的层号分组。用正则精确匹配 "layers.<数字>"，
        #    不能用 "norm" 这种宽泛关键词——层内的 input_layernorm/post_attention_layernorm
        #    必须跟所在层同 lr，不能被误判成 final norm。
        m = re.search(r"layers\.(\d+)\.", name)
        if m:
            layer_idx = int(m.group(1))
            key = f"layer_{layer_idx}"
            lr = args.lr * (decay ** (n_layers - 1 - layer_idx))
        elif "embed_tokens" in name or "lm_head" in name or name in ("model.norm.weight", "model.norm.bias") \
                or re.fullmatch(r"norm\.(weight|bias)", name):
            # 2) 只有 embed / lm_head / 最终 norm 用 head 组
            key = "embed_head_norm"
            lr = args.lr * mult
        else:
            key = "other"
            lr = args.lr
        if key not in groups:
            groups[key] = {"params": [], "lr": lr, "_names": []}
        groups[key]["params"].append(p)
        groups[key]["_names"].append(name)

    out = []
    for key in sorted(groups.keys()):
        g = groups[key]
        names = g.pop("_names")
        out.append(g)
        print(f"    [{key:<14}] lr={g['lr']:.2e}  {len(names)} 参数 "
              f"({', '.join(names[:2])}{'...' if len(names) > 2 else ''})")
    return out


def build_optimizer(param_groups: list[dict], args):
    """按 optim 名构造优化器；8bit 状态省显存，full 模式 T4 上的关键。"""
    if args.optim == "adamw8bit":
        try:
            import bitsandbytes as bnb
        except ImportError as exc:
            raise ImportError("adamw8bit 需要 bitsandbytes：pip install -U bitsandbytes") from exc
        opt = bnb.optim.AdamW8bit(param_groups, lr=args.lr, weight_decay=args.weight_decay)
        print(f"  优化器 AdamW8bit（8bit 状态，full 模式省 ~2/3 优化器显存）")
        return opt
    opt = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    print(f"  优化器 AdamW")
    return opt


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    smoke = args.max_steps > 0 or args.num_train > 0
    tag = args.run_name or f"qwen_{args.mode}_sft_{datetime.now():%m%d_%H%M%S}"
    out_root = paths.get_out_root() / ("smoke" if smoke else "")
    log_dir = out_root / "logs"; log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_root = out_root / "checkpoints" / tag
    ckpt_root.mkdir(parents=True, exist_ok=True)

    # lr 按模式给默认值：full 用 2e-5（用 2e-4 会溢出），lora 用 2e-4
    if args.lr is None:
        args.lr = 2e-5 if args.mode == "full" else 2e-4
    model_path = args.model_path or str(resolve_qwen())
    dtype = pick_dtype(args.dtype, device, args.mode)
    print("=" * 72)
    print(f"L4 Qwen3-0.6B {args.mode.upper()} SFT | device={device} dtype={dtype} | run={tag}")
    print(f"  model={model_path}")
    print(f"  epochs={args.epochs} bs={args.batch_size} grad_accum={args.grad_accum} "
          f"lr={args.lr} max_len={args.max_length}")
    if dtype == torch.float16:
        print("  [警告] fp16 主权重没有 GradScaler 兜底，full 模式可能 step 8 左右 loss=nan；"
              "建议 --dtype fp32（或原生 bf16 机器用 bf16）")
    if args.mode == "lora":
        print(f"  LoRA r={args.lora_r} alpha={args.lora_alpha} target=q/k/v/o_proj")
    else:
        print(f"  full 分层: layer_decay={args.layer_decay} "
              f"embed_head_lr_mult={args.embed_head_lr_mult} optim={args.optim} grad_ckpt={args.grad_ckpt}")
    print("=" * 72)

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    records = json.loads((paths.get_data_dir() / "train.json").read_text(encoding="utf-8"))
    if args.num_train > 0:
        records = records[:args.num_train]
    ds = SFTDataset(records, tok, max_length=args.max_length)
    print(f"  样本 {len(ds)} | prompt 截断 {ds.n_prompt_truncated} | 目标截断 {ds.n_target_truncated}")
    if ds.n_prompt_truncated > 0:
        print("  ⚠️ prompt 截断 > 0：这些样本的 JSON 目标被挤掉，loss 学不到东西，请调大 --max_length")

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        collate_fn=lambda b: collate_sft(b, int(tok.pad_token_id)))

    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype, trust_remote_code=False)

    if args.mode == "lora":
        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except ImportError as exc:
            raise ImportError("LoRA 需要 peft：pip install peft") from exc
        if args.grad_ckpt:
            model.gradient_checkpointing_enable()
            model.config.use_cache = False
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=args.lora_r, lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()
        param_groups = [{"params": [p for p in model.parameters() if p.requires_grad], "lr": args.lr}]
        n_layers = 0
    else:
        # 全量：梯度检查点几乎必开（否则激活吃不下）
        if not args.grad_ckpt:
            print("  [提示] full 模式建议加 --grad_ckpt，否则激活显存可能 OOM")
        else:
            model.gradient_checkpointing_enable()
            model.config.use_cache = False
        try:
            n_layers = int(model.config.num_hidden_layers)
        except AttributeError:
            n_layers = 28     # Qwen3-0.6B = 28 层，兜底
        print(f"  构建分层学习率（transformer {n_layers} 层）：")
        param_groups = build_param_groups(model, args, n_layers)

    model.to(device)
    opt = build_optimizer(param_groups, args)
    steps_per_ep = max(1, len(loader) // args.grad_accum)
    if args.max_steps > 0:
        steps_per_ep = min(steps_per_ep, args.max_steps)
    total_steps = steps_per_ep * args.epochs
    sched = get_linear_schedule_with_warmup(
        opt, num_warmup_steps=max(1, int(total_steps * args.warmup_ratio)),
        num_training_steps=total_steps)
    # fp16 主权重必须配 GradScaler（缩放 loss 防梯度下溢/溢出）；fp32 / bf16 不需要
    use_scaler = dtype == torch.float16
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    except (AttributeError, TypeError):      # 老 torch 兼容
        scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)
    print(f"  total_steps={total_steps} warmup={max(1,int(total_steps*args.warmup_ratio))} "
          f"scheduler=linear (warmup+decay) clip={args.max_grad_norm} "
          f"GradScaler={use_scaler}\n")

    history = []
    step = 0
    nan_skipped = 0
    model.train()
    for epoch in range(1, args.epochs + 1):
        acc_loss, acc_n = 0.0, 0
        opt.zero_grad(set_to_none=True)
        for i, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss / args.grad_accum
            if not torch.isfinite(loss):
                nan_skipped += 1
                if nan_skipped <= 5 or nan_skipped % 50 == 0:
                    print(f"  [warn] loss 非有限（第 {nan_skipped} 次，step {i}）跳过。"
                          f"若持续出现：full 模式请确认 dtype=fp32/bf16 且 lr=2e-5，不要用 fp16+2e-4")
                if nan_skipped >= 200:
                    raise RuntimeError("loss 连续非有限（≥200 次），训练无效，终止。"
                                       "检查 dtype（full 必须 fp32/bf16）与 lr（full 用 2e-5）")
                continue
            scaler.scale(loss).backward()
            acc_loss += float(out.loss.item()); acc_n += 1
            if (i + 1) % args.grad_accum == 0:
                # GradScaler 下必须先 unscale_ 再裁剪，否则裁的是被放大的梯度
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(opt)          # 内部检测 inf/nan，遇到就跳过本次更新
                scaler.update()
                sched.step(); opt.zero_grad(set_to_none=True)
                step += 1
                if step % 20 == 0:
                    print(f"  ep{epoch} opt_step {step}/{total_steps} "
                          f"loss={acc_loss/max(1,acc_n):.4f} lr={sched.get_last_lr()[0]:.2e}", flush=True)
                if args.max_steps > 0 and step >= args.max_steps:
                    break
        avg = acc_loss / max(1, acc_n)
        rec = {"epoch": epoch, "avg_loss": round(avg, 6), "opt_steps": step,
               "timestamp": datetime.now().isoformat(timespec="seconds")}
        history.append(rec)
        with (log_dir / f"train_{tag}.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"  [Epoch {epoch}] avg_loss={avg:.4f} opt_steps={step}")
        if args.max_steps > 0 and step >= args.max_steps:
            break

    # 保存：lora 存 adapter，full 存整模型
    if args.mode == "lora":
        save_dir = ckpt_root / "adapter"
        model.save_pretrained(str(save_dir))
        tok.save_pretrained(str(save_dir))
    else:
        save_dir = ckpt_root / "full"
        model.save_pretrained(str(save_dir))
        tok.save_pretrained(str(save_dir))
    summary = {"run_name": tag, "method": f"Qwen3-0.6B {args.mode.upper()} SFT",
               "mode": args.mode, "ckpt_dir": str(save_dir),
               "history": history, "args": vars(args),
               "n_prompt_truncated": ds.n_prompt_truncated,
               "n_target_truncated": ds.n_target_truncated,
               "layer_decay": args.layer_decay if args.mode == "full" else None,
               "finished": datetime.now().isoformat(timespec="seconds")}
    (log_dir / f"summary_{tag}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  {'adapter' if args.mode=='lora' else 'full 权重'} 已保存: {save_dir}")
    print(f"  下一步: python llm_lora/evaluate_sft.py --ckpt {save_dir} --split test")


if __name__ == "__main__":
    main()
