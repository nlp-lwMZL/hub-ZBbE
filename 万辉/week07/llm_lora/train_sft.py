# -*- coding: utf-8 -*-
"""
L4 生成式微调：LoRA / 全量 SFT，让 Qwen3-0.6B 学会「公告句 → 事件 JSON」

对应课程里的 L4「监督微调」层：zero-shot 只能拿到"大致能用"的抽取，
要让 0.6B 这种小模型稳定输出符合 schema 的事件 JSON，最有效的是 SFT。

教学重点：
  1. **loss masking**：只在 JSON 输出部分算 loss，prompt 部分全置 -100。
     不 mask 的话模型会花一半算力去学"复述这句话"，而且 loss 被长 prompt 稀释。
  2. **目标格式必须与推理/评测一致**：这里的目标串由 event_data.to_target_json() 生成，
     和 llm_extract.py / evaluate_sft.py 用的是同一个函数（同一个 prompt 也是）。
     目标是 {'{"events": [{"type", "trigger", "arguments"}]}'}。
  3. **max_length 不是拍脑袋定的**：实测本任务 prompt（含 13 类 schema）+ 目标 JSON
     平均 1011 token、p95 1161、max 1280。照抄 NER 项目的 384 会让 **100% 的样本
     prompt 被截断**，labels 全是 -100 → loss 直接 nan，一个 token 都学不到。
     所以这里默认 1280（train 里 99.8% 的样本完整装下）。
  4. **LoRA 只训 0.2% 参数**：r=8/alpha=16，挂到 q/k/v/o_proj 上。
     8GB 显卡跑 0.6B 全量微调基本没戏，LoRA 是唯一现实选择（--full_ft 仅作对比开关）。

使用方式：
  python llm_lora/train_sft.py                          # LoRA，1 epoch，全量 5747 条
  python llm_lora/train_sft.py --max_steps 20           # 冒烟：只跑 20 步
  python llm_lora/train_sft.py --num_train 1000         # 小数据快速验证
  python llm_lora/train_sft.py --full_ft --lr 2e-5      # 全量微调（显存不够会 OOM）

依赖：pip install torch transformers peft tqdm
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # Windows: torch/numpy 的 OpenMP 冲突
# 8GB 卡上训练 0.6B 会遇到显存碎片：不加这两个设置，训练到一半会报
# "Tried to allocate 2.43 GiB ... 339 MiB is reserved but unallocated"。
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
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]   # 项目根目录
sys.path.insert(0, str(ROOT / "common"))        # 公共模块
sys.path.insert(0, str(ROOT / "llm_zero_shot"))  # 生成式链路共用：event_data / event_prompt / local_generate
sys.path.insert(0, str(Path(__file__).resolve().parent))  # llm_lora 自己的 model_paths.py

from log_utils import TrainLogger, peak_gpu_mb  # noqa: E402
from event_data import (  # noqa: E402
    ROOT, SEED, canonical_order, load_jsonl, load_schema, to_target_json,
)
from event_prompt import build_system_prompt  # noqa: E402
from model_paths import QWEN_MODEL_DIR, resolve_model_path  # noqa: E402

try:
    from peft import LoraConfig, TaskType, get_peft_model
    PEFT_AVAILABLE = True
except ImportError:      # 全量微调模式下不需要 peft
    PEFT_AVAILABLE = False

DATA_DIR   = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "outputs" / "llm" / "ckpt"

# ══════════════════════════════════════════════════════════════════════════════
# 数据集：公告句 → chat 格式 + loss mask
# ══════════════════════════════════════════════════════════════════════════════

class EventSFTDataset(Dataset):
    """
    构造一条训练样本（以 Qwen3 的 chat template 为例）：

      ┌─────────────────────────────────────────────────────────────┐
      │ <|im_start|>system\n{13 类 schema 提示词}<|im_end|>          │  → label = -100
      │ <|im_start|>user\n句子：{公告原文}<|im_end|>                  │  → label = -100
      │ <|im_start|>assistant\n{"events": [...]}<|im_end|>           │  → 真实 token id（算 loss）
      └─────────────────────────────────────────────────────────────┘

    labels 长度与 input_ids 严格对齐（HF 的 causal LM 内部会右移一位，
    所以第 i 个 label 对应"用前 i 个 token 预测第 i 个 token"）。
    """

    def __init__(self, rows, tokenizer, system_prompt: str, max_length: int = 1280):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        self.n_prompt_truncated = 0   # prompt 长到把整个目标挤掉的样本数（必须监控）
        self.n_target_truncated = 0   # 目标 JSON 被截断尾巴的样本数

        for row in tqdm(rows, desc="tokenize", ncols=80):
            target = to_target_json(row["event"])
            # tokenize=False → 拿 prompt 纯文本；transformers 5.x 里 tokenize=True 会返回
            # BatchEncoding 而不是 tensor，容易踩坑，这里统一用 False 再自己 encode。
            prompt_text = tokenizer.apply_chat_template(
                [{"role": "system", "content": system_prompt},
                 {"role": "user", "content": f"句子：{row['text']}"}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
            prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
            # 目标 = JSON + EOS：**必须带 EOS**，否则模型学不会"在哪里停"，
            # 推理时会一直往下编（生成上限 max_new_tokens 也救不回来）
            response_ids = tokenizer.encode(target, add_special_tokens=False) + \
                [tokenizer.eos_token_id]

            input_ids = (prompt_ids + response_ids)[: max_length]
            labels = ([-100] * len(prompt_ids) + response_ids)[: max_length]
            if len(prompt_ids) >= max_length:
                self.n_prompt_truncated += 1
            elif len(prompt_ids) + len(response_ids) > max_length:
                self.n_target_truncated += 1
            if all(x == -100 for x in labels):
                continue  # 整条都没有可学的 token（prompt 就超长），直接丢弃
            self.samples.append({
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def collate_fn(batch, pad_id: int):
    """
    动态 padding 到本 batch 最长，并且**补在左边**。

    为什么这里必须左 padding（和别的任务不太一样）：
      我们的 loss 只算序列尾部"目标段"的 logits（见 compute_lm_loss）——
      这要求**目标 JSON 对齐在每个序列的最右端**。
      右 padding 会在 JSON 之后再补 pad，使"尾部"变成 pad，只能退回全序列算 logits（显存翻倍）。
      左 padding 相当于把整条序列右对齐，目标段一定在最后，同时用 attention_mask 屏蔽 pad。
      （生成阶段也一样是左 padding，见 local_generate.py。）
    """
    max_len = max(item["input_ids"].size(0) for item in batch)
    input_ids_list, labels_list, mask_list = [], [], []
    for item in batch:
        n = item["input_ids"].size(0)
        pad = max_len - n
        pad_ids = torch.full((pad,), pad_id, dtype=torch.long)
        pad_labels = torch.full((pad,), -100, dtype=torch.long)
        pad_mask = torch.zeros(pad, dtype=torch.long)
        input_ids_list.append(torch.cat([pad_ids, item["input_ids"]]))
        labels_list.append(torch.cat([pad_labels, item["labels"]]))
        mask_list.append(torch.cat([pad_mask, torch.ones(n, dtype=torch.long)]))
    return {"input_ids": torch.stack(input_ids_list),
            "labels": torch.stack(labels_list),
            "attention_mask": torch.stack(mask_list)}

# ══════════════════════════════════════════════════════════════════════════════
# loss：只对「目标段」计算 logits（本任务关键优化）
# ══════════════════════════════════════════════════════════════════════════════

def compute_lm_loss(model, input_ids, attention_mask, labels):
    """
    计算带 loss mask 的语言模型 loss，但**只对序列尾部（目标段）算 logits**。

    为什么可以这么省：
      - loss mask 已经保证只有 assistant 的 JSON 部分是有效 label（其余 -100）；
      - JSON 一定在序列**末尾**（prompt 在前、输出在后），所以反向传播只需要末尾
        `keep` 个位置的 logits，前面 prompt 的 logits 算出来也会被 ignore。
      - Qwen3-0.6B 的词表是 151936：全序列算 logits 时，batch×1280×151936 的
        bf16 logits + 转 fp32 再算交叉熵，光这一块就要 4~5 GB，
        是 8GB 卡上 batch 上不去的真正原因（不是模型权重）。
      - 目标段平均只有 ~103 个 token，只算尾部能把这块显存降到 1/4 以下，
        因此可以把 batch 从 2 放大到 4，整体训练明显更快。

    实现要点（写错就会算错 loss）：
      - causal LM 用位置 t 的 logits 预测 t+1 的 token → logits 要多留 1 位做 shift；
      - `keep = 本 batch 最长目标 token 数 + 1`，动态的，不需要固定常数；
      - labels 取尾部 `keep-1` 位与 `shift_logits` 对齐。
    """
    import torch.nn.functional as F

    # 注意：CausalLM 的输出对象只有 logits / hidden_states（要 last_hidden_state 得去拿 backbone），
    # 这里统一用 output_hidden_states=True 取最后一层（= 过完 final norm 的隐状态，
    # 正是 lm_head 的输入），对「裸模型 / LoRA 包装模型 / 全量微调模型」都成立。
    outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                    use_cache=False, output_hidden_states=True)
    hidden = outputs.hidden_states[-1]                       # [B, L, H]

    n_target = int((labels != -100).sum(dim=1).max().item())  # 本 batch 最长目标段长度
    if n_target <= 0:
        return None                                          # 整批都没有可学 token，调用方跳过
    keep = min(n_target + 1, hidden.size(1))                 # +1 是为了 shift
    logits = model.lm_head(hidden[:, -keep:, :]).float()     # 只对尾部做投影
    shift_logits = logits[:, :-1, :].reshape(-1, logits.size(-1))
    shift_labels = labels[:, -(keep - 1):].reshape(-1)
    return F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)

# ══════════════════════════════════════════════════════════════════════════════
# 训练
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="事件抽取 LLM SFT 训练（LoRA / 全量微调）")
    p.add_argument("--model_path", default=str(resolve_model_path(QWEN_MODEL_DIR)))
    p.add_argument("--data_dir", default=str(DATA_DIR))
    p.add_argument("--output_dir", default=str(OUTPUT_DIR))
    p.add_argument("--num_train", default=-1, type=int, help="-1 = 用全部 5747 条")
    p.add_argument("--epochs", default=1, type=int, help="默认 1 epoch（教学项目够用）")
    p.add_argument("--batch_size", default=2, type=int,
                   help="8GB 卡实测 batch=4/max_length=1280 会 OOM（词表 151936 的 logits 很吃显存）")
    p.add_argument("--grad_accum", default=8, type=int, help="等效 batch = batch_size*grad_accum")
    p.add_argument("--lr", default=None, type=float, help="默认 LoRA=2e-4，全量=2e-5")
    p.add_argument("--max_length", default=1280, type=int,
                   help="默认 1280：实测 prompt 平均 907 token、prompt+目标 p95=1161")
    p.add_argument("--full_ft", action="store_true",
                   help="全量微调（0.6B 全参数；8GB 卡大概率 OOM，只作对比开关）")
    p.add_argument("--lora_r", default=8, type=int)
    p.add_argument("--lora_alpha", default=16, type=int)
    p.add_argument("--lora_dropout", default=0.05, type=float)
    p.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp32"],
                   help="auto = 有 CUDA 就用 bf16（Qwen3 权重原生 bf16）")
    p.add_argument("--grad_ckpt", action="store_true",
                   help="梯度检查点：省约一半激活显存，代价是速度慢约 20%%")
    p.add_argument("--seed", default=SEED, type=int)
    p.add_argument("--log_every", type=int, default=50,
                   help="每多少步往 steps.jsonl 写一行（供学生画 loss 曲线）")
    p.add_argument("--run_name", default="llm_lora",
                   help="训练日志文件名前缀：outputs/logs/<run_name>_*.json")
    p.add_argument("--max_steps", default=0, type=int, help=">0 时只跑这么多优化器步数（冒烟用）")
    p.add_argument("--val_size", default=200, type=int,
                   help="从 train 里留出的验证条数（**不用 dev**，避免用评测集做模型选择）")
    p.add_argument("--std_logits", action="store_true",
                   help="关闭「只算目标段 logits」优化，退回标准 HF loss（慢、费显存，仅作对照）")
    return p.parse_args()

def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.lr is None:
        args.lr = 2e-5 if args.full_ft else 2e-4

    data_dir, output_dir = Path(args.data_dir), Path(args.output_dir)
    ckpt_dir = output_dir / ("sft_full_ckpt" if args.full_ft else "sft_adapter")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device} | 模式: {'全量微调' if args.full_ft else 'LoRA'}")

    # ── 数据：train 里留出 val（绝不拿 dev 选模型）────────────────────────────
    rows = load_jsonl(data_dir / "train_extract.jsonl")
    ordered = canonical_order(rows, args.seed)     # 固定 seed，可复现
    n_val = min(args.val_size, len(ordered) // 20)
    val_rows, train_rows = ordered[:n_val], ordered[n_val:]
    if args.num_train > 0:
        train_rows = train_rows[: args.num_train]
    print(f"训练 {len(train_rows)} 条 | 验证(取自 train 尾部) {len(val_rows)} 条")

    schema = load_schema()
    system_prompt = build_system_prompt(schema)

    # ── tokenizer / model ────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    torch_dtype = {"auto": torch.bfloat16 if device.type == "cuda" else torch.float32,
                   "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    print(f"加载基座: {args.model_path}（dtype={torch_dtype}）")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch_dtype, trust_remote_code=True,
    )
    model.config.use_cache = False   # 训练时不需要 KV cache，关掉省显存

    train_ds = EventSFTDataset(train_rows, tokenizer, system_prompt, args.max_length)
    val_ds = EventSFTDataset(val_rows, tokenizer, system_prompt, args.max_length)
    print(f"prompt 被完全截断（丢弃）：{train_ds.n_prompt_truncated} 条 | "
          f"目标尾部被截断：{train_ds.n_target_truncated} 条（max_length={args.max_length}）")
    if train_ds.n_prompt_truncated > 0:
        print("[警告] 有样本 prompt 超过 max_length，labels 全 -100 → 白跑。"
              "请调大 --max_length（本任务建议 ≥1280）")

    _collate = lambda b: collate_fn(b, tokenizer.pad_token_id)   # noqa: E731
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=_collate, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=_collate)

    # ── LoRA / 全量 ──────────────────────────────────────────────────────────
    if args.full_ft:
        total = sum(p.numel() for p in model.parameters())
        print(f"trainable params: {total:,} || all params: {total:,} || trainable%: 100.0")
    else:
        if not PEFT_AVAILABLE:
            raise ImportError("LoRA 模式需要 peft：pip install peft>=0.14.0")
        # target_modules 用 Qwen 系列的注意力投影层名（q/k/v/o_proj）；
        # 只挂 attention 是最省参数、性价比最高的做法（挂 MLP 显存和训练时间都会涨）
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=args.lora_r, lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=args.lora_dropout, bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    if args.grad_ckpt:
        model.gradient_checkpointing_enable()
    model = model.to(device)

    optimizer = AdamW([p for p in model.parameters() if p.requires_grad],
                      lr=args.lr, weight_decay=0.01)
    steps_per_epoch = max(1, len(train_loader) // args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    if args.max_steps > 0:
        total_steps = min(total_steps, args.max_steps)
    print(f"训练步数: {total_steps}（每 epoch {steps_per_epoch} 步，"
          f"等效 batch={args.batch_size * args.grad_accum}，lr={args.lr}）\n")

    # ── 训练循环 ─────────────────────────────────────────────────────────────
    # 生成式链路的训练日志：与判别式共用 common/log_utils.py，字段口径一致，
    # 这样 eval/plot_compare.py 能把两边的 loss 曲线画在一张图里。
    logger = TrainLogger(method=getattr(args, "run_name", None) or "llm_lora", args=args, root=ROOT,
                         params=sum(p.numel() for p in model.parameters() if p.requires_grad),
                         notes='LoRA SFT（Qwen3-0.6B 生成式链路）')

    history, global_step = [], 0
    best_val_loss = float("inf")
    t_train0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss, running_tokens, step_log = 0.0, 0, []
        optimizer.zero_grad()
        t0 = time.time()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", ncols=100)
        for step, batch in enumerate(pbar, 1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            out = None
            if args.std_logits:
                out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = out.loss
            else:
                # 快路径：只对尾部目标段算 logits（见 compute_lm_loss 的说明）
                loss = compute_lm_loss(model, input_ids, attention_mask, labels)
                if loss is None:
                    optimizer.zero_grad()
                    continue
            if torch.isnan(loss) or torch.isinf(loss):
                # 出现 nan 基本只有一个原因：这一批 labels 全是 -100（prompt 就超长）。
                # 直接跳过这一批，比让 nan 污染全部参数好得多。
                print("[警告] loss=nan，跳过该 batch（检查 max_length 是否够大）")
                optimizer.zero_grad()
                continue

            logger.log_step(epoch=epoch, step=global_step, loss=float(loss.item()),
                            lr=optimizer.param_groups[0]["lr"])
            (loss / args.grad_accum).backward()
            if step % args.grad_accum == 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                step_log.append(round(loss.item(), 4))
                if global_step % 20 == 0:
                    pbar.set_postfix(loss=f"{loss.item():.4f}", step=global_step)
                if args.max_steps > 0 and global_step >= args.max_steps:
                    break

            n_tokens = int((labels != -100).sum().item())
            running_loss += loss.item() * n_tokens
            running_tokens += n_tokens
        train_loss = running_loss / max(running_tokens, 1)
        elapsed = time.time() - t0

        # ── 验证 loss（监控用；1 epoch 时没有"选点"问题）────────────────────
        model.eval()
        v_loss, v_tokens = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                out = None
                if args.std_logits:
                    out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    val_batch_loss = out.loss
                else:
                    val_batch_loss = compute_lm_loss(model, input_ids, attention_mask, labels)
                n_tokens = int((labels != -100).sum().item())
                if val_batch_loss is None:
                    continue
                v_loss += val_batch_loss.item() * n_tokens
                v_tokens += n_tokens
        val_loss = v_loss / max(v_tokens, 1)

        print(f"Epoch {epoch}/{args.epochs} | train_loss={train_loss:.4f} "
              f"val_loss={val_loss:.4f} | {elapsed:.0f}s（{elapsed / max(global_step, 1):.2f}s/step）")
        # 每 epoch 也写一条结构化记录（与判别式方法同口径：train_loss + 验证 + 耗时）
        logger.log_epoch(epoch=epoch, train_loss=train_loss, elapsed=elapsed,
                         val_metrics={"val_loss": round(val_loss, 4)},
                         extra={"optimizer_steps": global_step})
        history.append({"epoch": epoch, "train_loss": round(train_loss, 4),
                        "val_loss": round(val_loss, 4), "elapsed_s": round(elapsed, 1),
                        "optimizer_steps": global_step})
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_pretrained(str(ckpt_dir))
            tokenizer.save_pretrained(str(ckpt_dir))
            print(f"  ✓ {('完整模型' if args.full_ft else 'LoRA adapter')}已保存 → {ckpt_dir}")
        if args.max_steps > 0 and global_step >= args.max_steps:
            print(f"  已达 --max_steps={args.max_steps}，结束")
            break

    total_s = time.time() - t_train0
    log = {
        "mode": "full_ft" if args.full_ft else "lora",
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "num_train": len(train_ds), "num_val": len(val_ds),
        "prompt_truncated": train_ds.n_prompt_truncated,
        "target_truncated": train_ds.n_target_truncated,
        "optimizer_steps": global_step, "history": history,
        "best_val_loss": round(best_val_loss, 4),
        "total_train_s": round(total_s, 1),
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "all_params": sum(p.numel() for p in model.parameters()),
        "gpu_peak_mem_gib": round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2)
        if torch.cuda.is_available() else 0,
    }
    log_path = output_dir / ("train_log_full_ft.json" if args.full_ft else "train_log_sft.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)
    logger.finalize(total_s=time.time() - t_train0,
                    optimizer_steps=global_step,
                    best_val_loss=best_val_loss,
                    peak_gpu_mb=peak_gpu_mb())
    print(f"\n训练完成：{total_s / 60:.1f} 分钟，峰值显存 {log['gpu_peak_mem_gib']} GiB")
    print(f"日志 → {log_path}\ncheckpoint → {ckpt_dir}")
    print("\n下一步：python llm_lora/evaluate_sft.py --ckpt_dir", ckpt_dir)

if __name__ == "__main__":
    main()
