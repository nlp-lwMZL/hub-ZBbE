#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
基础模型：Qwen2.5-0.5B-Instruct
数据：sft_train.jsonl / sft_val.jsonl，每行 {"instruction": "...", "output": "..."}
流程：
  1. 读数据，用模型的 chat 模板把 (指令, 回答) 拼成一段完整对话文本
  2. tokenize，并把「指令部分」的 label 设成 -100（只学怎么答，不学怎么问）
  3. 冻结基座权重，只训练 LoRA 低秩矩阵
  4. 手写训练循环：前向 -> 反向 -> 梯度累积 -> 更新，cosine 学习率
  5. 在验证集上算 loss，最后对比「训练前 / 训练后」的回答，保存 adapter

用法：
    在同目录下放models/Qwen2.5-0.5B-Instruct、stf_train.jsonl、sft_val.jsonl
"""
import argparse
import json
import math
import time

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

SYSTEM_PROMPT = "你是一个乐于助人的中文 AI 助手。回答要简洁、准确。"
IGNORE_INDEX = -100  # CrossEntropyLoss 会忽略这个标签，即"这部分不算 loss"


# ----------------------------- 1. 数据 -> 训练张量 -----------------------------
def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def encode_example(tokenizer, instruction, output, max_len):
    """把一条样本编码成 input_ids + labels；instruction 部分 label=-100。"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": instruction},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    full = prompt + output + "<im_end>\n"
    ids = tokenizer(full, add_special_tokens=False)["input_ids"]
    prompt_len = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    if len(ids) > max_len:          # 超长就截断（本数据最长 126 token，基本用不到）
        ids = ids[:max_len]
    if len(ids) <= prompt_len:      # 截断到只剩指令，丢弃这条
        return None
    labels = [IGNORE_INDEX] * prompt_len + ids[prompt_len:]
    return {"input_ids": ids, "labels": labels}


def collate(items, pad_id):
    """batch 内动态 pad：input_ids 补 pad_id，labels 补 -100，attention_mask 补 0。"""
    max_len = max(len(x["input_ids"]) for x in items)
    input_ids = torch.full((len(items), max_len), pad_id, dtype=torch.long)
    labels = torch.full((len(items), max_len), IGNORE_INDEX, dtype=torch.long)
    mask = torch.zeros((len(items), max_len), dtype=torch.long)
    for i, x in enumerate(items):
        n = len(x["input_ids"])
        input_ids[i, :n] = torch.tensor(x["input_ids"])
        labels[i, :n] = torch.tensor(x["labels"])
        mask[i, :n] = 1
    return {"input_ids": input_ids, "labels": labels, "attention_mask": mask}


def make_batches(encoded, batch_size, pad_id):
    """按长度排序后分批，尽量减少 padding 浪费（CPU 上很关键）。"""
    ordered = sorted(encoded, key=lambda x: len(x["input_ids"]))
    return [collate(ordered[i:i + batch_size], pad_id)
            for i in range(0, len(ordered), batch_size)]


# ----------------------------- 2. 评估与生成 -----------------------------
@torch.no_grad()
def evaluate_loss(model, encoded, batch_size, pad_id):
    """验证集平均 loss（只统计回答部分的 token）。"""
    model.eval()
    total, count = 0.0, 0
    for batch in make_batches(encoded, batch_size, pad_id):
        out = model(**batch)
        n = int((batch["labels"] != IGNORE_INDEX).sum())
        total += float(out.loss) * n
        count += n
    model.train()
    return total / max(count, 1)


@torch.no_grad()
def generate_answer(model, tokenizer, instruction, max_new_tokens=24):
    """用当前模型（基座或基座+LoRA）回答一个问题。"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": instruction},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    model.eval()
    # 单独给一份 generation 配置：贪心解码、不采样（否则会被 Qwen 自带的采样参数干扰）
    gen_cfg = GenerationConfig(
        do_sample=False, max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
    )
    output = model.generate(**inputs, generation_config=gen_cfg)
    new_tokens = output[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).split("<im_end>")[0].strip()


# ----------------------------- 3. 主流程 -----------------------------
def main():
    ap = argparse.ArgumentParser(description="单文件 LoRA SFT（CPU）")
    ap.add_argument("--model-dir", default=r"E:\learn\ML\sft_train\models\Qwen2.5-0.5B-Instruct")
    ap.add_argument("--train-file", default=r"E:\learn\ML\sft_train\sft_train.jsonl")
    ap.add_argument("--val-file", default=r"E:\learn\ML\sft_train\sft_val.jsonl")
    ap.add_argument("--output-dir", default=r"E:\learn\ML\sft_train\adapter")
    ap.add_argument("--max-samples", type=int, default=300, help="用多少条训练（0=全量 3000）")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4, help="梯度累积，等效 batch=4*4=16")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-len", type=int, default=160)
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--num-threads", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.set_num_threads(args.num_threads)
    t_start = time.time()

    # 数据
    train_rows = read_jsonl(args.train_file)
    if args.max_samples > 0:
        import random
        random.seed(args.seed)
        train_rows = random.sample(train_rows, min(args.max_samples, len(train_rows)))
    val_rows = read_jsonl(args.val_file)[:100]
    print(f"[数据] 训练 {len(train_rows)} 条，验证 {len(val_rows)} 条")

    # 模型 + tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(args.model_dir, torch_dtype=torch.float32)
    base_model.config.use_cache = False
    print(f"[模型] 参数量 {sum(p.numel() for p in base_model.parameters()) / 1e6:.0f}M，CPU fp32")

    # 编码数据（指令部分 label=-100）
    train_enc = [e for e in (encode_example(tokenizer, r["instruction"], r["output"], args.max_len)
                             for r in train_rows) if e]
    val_enc = [e for e in (encode_example(tokenizer, r["instruction"], r["output"], args.max_len)
                           for r in val_rows) if e]
    avg_len = sum(len(e["input_ids"]) for e in train_enc) / max(len(train_enc), 1)
    print(f"[编码] 平均 {avg_len:.0f} token/条（含完整对话模板）")

    # 固定问题：训练前后各回答一次，用于直观对比
    demo_questions = ["计算 128 加 356 等于多少？", "把这句话改写成被动语态：Tom 修好了电脑。"]
    print("\n[训练前] 基座模型回答：")
    for q in demo_questions:
        print(f"  Q: {q}\n  A: {generate_answer(base_model, tokenizer, q)}")

    # 挂 LoRA（只训练低秩矩阵，基座参数冻结）
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=args.lora_r, lora_alpha=2 * args.lora_r,
        lora_dropout=0.05, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(base_model, lora_cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[LoRA] r={args.lora_r}，可训练参数 {trainable / 1e6:.2f}M（"
          f"{100 * trainable / sum(p.numel() for p in model.parameters()):.2f}%）")

    print(f"[验证] 训练前 val_loss = {evaluate_loss(model, val_enc, args.batch_size, tokenizer.pad_token_id):.4f}")

    # 优化器 + cosine 学习率
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.0)
    steps_per_epoch = math.ceil(len(train_enc) / (args.batch_size * args.grad_accum))
    total_steps = max(steps_per_epoch * args.epochs, 1)
    warmup_steps = max(1, int(0.05 * total_steps))

    def lr_at(step):
        if step < warmup_steps:
            return args.lr * (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return args.lr * 0.5 * (1 + math.cos(math.pi * progress))

    print(f"[训练] {total_steps} 个优化步，等效 batch {args.batch_size * args.grad_accum}，"
          f"每步约 {len(train_enc) / max(total_steps, 1):.0f} 条")
    model.train()
    global_step, t0 = 0, time.time()
    for epoch in range(args.epochs):
        micro_batch_index = 0
        optimizer.zero_grad(set_to_none=True)
        for batch in make_batches(train_enc, args.batch_size, tokenizer.pad_token_id):
            out = model(**batch)
            loss = out.loss
            (loss / args.grad_accum).backward()
            micro_batch_index += 1
            if micro_batch_index % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                lr_now = lr_at(global_step)
                for g in optimizer.param_groups:
                    g["lr"] = lr_now
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if global_step % 5 == 0 or global_step == total_steps:
                    print(f"  step {global_step}/{total_steps} | loss {float(loss):.4f} | "
                          f"lr {lr_now:.2e} | 用时 {time.time() - t0:.0f}s")
        # 每个 epoch 结束再做一次验证
        val_loss = evaluate_loss(model, val_enc, args.batch_size, tokenizer.pad_token_id)
        print(f"  [验证] epoch {epoch + 1} 结束 val_loss = {val_loss:.4f}")

    # 保存 LoRA adapter（只有几十 MB，可随时加载/卸载）
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"[保存] adapter -> {args.output_dir}")

    # 训练后回答
    print("\n[训练后] LoRA 模型回答：")
    for q in demo_questions:
        print(f"  Q: {q}\n  A: {generate_answer(model, tokenizer, q)}")

    print(f"\n完成：训练 {global_step} 步，总用时 {(time.time() - t_start) / 60:.1f} 分钟")


if __name__ == "__main__":
    main()
