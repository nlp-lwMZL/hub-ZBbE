"""
LLM SFT（LoRA 指令微调）—— 人民日报 NER 版

参考 ../src_llm/train_sft.py，针对 data/peoples_daily 重写。改动的地方：

  1. **数据来源**：cluener 的 span 格式 → CoNLL 的 BIO 格式，用 common.bio_to_entities()
     转成和 cluener 一样的 {"entities": [...]} 目标串。
  2. **实体类型**：10 类小写英文 → 3 类大写（PER / ORG / LOC）。
  3. **max_length 256 → 384**：实测「prompt + JSON 目标」总长 P99=321、max=356，
     用 256 会截断约 3% 的样本，而截断的是**目标串的尾巴**（JSON 直接被切掉一半），
     比截断输入文本恶劣得多。脚本里会显式统计截断条数并告警。
  4. **输出目录**：outputs/ → outputs_llm_work/，和 BERT 的 outputs_work/ 完全隔离。
  5. **默认规模**：10000 条 × 2 轮（全量 20864 条 × 3 轮约 5 小时，这里取了折中）。

模型输出示例（3 类实体，比 cluener 的 10 类短很多）：
  {"entities": [{"text": "上海市", "type": "LOC"}, {"text": "韩正", "type": "PER"}]}

注意：本数据集 **35% 的句子没有实体**，目标就是 {"entities": []}。
这一类样本如果采样不足，模型会倾向于「见词就标」，precision 会崩，所以训练时不做过滤。

使用方式：
  python src_llm_work/train_sft_work.py                        # 默认 10000 条 × 2 轮
  python src_llm_work/train_sft_work.py --num_train 200 --epochs 1   # 冒烟
  python src_llm_work/train_sft_work.py --num_train -1 --epochs 3     # 全量
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# 6GB 卡跑 fp32 的 0.5B + 384 长度，显存很紧。expandable_segments 让分配器按需扩展
# 显存段，而不是被碎片卡住 —— 不开的话会看到 CUDACachingAllocator 的 OOM 告警。
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

try:
    from peft import get_peft_model, LoraConfig, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

from common import (
    MODEL_PATH, DATA_DIR, OUT_DIR, ADAPTER_DIR, LOG_DIR,
    SYSTEM_PROMPT, record_to_target, load_split,
)


# ══════════════════════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════════════════════

class SFTDataset(Dataset):
    """
    一条样本 = chat 格式的 prompt + JSON 目标。

    Loss mask 结构（和分类任务、cluener 版完全一致）：
      ┌──────────────────────────────────────────────────────┐
      │ <|im_start|>system ... <|im_start|>user {原文}       │  → -100，不算 loss
      │ <|im_start|>assistant                                │  → -100
      │ {"entities": [...]} <EOS>                            │  → 真实 id，只在这里算 loss
      └──────────────────────────────────────────────────────┘
    只学「怎么输出」，不学「怎么读输入」—— 输入是已经知道的东西，让它算 loss 是浪费。

    两个和参考实现不同的做法：
      1. **在 __init__ 里一次性 tokenize 完并缓存**。参考实现每次 __getitem__ 都重新
         tokenize，训练 N 个 epoch 就重复 N 遍，纯浪费 CPU。
      2. **超长的样本直接丢掉，不截断**。参考实现用 [:max_length] 硬切，切掉的是
         目标 JSON 的尾巴 —— 等于教模型输出残缺 JSON。宁可少 0.5% 的样本，
         也不要往训练集里塞坏标签。丢了多少条会打出来。
    """

    def __init__(self, data, tokenizer, max_length=384):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.items: list[tuple[list[int], list[int]]] = []
        self.n_dropped = 0

        for item in data:
            text = "".join(item["tokens"])
            target = record_to_target(item)

            prompt_text = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": text},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
            response_ids = (
                tokenizer.encode(target, add_special_tokens=False)
                + [tokenizer.eos_token_id]
            )

            if len(prompt_ids) + len(response_ids) > max_length:
                self.n_dropped += 1
                continue

            self.items.append((
                prompt_ids + response_ids,
                [-100] * len(prompt_ids) + response_ids,
            ))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        input_ids, labels = self.items[idx]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels":    torch.tensor(labels,    dtype=torch.long),
        }


def collate_fn(batch, pad_id):
    """右侧 padding；labels 用 -100 补齐（不算 loss），attention_mask 手搓。"""
    max_len = max(item["input_ids"].size(0) for item in batch)
    input_ids_list, labels_list, mask_list = [], [], []
    for item in batch:
        n = item["input_ids"].size(0)
        pad = max_len - n
        input_ids_list.append(torch.cat([item["input_ids"],
                                         torch.full((pad,), pad_id, dtype=torch.long)]))
        labels_list.append(torch.cat([item["labels"],
                                      torch.full((pad,), -100, dtype=torch.long)]))
        mask_list.append(torch.cat([torch.ones(n, dtype=torch.long),
                                    torch.zeros(pad, dtype=torch.long)]))
    return {
        "input_ids":      torch.stack(input_ids_list),
        "labels":         torch.stack(labels_list),
        "attention_mask": torch.stack(mask_list),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 训练
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description="LLM SFT NER 训练（人民日报，LoRA）")
    parser.add_argument("--model_path", default=str(MODEL_PATH))
    parser.add_argument("--num_train",  default=10000, type=int,
                        help="训练样本数，-1 用全部 20864 条")
    parser.add_argument("--num_val",    default=300,   type=int,
                        help="每轮验证用的条数（只算 val_loss 选最优，不做 F1）")
    parser.add_argument("--epochs",     default=2,     type=int)
    # batch=2 + accum=8 = 有效 batch 16，和参考实现的 4x4 一样，但峰值显存从
    # 6.88 GB（超过 6.14 GB 物理显存，走共享内存）降到 4.36 GB，而且每步更快
    parser.add_argument("--batch_size", default=2,     type=int)
    parser.add_argument("--grad_accum", default=8,     type=int)
    parser.add_argument("--lr",         default=2e-4,  type=float)
    parser.add_argument("--max_length", default=384,   type=int,
                        help="prompt+目标的最大长度；本数据集 P99=321，384 基本不截断")
    parser.add_argument("--lora_r",     default=8,     type=int)
    parser.add_argument("--lora_alpha", default=16,    type=int)
    parser.add_argument("--seed",       default=42,    type=int)
    parser.add_argument("--grad_checkpoint", action="store_true", default=True,
                        help="梯度检查点（默认开）；--no-grad_checkpoint 可关闭")
    parser.add_argument("--no-grad_checkpoint", dest="grad_checkpoint", action="store_false")
    parser.add_argument("--ckpt_name",  default="sft_adapter",
                        help="adapter 子目录名；冒烟测试请换一个，别覆盖正式权重")
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    ckpt_dir = OUT_DIR / args.ckpt_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # ── 数据 ──────────────────────────────────────────────────────────────────
    train_raw = load_split("train")
    val_raw = load_split("validation")
    if args.num_train > 0:
        train_raw = random.Random(args.seed).sample(
            train_raw, min(args.num_train, len(train_raw)))
    print(f"训练集: {len(train_raw)} 条 | 验证（选模型用）: {min(args.num_val, len(val_raw))} 条")

    # ── Tokenizer ────────────────────────────────────────────────────────────
    print(f"\n加载 tokenizer: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        str(Path(args.model_path).resolve()), trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ── Dataset / DataLoader ─────────────────────────────────────────────────
    train_dataset = SFTDataset(train_raw, tokenizer, args.max_length)
    val_dataset = SFTDataset(val_raw[: args.num_val], tokenizer, args.max_length)
    for ds, name in [(train_dataset, "train"), (val_dataset, "val")]:
        if ds.n_dropped:
            print(f"  [提示] {name} 有 {ds.n_dropped} 条超过 max_length={args.max_length}，"
                  f"已丢弃（不截断，避免教模型输出残缺 JSON）")
    print(f"  实际入训: train {len(train_dataset)} 条 | val {len(val_dataset)} 条")

    _collate = lambda b: collate_fn(b, tokenizer.pad_token_id)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, collate_fn=_collate)
    # 验证没有 backward，logits 不存梯度，可以放心用 4；再大会撑不住长序列
    val_loader = DataLoader(val_dataset, batch_size=max(4, args.batch_size),
                            shuffle=False, collate_fn=_collate)

    # ── 模型 ──────────────────────────────────────────────────────────────────
    print(f"加载 base model: {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        str(Path(args.model_path).resolve()),
        dtype=torch.float32,      # transformers 5.x 用 dtype=，4.x 用 torch_dtype=
        trust_remote_code=True,
    )

    if not PEFT_AVAILABLE:
        raise ImportError("需要 peft 库：pip install peft")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 梯度检查点：不存中间激活，反向时重算一遍。显存省一半以上，代价是慢 20~30%。
    # 不开这个的话，batch=2、长度 384 时显存会顶到 5.8 GB（物理只有 6 GB），
    # 驱动开始往系统内存倒页 —— 实测单步从 0.5 s 涨到 3 分钟，看着像死机。
    if args.grad_checkpoint:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()   # PEFT 冻结了 embedding，得手动打开输入梯度
        model.config.use_cache = False       # 和检查点不兼容，训练时本来也用不到 KV cache
        print("已启用梯度检查点")

    model = model.to(device)

    # ── 优化器 ────────────────────────────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs // args.grad_accum
    print(f"总训练步数: {total_steps}（batch={args.batch_size}, grad_accum={args.grad_accum}, "
          f"epochs={args.epochs}, lr={args.lr}, max_length={args.max_length}）\n")

    # ── 训练循环 ──────────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    log_records = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, total_tokens = 0.0, 0
        optimizer.zero_grad()
        t0 = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]", leave=False)
        for step, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            (loss / args.grad_accum).backward()
            if (step + 1) % args.grad_accum == 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            n_tokens = (labels != -100).sum().item()
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = total_loss / max(total_tokens, 1)

        # ── 验证 loss（只用来挑最优 epoch，F1 在 evaluate_sft_work.py 里算）────
        model.eval()
        if device.type == "cuda":
            torch.cuda.empty_cache()   # 先把训练阶段的碎片还给驱动，否则长序列验证容易 OOM
        val_loss, val_tokens = 0.0, 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Val", leave=False):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                n_tokens = (labels != -100).sum().item()
                val_loss += outputs.loss.item() * n_tokens
                val_tokens += n_tokens
        avg_val_loss = val_loss / max(val_tokens, 1)

        elapsed = time.time() - t0
        peak_gb = (torch.cuda.max_memory_allocated() / 1024 ** 3) if device.type == "cuda" else 0.0
        print(f"Epoch {epoch}/{args.epochs} | train_loss={avg_train_loss:.4f}  "
              f"val_loss={avg_val_loss:.4f} | {elapsed:.0f}s | 峰值显存 {peak_gb:.2f} GB")

        log_records.append({
            "epoch": epoch, "train_loss": round(avg_train_loss, 6),
            "val_loss": round(avg_val_loss, 6), "elapsed_s": round(elapsed, 1),
        })

        # 每个 epoch 都落盘，中途被打断也能看到进度
        log_path = LOG_DIR / "train_sft.json"
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(log_records, f, ensure_ascii=False, indent=2)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            print(f"  ★ 最优 adapter 已保存 → {ckpt_dir}  (val_loss={avg_val_loss:.4f})")

    print(f"\n训练完成。最优 val_loss={best_val_loss:.4f}")
    print(f"  训练日志:   {LOG_DIR / 'train_sft.json'}")
    print(f"  LoRA adapter: {ckpt_dir}")
    print(f"\n下一步：python src_llm_work/evaluate_sft_work.py")


if __name__ == "__main__":
    main()
