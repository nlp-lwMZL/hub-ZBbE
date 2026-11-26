"""
LLM SFT（监督微调）训练脚本 — 基于 LoRA 高效微调 Qwen2-0.5B-Instruct

教学重点：
  1. 指令微调格式：把分类任务转化为 system/user/assistant chat 格式
  2. Loss masking：只在 assistant 输出部分（类别名）计算 cross-entropy loss
     prompt 部分 labels 全设为 -100，PyTorch 自动忽略
  3. LoRA 原理：冻结原始权重，旁路低秩矩阵 ΔW = B·A，仅训练约 0.5% 参数
  4. 与 BERT fine-tune 对比：生成式分类 vs 判别式分类，各自适用场景

使用方式：
  python train_sft.py                        # LoRA 微调，5000 条快速演示（默认）
  python train_sft.py --num_train -1         # LoRA 微调，全部 53K 条
  python train_sft.py --epochs 1             # 快速验证流程
  python train_sft.py --lora_r 16            # 增大 LoRA rank，更多参数

  # ── 全量微调（Full Fine-Tuning） ──────────────────────────────────────────
  python train_sft.py --full_ft --lr 2e-5    # 全量微调，推荐显存 ≥ 16GB
  python train_sft.py --full_ft --lr 2e-5 --num_train 5000

  全量微调与 LoRA 的区别：
    --full_ft   不加载 LoRA adapter，所有 495M 参数均参与更新
    --lr 2e-5   全参更新需要更小学习率（LoRA 默认 2e-4 会破坏预训练权重）
    checkpoint  保存完整模型到 outputs/sft_full_ckpt/（可直接用 from_pretrained 加载）

  显存估算（Qwen2-0.5B，batch=1，seq=128）：
    LoRA r=8  ≈  3GB（RTX 4060 8GB 轻松运行）
    全量微调  ≈  8~10GB（需要 A100/4090 或开启 gradient_checkpointing）

依赖：
  pip install torch transformers peft tqdm   # LoRA 模式
  pip install torch transformers tqdm        # 全量微调模式（不需要 peft）
"""

import os
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

# Windows 多进程 OpenMP 冲突规避
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT        = Path(__file__).parent.parent
DATA_DIR    = ROOT / "data"
MODEL_PATH  = ROOT.parent.parent / "pretrain_models" / "Qwen2-0.5B-Instruct"
OUTPUT_DIR  = ROOT / "outputs"
ADAPTER_DIR = OUTPUT_DIR / "sft_adapter"

LABEL_NAMES = [
    "故事", "文化", "娱乐", "体育", "财经",
    "房产", "汽车", "教育", "科技", "军事",
    "旅游", "国际", "证券", "农业", "电竞",
]

SYSTEM_PROMPT = (
    "你是一个新闻标题分类助手。请将给定的新闻标题分类到以下类别之一，"
    "只输出类别名称，不要输出任何其他内容。\n"
    "可选类别：" + "、".join(LABEL_NAMES)
)


# ══════════════════════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════════════════════

class SFTDataset(Dataset):
    """
    把分类数据转换为 chat-format 的 SFT 训练样本。

    每条样本的 labels 构成：
      ┌─────────────────────────────────────────────┐
      │ <system> ... <user> ... <assistant>\\n        │  → labels = -100（不计算 loss）
      │ 科技 <|im_end|>                              │  → labels = token id（计算 loss）
      └─────────────────────────────────────────────┘

    这是 SFT 与 pretraining 的核心区别：
      - Pretrain：对所有 token 计算 loss
      - SFT：只对模型"应该生成"的部分（assistant 回复）计算 loss
    """

    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        label_name = LABEL_NAMES[item["label"]]

        # ── Step 1：构建 prompt（system + user），末尾追加 <|im_start|>assistant\n ──
        # 用 tokenize=False 拿到文本串，再 encode——避免 transformers 5.x 中
        # apply_chat_template(tokenize=True) 返回 BatchEncoding 而非 list 的问题
        prompt_text = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": "新闻标题：" + item["sentence"] + "\n类别："},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        prompt_len = len(prompt_ids)

        # ── Step 2：response = 类别名 + EOS ────────────────────────────────────
        response_ids = (
            self.tokenizer.encode(label_name, add_special_tokens=False)
            + [self.tokenizer.eos_token_id]
        )

        # ── Step 3：拼接完整序列，截断至 max_length ────────────────────────────
        input_ids = (prompt_ids + response_ids)[: self.max_length]

        # ── Step 4：loss mask：prompt 部分全设 -100 ────────────────────────────
        # PyTorch CrossEntropyLoss(ignore_index=-100) 会自动跳过这些位置
        labels = ([-100] * prompt_len + response_ids)[: self.max_length]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels":    torch.tensor(labels,    dtype=torch.long),
        }


def collate_fn(batch, pad_id):
    """右填充（padding）使同批次序列等长。"""
    max_len = max(item["input_ids"].size(0) for item in batch)

    input_ids_list, labels_list, mask_list = [], [], []
    for item in batch:
        n   = item["input_ids"].size(0)
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
# Training
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description="LLM SFT 文本分类训练（LoRA / 全量微调）")
    parser.add_argument("--model_path",  default=str(MODEL_PATH))
    parser.add_argument("--data_dir",    default=str(DATA_DIR))
    parser.add_argument("--output_dir",  default=str(OUTPUT_DIR))
    parser.add_argument("--num_train",   default=5000, type=int,
                        help="训练样本数，-1 使用全部 53K 条")
    parser.add_argument("--epochs",      default=3,    type=int)
    parser.add_argument("--batch_size",  default=4,    type=int)
    parser.add_argument("--grad_accum",  default=4,    type=int,
                        help="梯度累积步数，等效 batch = batch_size × grad_accum")
    parser.add_argument("--lr",          default=None, type=float,
                        help="学习率；默认 LoRA=2e-4，全量微调=2e-5（自动判断）")
    parser.add_argument("--max_length",  default=128,  type=int)
    # ── 全量微调开关 ──────────────────────────────────────────────────────────
    parser.add_argument("--full_ft",     action="store_true",
                        help="全量微调：跳过 LoRA，更新所有 495M 参数（需显存 ≥ 16GB）")
    # ── LoRA 超参（full_ft 时忽略）────────────────────────────────────────────
    parser.add_argument("--lora_r",      default=8,    type=int,
                        help="LoRA rank：越大参数越多，效果越强，显存越大")
    parser.add_argument("--lora_alpha",  default=16,   type=int,
                        help="缩放因子，有效学习率 ≈ lr × alpha/r")
    parser.add_argument("--seed",        default=42,   type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 学习率：未指定时按模式自动选默认值
    if args.lr is None:
        args.lr = 2e-5 if args.full_ft else 2e-4

    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    # 全量微调保存完整模型；LoRA 只保存 adapter，用不同目录区分
    ckpt_dir   = output_dir / ("sft_full_ckpt" if args.full_ft else "sft_adapter")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode_str = "全量微调（Full Fine-Tuning）" if args.full_ft else "LoRA 微调"
    print(f"使用设备: {device}  |  微调模式: {mode_str}")

    # ── 加载数据 ──────────────────────────────────────────────────────────────
    with open(data_dir / "train.json", encoding="utf-8") as f:
        train_raw = json.load(f)
    with open(data_dir / "val.json", encoding="utf-8") as f:
        val_raw = json.load(f)

    if args.num_train > 0:
        train_raw = random.sample(train_raw, min(args.num_train, len(train_raw)))
    print(f"训练集: {len(train_raw)} 条 | 验证集（前500条）: 500 条")

    # ── 加载 Tokenizer ─────────────────────────────────────────────────────────
    print(f"\n加载 tokenizer: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        str(Path(args.model_path).resolve()),
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ── 构建数据集 ─────────────────────────────────────────────────────────────
    train_dataset = SFTDataset(train_raw, tokenizer, args.max_length)
    val_dataset   = SFTDataset(val_raw[:500], tokenizer, args.max_length)

    _collate = lambda b: collate_fn(b, tokenizer.pad_token_id)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, collate_fn=_collate)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size * 2,
                              shuffle=False, collate_fn=_collate)

    # ── 加载模型 ───────────────────────────────────────────────────────────────
    print(f"加载 base model: {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        str(Path(args.model_path).resolve()),
        dtype=torch.float32,  # CPU/GPU fp32；transformers 5.x 用 dtype= 不用 torch_dtype=
        trust_remote_code=True,
    )

    # ── LoRA 或 全量微调，二选一 ───────────────────────────────────────────────
    if args.full_ft:
        # 全量微调：所有参数均可更新，无需任何额外配置
        # 对比 LoRA 的教学点：
        #   - 优点：无近似误差，上限更高；checkpoint 是标准 HuggingFace 格式，直接 from_pretrained 加载
        #   - 缺点：显存需求约为 LoRA 的 3~4 倍（优化器状态占大头）；学习率必须更小（2e-5）
        total_params     = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"trainable params: {trainable_params:,} || "
              f"all params: {total_params:,} || trainable%: 100.0000")
    else:
        # LoRA：冻结原始权重，只训练旁路低秩矩阵 ΔW = B·A
        #   A ∈ R^{r×d}，B ∈ R^{d×r}，r << d，参数量约为全量的 0.2%
        if not PEFT_AVAILABLE:
            raise ImportError("LoRA 模式需要 peft 库：pip install peft>=0.14.0")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()  # 打印可训练参数占比（教学关键点）

    model = model.to(device)

    # ── 优化器 ────────────────────────────────────────────────────────────────
    optimizer   = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs // args.grad_accum
    print(f"总训练步数: {total_steps}（batch={args.batch_size}, "
          f"grad_accum={args.grad_accum}, epochs={args.epochs}, lr={args.lr}）\n")

    # ── 训练循环 ──────────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    log_records   = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, total_tokens = 0.0, 0
        optimizer.zero_grad()
        t0 = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]", leave=False)
        for step, batch in enumerate(pbar):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            # forward：HuggingFace CausalLM 内部自动处理 -100 的 loss masking
            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels)
            loss = outputs.loss

            (loss / args.grad_accum).backward()
            if (step + 1) % args.grad_accum == 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            # 按非 -100 token 数加权统计 loss（分母是真实 label token 数）
            n_tokens      = (labels != -100).sum().item()
            total_loss   += loss.item() * n_tokens
            total_tokens += n_tokens
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = total_loss / max(total_tokens, 1)

        # ── 验证 loss ─────────────────────────────────────────────────────────
        model.eval()
        val_loss, val_tokens = 0.0, 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Val", leave=False):
                input_ids      = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels         = batch["labels"].to(device)
                outputs = model(input_ids=input_ids,
                                attention_mask=attention_mask,
                                labels=labels)
                n_tokens   = (labels != -100).sum().item()
                val_loss   += outputs.loss.item() * n_tokens
                val_tokens += n_tokens
        avg_val_loss = val_loss / max(val_tokens, 1)

        elapsed = time.time() - t0
        print(f"Epoch {epoch}/{args.epochs} | "
              f"train_loss={avg_train_loss:.4f}  val_loss={avg_val_loss:.4f} | "
              f"{elapsed:.0f}s")

        log_records.append({
            "epoch": epoch, "train_loss": avg_train_loss,
            "val_loss": avg_val_loss, "elapsed_s": elapsed,
        })

        # 只保存 val loss 最优的 checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            ckpt_label = "完整模型" if args.full_ft else "LoRA adapter"
            print(f"  ✓ 最优{ckpt_label}已保存 → {ckpt_dir}  "
                  f"(val_loss={avg_val_loss:.4f})")

    # ── 保存训练日志 ──────────────────────────────────────────────────────────
    log_tag  = "full_ft" if args.full_ft else "sft"
    log_path = output_dir / f"train_log_{log_tag}.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_records, f, ensure_ascii=False, indent=2)

    ckpt_label = "完整模型" if args.full_ft else "LoRA adapter"
    print(f"\n训练完成。最优 val_loss={best_val_loss:.4f}")
    print(f"训练日志 → {log_path}")
    print(f"{ckpt_label} → {ckpt_dir}")
    print(f"\n下一步：运行 evaluate_sft.py 查看分类准确率与三方对比")


if __name__ == "__main__":
    main()
