"""
方法五：Qwen3-0.6B 指令微调（LoRA 与全量微调两种模式）

教学重点：
  1. SFT 的关键不是模型，而是三件事：**chat 模板、Loss Masking、数据格式**。
  2. Loss Masking：prompt 部分 labels 全设 -100（不算 loss），只在答案 token 上算；
     不 mask 的话模型学的是「复述模板」，不是「答标签」。
  3. 两种微调模式的分工：
       LoRA（默认）     只训低秩旁路，2.29M / 596M = 0.38%，显存约 3 GB，adapter 9 MB
       全量微调 --full_ft  596M 参数全更新，显存 6~8 GB（8G 卡要开梯度检查点），权重 1.2 GB
     实测在这个 15 类任务上全量与 LoRA 的差距很小（都约 0.48–0.49），
     但成本差一个量级 —— 这就是「什么时候不值得全量微调」的答案。
  4. Qwen3 必须 enable_thinking=False，且训练与推理的模板必须完全一致。

用法：
  python llm_lora/train_lora.py                                   # LoRA，5000 条 × 2 epoch
  python llm_lora/train_lora.py --num_train 1000 --epochs 1        # 课堂快速版
  python llm_lora/train_lora.py --full_ft --num_train 3000 --epochs 2   # 全量微调
  python llm_lora/train_lora.py --max_steps 20                     # 冒烟测流程
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.data import load_splits
from common.labels import LABELS
from common.model_paths import qwen3_path

CODE_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = CODE_ROOT / "outputs"

SYSTEM = ("你是一个新闻标题分类助手。请将给定的新闻标题分类到以下类别之一，"
          "只输出类别名称，不要输出任何其他内容。\n可选类别：" + "、".join(LABELS))


class SFTDataset(Dataset):
    """
    把分类样本转成 chat 格式的 SFT 样本。

    labels 的构造是本文件最核心的一行：
      prompt 部分 = -100（CrossEntropyLoss 会忽略 -100）
      答案部分   = 真实 token id（只有这里产生梯度）
    """

    def __init__(self, data, tok, max_length=128):
        self.data, self.tok, self.max_length = data, tok, max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        d = self.data[i]
        answer = LABELS[d["label"]]
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": d["sentence"]}]
        # tokenize=False 先拿文本：transformers 5.x 里 tokenize=True 返回 BatchEncoding，不好拼
        prompt = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        p_ids = self.tok(prompt, add_special_tokens=False)["input_ids"]
        a_ids = self.tok(answer, add_special_tokens=False)["input_ids"] + [self.tok.eos_token_id]
        input_ids = (p_ids + a_ids)[: self.max_length]
        labels = ([-100] * len(p_ids) + a_ids)[: self.max_length]
        return {"input_ids": torch.tensor(input_ids), "labels": torch.tensor(labels)}


def collate(batch, pad_id):
    """手写 collate：input_ids 用 pad 补齐，labels 用 -100 补齐（PAD 不产生 loss）"""
    L = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
    lab = torch.full((len(batch), L), -100, dtype=torch.long)
    att = torch.zeros((len(batch), L), dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b["input_ids"])
        ids[i, :n] = b["input_ids"]
        lab[i, :n] = b["labels"]
        att[i, :n] = 1
    return {"input_ids": ids, "labels": lab, "attention_mask": att}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_train", type=int, default=5000)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=4, help="全量微调建议 1~2")
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=None, help="不填则 LoRA=2e-4、全量=2e-5")
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--full_ft", action="store_true", help="全量微调；不加则 LoRA")
    ap.add_argument("--gradient_checkpointing", action="store_true", help="省显存（全量微调在 8G 卡上必开）")
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--max_steps", type=int, default=0, help=">0 时只跑这么多步（冒烟测试）")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 全量微调：用 bf16 训练（fp16 + Adam 容易溢出）；LoRA：fp16 足够
    dtype = torch.bfloat16 if (args.full_ft and device == "cuda") else (
        torch.float16 if device == "cuda" else torch.float32)
    tok = AutoTokenizer.from_pretrained(str(qwen3_path()))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(str(qwen3_path()), torch_dtype=dtype)

    if args.full_ft:
        model = base.to(device)                     # 所有参数默认 requires_grad=True
        if args.gradient_checkpointing:
            # 梯度检查点：用时间换显存，激活值不再常驻（全量微调在 8G 卡上基本必开）
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()
        lr = args.lr or 2e-5                        # 全量更新必须用更小的 lr，否则冲掉预训练知识
        print(f"[full-ft] 参数量 {sum(p.numel() for p in model.parameters())/1e6:.1f}M（全部可训练），lr={lr}")
    else:
        from peft import LoraConfig, TaskType, get_peft_model
        cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, r=args.lora_r, lora_alpha=args.lora_alpha,
                         lora_dropout=0.05, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
        model = get_peft_model(base, cfg).to(device)
        lr = args.lr or 2e-4
        # 一定要打印：target_modules 写错名字不会报错，只会「训了等于没训」
        model.print_trainable_parameters()

    train, val = load_splits()
    train = train[: args.num_train]
    val = val[: min(500, len(val))]
    train_dl = DataLoader(SFTDataset(train, tok, args.max_length), batch_size=args.batch_size,
                          shuffle=True, collate_fn=lambda b: collate(b, tok.pad_token_id))
    val_dl = DataLoader(SFTDataset(val, tok, args.max_length), batch_size=args.batch_size,
                        collate_fn=lambda b: collate(b, tok.pad_token_id))

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda" and dtype == torch.float16))
    log, step = [], 0
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running, n_batch = 0.0, 0
        for i, batch in enumerate(train_dl):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda" and dtype == torch.float16)):
                loss = model(**batch).loss / args.grad_accum     # 梯度累积：等效放大 batch
            scaler.scale(loss).backward()
            if (i + 1) % args.grad_accum == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                step += 1
            running += loss.item() * args.grad_accum
            n_batch += 1
            if i % 100 == 0:
                print(f"  epoch {epoch} {i}/{len(train_dl)} loss={loss.item()*args.grad_accum:.4f}", flush=True)
            if args.max_steps and step >= args.max_steps:
                break

        model.eval()
        vloss, vn = 0.0, 0
        with torch.no_grad():
            for batch in val_dl:
                batch = {k: v.to(device) for k, v in batch.items()}
                vloss += model(**batch).loss.item()
                vn += 1
        log.append({"epoch": epoch, "train_loss": round(running / max(1, n_batch), 4),
                    "val_loss": round(vloss / max(1, vn), 4), "steps": step,
                    "elapsed_s": round(time.time() - t_start, 1),
                    "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2) if device == "cuda" else None})
        print(f"epoch {epoch}: train_loss={log[-1]['train_loss']} val_loss={log[-1]['val_loss']}", flush=True)
        if args.max_steps:
            break

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.full_ft:
        # 全量微调存整个权重（约 1.2 GB）；交付件里不带，学生自己跑出来
        model.save_pretrained(str(OUT_DIR / "full_ft_ckpt"))
        log_path = OUT_DIR / "full_ft_train_log.json"
        print(f"全量权重 → {OUT_DIR / 'full_ft_ckpt'}")
    else:
        model.save_pretrained(str(OUT_DIR / "lora_adapter"))   # 只存 adapter，9 MB 左右
        log_path = OUT_DIR / "lora_train_log.json"
        print(f"adapter → {OUT_DIR / 'lora_adapter'}")
    log_path.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
