"""
Qwen3-0.6B 基金公告分类：LoRA / 全量微调（SFT）

数据
----
data/dataset/split/cls_{train,val,test}.jsonl（字段 text / cls_label）
标签来自 common/label.py 的 6 个公告类型。

关键点
------
1. Loss Masking：prompt 部分 labels 全设 -100（不算 loss），只在答案 token 上算。
   不 mask 的话模型学的是「复述模板」，不是「答标签」。
2. 截断时必须先保住答案：只截 prompt，不能把答案截掉，否则整条样本 loss 全为 -100，
   反向传播会得到 NaN。
3. Qwen3 必须 enable_thinking=False，且训练与推理的模板必须完全一致。
4. 两种模式：LoRA（默认，只训低秩旁路）/ 全量微调（--full_ft）。

用法
----
python lora/train_lora.py --epochs 2                        # LoRA
python lora/train_lora.py --full_ft --epochs 2              # 全量微调
python lora/train_lora.py --max_steps 3 --num_train 64      # 冒烟测试
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

CODE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_ROOT))

from common.label import CLS_ID2LABEL, CLS_NUM_LABELS  # noqa: E402

SPLIT_DIR = CODE_ROOT / "data" / "dataset" / "split"
MODEL_DIR = CODE_ROOT / "pretrain_models" / "Qwen3-0___6B"
OUT_DIR = CODE_ROOT / "output" / "lora"

LABELS = [CLS_ID2LABEL[i] for i in range(CLS_NUM_LABELS)]
SYSTEM = ("你是一个基金公告分类助手。请将给定的公告文本分类到以下类别之一，"
          "只输出类别名称，不要输出任何其他内容。\n可选类别：" + "、".join(LABELS))


def load_split(name: str, limit: int | None = None) -> list[dict]:
    """读取 data/dataset/split/cls_{name}.jsonl，返回 [{"text","cls_label"}]。"""
    path = SPLIT_DIR / f"cls_{name}.jsonl"
    rows = []
    for line in path.open(encoding="utf-8"):
        rec = json.loads(line)
        rows.append({"text": rec["text"], "cls_label": int(rec["cls_label"])})
        if limit and len(rows) >= limit:
            break
    return rows


def parse_label(text: str) -> int:
    """把模型输出解析成类别 id；匹配不到返回 -1（生成式模型可能不听话）。"""
    hits = [(len(name), idx) for idx, name in enumerate(LABELS) if name in text]
    return max(hits)[1] if hits else -1


class SFTDataset(Dataset):
    """把分类样本转成 chat 格式的 SFT 样本（prompt 段 labels=-100）。"""

    def __init__(self, data: list[dict], tok, max_length: int = 192):
        self.data, self.tok, self.max_length = data, tok, max_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, i: int) -> dict:
        d = self.data[i]
        answer = LABELS[d["cls_label"]]
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": d["text"]}]
        # tokenize=False 先拿文本，避免 BatchEncoding 拼接问题
        prompt = self.tok.apply_chat_template(msgs, tokenize=False,
                                              add_generation_prompt=True,
                                              enable_thinking=False)
        p_ids = self.tok(prompt, add_special_tokens=False)["input_ids"]
        a_ids = self.tok(answer, add_special_tokens=False)["input_ids"] + [self.tok.eos_token_id]
        # 只截 prompt，保证答案一定完整保留（否则整条样本 labels 全 -100 → NaN loss）
        p_ids = p_ids[: max(0, self.max_length - len(a_ids))]
        input_ids = p_ids + a_ids
        labels = [-100] * len(p_ids) + a_ids
        return {"input_ids": torch.tensor(input_ids), "labels": torch.tensor(labels)}


def collate(batch: list[dict], pad_id: int) -> dict:
    """手写 collate：input_ids 用 pad 补齐，labels 用 -100 补齐（PAD 不产生 loss）。"""
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_train", type=int, default=5000)
    ap.add_argument("--num_val", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=4, help="全量微调建议 1~2")
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=None, help="不填则 LoRA=2e-4、全量=2e-5")
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--full_ft", action="store_true", help="全量微调；不加则 LoRA")
    ap.add_argument("--gradient_checkpointing", action="store_true", help="省显存")
    ap.add_argument("--max_length", type=int, default=192,
                    help="含 chat 模板的总长度；本项目 prompt 实测均值 107、最大 182")
    ap.add_argument("--max_steps", type=int, default=0, help=">0 时只跑这么多优化步（冒烟测试）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_save", action="store_true", help="冒烟测试用，不落盘权重")
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 全量微调用 bf16（fp16 + Adam 易溢出）；LoRA 用 fp16 足够；CPU 一律 fp32
    dtype = torch.bfloat16 if (args.full_ft and device == "cuda") else (
        torch.float16 if device == "cuda" else torch.float32)
    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # transformers 4.55 仍用 torch_dtype（4.56+ 才改名 dtype）
    base = AutoModelForCausalLM.from_pretrained(str(MODEL_DIR), torch_dtype=dtype)
    n_layers = base.config.num_hidden_layers
    print(f"[cfg] device={device} dtype={dtype} layers={n_layers} "
          f"full_ft={args.full_ft} epochs={args.epochs} bs={args.batch_size} "
          f"accum={args.grad_accum} max_len={args.max_length}", flush=True)

    if args.full_ft:
        model = base.to(device)
        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()
        lr = args.lr or 2e-5
        print(f"[full-ft] 参数量 {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M（全部可训练），lr={lr}")
    else:
        from peft import LoraConfig, TaskType, get_peft_model
        cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, r=args.lora_r,
                         lora_alpha=args.lora_alpha, lora_dropout=0.05,
                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
        model = get_peft_model(base, cfg).to(device)
        lr = args.lr or 2e-4
        # 一定要打印：target_modules 写错名字不会报错，只会「训了等于没训」
        model.print_trainable_parameters()

    train = load_split("train", args.num_train)
    val = load_split("val", args.num_val)
    train_dl = DataLoader(SFTDataset(train, tok, args.max_length),
                          batch_size=args.batch_size, shuffle=True,
                          collate_fn=lambda b: collate(b, tok.pad_token_id))
    val_dl = DataLoader(SFTDataset(val, tok, args.max_length),
                        batch_size=args.batch_size,
                        collate_fn=lambda b: collate(b, tok.pad_token_id))
    print(f"[data] train={len(train)} val={len(val)} steps/epoch={len(train_dl)}", flush=True)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda" and dtype == torch.float16))
    log, step = [], 0
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running, n_batch = 0.0, 0
        for i, batch in enumerate(train_dl):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.float16,
                               enabled=(device == "cuda" and dtype == torch.float16)):
                loss = model(**batch).loss / args.grad_accum
            scaler.scale(loss).backward()
            if (i + 1) % args.grad_accum == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                step += 1
            running += loss.item() * args.grad_accum
            n_batch += 1
            if i % 5 == 0:
                print(f"  epoch {epoch} {i}/{len(train_dl)} loss={loss.item() * args.grad_accum:.4f}",
                      flush=True)
            if args.max_steps and step >= args.max_steps:
                break

        model.eval()
        vloss, vn = 0.0, 0
        with torch.no_grad():
            for batch in val_dl:
                batch = {k: v.to(device) for k, v in batch.items()}
                vloss += model(**batch).loss.item()
                vn += 1
        log.append({"epoch": epoch,
                    "train_loss": round(running / max(1, n_batch), 4),
                    "val_loss": round(vloss / max(1, vn), 4),
                    "steps": step,
                    "elapsed_s": round(time.time() - t_start, 1),
                    "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2)
                    if device == "cuda" else None})
        print(f"epoch {epoch}: train_loss={log[-1]['train_loss']} "
              f"val_loss={log[-1]['val_loss']} steps={step}", flush=True)
        if args.max_steps:
            break

    if args.no_save:
        print("[save] --no_save 已跳过权重保存")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.full_ft:
        model.save_pretrained(str(OUT_DIR / "full_ft_ckpt"))
        log_path = OUT_DIR / "full_ft_train_log.json"
        print(f"全量权重 → {OUT_DIR / 'full_ft_ckpt'}")
    else:
        model.save_pretrained(str(OUT_DIR / "lora_adapter"))
        log_path = OUT_DIR / "lora_train_log.json"
        print(f"adapter → {OUT_DIR / 'lora_adapter'}")
    log_path.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
