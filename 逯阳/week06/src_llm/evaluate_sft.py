"""
加载 SFT LoRA adapter，在验证集上评估分类准确率，与 zero-shot 和 BERT 三方对比

教学重点：
  1. LoRA adapter 加载：base model + adapter → merge_and_unload() 合并权重加速推理
  2. 生成式分类：generate() → decode → 模糊匹配类别名（与 zero-shot 推理代码完全一致）
  3. 三方对比：BERT fine-tune（判别式）/ LLM zero-shot / LLM SFT（生成式）

使用方式：
  python evaluate_sft.py                              # 评估 LoRA 模型（默认）
  python evaluate_sft.py --ckpt_dir ../outputs/sft_full_ckpt  # 评估全量微调模型
  python evaluate_sft.py --num_samples 500            # 采样 500 条
  python evaluate_sft.py --demo                       # 只跑 5 条示例（快速演示）

  脚本自动识别 checkpoint 类型：
    outputs/sft_adapter/    含 adapter_config.json → LoRA 模式
    outputs/sft_full_ckpt/  含 config.json（无 adapter_config.json）→ 全量微调模式

依赖：
  pip install torch transformers peft
"""

import os
import argparse
import json
import random
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT        = Path(__file__).parent.parent
DATA_DIR    = ROOT / "data"
MODEL_PATH  = ROOT.parent.parent / "pretrain_models" / "Qwen2-0.5B-Instruct"
ADAPTER_DIR = ROOT / "outputs" / "sft_adapter"
OUTPUT_DIR  = ROOT / "outputs"
# 评测结果统一放这里，文件名自带方法和模型名
RESULTS_DIR = OUTPUT_DIR / "results"

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


def load_model(model_path: str, ckpt_dir: str, device: torch.device):
    ckpt_path = Path(ckpt_dir)
    is_lora   = (ckpt_path / "adapter_config.json").exists()

    if is_lora:
        # ── LoRA adapter：加载 base model，再套 PeftModel，合并权重 ──────────
        if not PEFT_AVAILABLE:
            raise ImportError("加载 LoRA adapter 需要 peft 库：pip install peft>=0.14.0")
        print(f"检测到 LoRA adapter，加载 base model: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(
            str(Path(model_path).resolve()), trust_remote_code=True
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            str(Path(model_path).resolve()),
            dtype=torch.float16 if device.type == "cuda" else torch.float32,
            device_map="auto" if device.type == "cuda" else None,
            trust_remote_code=True,
        )
        print(f"加载 LoRA adapter: {ckpt_dir}")
        model = PeftModel.from_pretrained(base_model, str(ckpt_path))
        # merge_and_unload()：把 B·A 合并进 W，推理速度与原始模型相同
        model = model.merge_and_unload()
    else:
        # ── 全量微调 checkpoint：直接 from_pretrained 加载完整模型 ────────────
        print(f"检测到全量微调 checkpoint，直接加载: {ckpt_dir}")
        tokenizer = AutoTokenizer.from_pretrained(
            str(ckpt_path), trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            str(ckpt_path),
            dtype=torch.float16 if device.type == "cuda" else torch.float32,
            device_map="auto" if device.type == "cuda" else None,
            trust_remote_code=True,
        )

    if device.type != "cuda":
        model = model.to(device)
    model.eval()
    ckpt_type = "LoRA adapter 已合并" if is_lora else "全量微调模型"
    print(f"模型加载完成（{ckpt_type}）\n")
    return model, tokenizer


def classify_one(text: str, model, tokenizer, device: torch.device,
                 max_new_tokens: int = 8) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"新闻标题：{text}\n类别："},
    ]
    encoding = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_tensors="pt", return_dict=True,
    )
    input_ids      = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    prompt_len     = input_ids.shape[-1]

    with torch.no_grad():
        output_ids = model.generate(
            input_ids, attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output_ids[0][prompt_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def parse_prediction(raw_output: str) -> str | None:
    for name in LABEL_NAMES:
        if name in raw_output:
            return name
    return None


def main():
    parser = argparse.ArgumentParser(description="LLM SFT 分类评估")
    parser.add_argument("--model_path",  default=str(MODEL_PATH))
    parser.add_argument("--ckpt_dir",    default=str(ADAPTER_DIR),
                        help="checkpoint 目录；含 adapter_config.json 则走 LoRA，否则走全量微调")
    parser.add_argument("--data_dir",    default=str(DATA_DIR))
    parser.add_argument("--num_samples", default=200, type=int,
                        help="从验证集随机采样的样本数")
    parser.add_argument("--seed",        default=42,  type=int)
    parser.add_argument("--demo",        action="store_true",
                        help="只跑 5 条示例（快速演示）")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 检查 checkpoint 是否存在
    ckpt_dir = Path(args.ckpt_dir)
    if not ckpt_dir.exists():
        print(f"[错误] checkpoint 目录不存在：{ckpt_dir}")
        print("请先运行 train_sft.py 完成训练。")
        print("  LoRA 默认保存到:    outputs/sft_adapter/")
        print("  全量微调保存到:     outputs/sft_full_ckpt/")
        return

    # ── 加载数据 ──────────────────────────────────────────────────────────────
    with open(Path(args.data_dir) / "val.json", encoding="utf-8") as f:
        val_data = json.load(f)
    with open(Path(args.data_dir) / "label_map.json", encoding="utf-8") as f:
        label_map = json.load(f)
    id2name = {int(k): v for k, v in label_map["id2name"].items()}

    random.seed(args.seed)
    n = 5 if args.demo else args.num_samples
    samples = random.sample(val_data, min(n, len(val_data)))
    print(f"评估样本数: {len(samples)}\n")

    # ── 加载模型（自动识别 LoRA / 全量微调）─────────────────────────────────────
    model, tokenizer = load_model(args.model_path, str(ckpt_dir), device)

    # ── 推理 ──────────────────────────────────────────────────────────────────
    correct, total, unparseable = 0, 0, 0
    results = []
    t0 = time.time()

    for i, item in enumerate(samples):
        text      = item["sentence"]
        true_id   = item["label"]
        true_name = id2name[true_id]

        raw_output = classify_one(text, model, tokenizer, device)
        pred_name  = parse_prediction(raw_output)

        is_correct = (pred_name == true_name)
        if pred_name is None:
            unparseable += 1
        if is_correct:
            correct += 1
        total += 1

        results.append({
            "text": text, "true_label": true_name,
            "pred_label": pred_name, "raw_output": raw_output,
            "correct": is_correct,
        })

        status = "✓" if is_correct else ("?" if pred_name is None else "✗")
        print(f"[{i+1:3d}/{len(samples)}] {status}  "
              f"真实:{true_name:4s}  预测:{str(pred_name):4s}  |  {text[:35]}")

    elapsed = time.time() - t0
    acc = correct / total if total > 0 else 0

    # ── 读取已有 zero-shot 结果，做三方对比 ───────────────────────────────────
    zero_shot_acc_str = "（未运行 classify_llm.py）"
    zero_shot_path = RESULTS_DIR / "01_zeroshot_qwen2-0.5b_result.json"
    if zero_shot_path.exists():
        with open(zero_shot_path, encoding="utf-8") as f:
            zs = json.load(f)
        zero_shot_acc_str = f"{zs['accuracy']:.4f}（{zs['total']} 条）"

    print(f"\n{'='*60}")
    print(f"LLM SFT 分类结果")
    print(f"{'='*60}")
    print(f"  样本数    : {total}")
    print(f"  准确率    : {correct}/{total} = {acc:.4f}")
    print(f"  无法解析  : {unparseable} 条 ({unparseable/total*100:.1f}%)")
    print(f"  总耗时    : {elapsed:.1f}s，均值 {elapsed/total:.2f}s/条")

    print(f"""
三方对比（val 集随机采样，seed=42）
  ┌──────────────────────────────────────────┬──────────┐
  │ 方法                                     │ 准确率   │
  ├──────────────────────────────────────────┼──────────┤
  │ BERT fine-tune（全部 53K 条，3 epochs）   │ ~0.57~62 │
  │ Qwen2-0.5B zero-shot                     │ {zero_shot_acc_str:<8} │
  │ Qwen2-0.5B SFT（LoRA，{total} 条样本）    │ {acc:.4f}   │
  └──────────────────────────────────────────┴──────────┘

思考题：
  1. SFT 相比 zero-shot 提升了多少？这符合你的预期吗？
  2. BERT 用了全部 53K 条，SFT 只用了 5K 条；如果数据量相同，谁更有优势？
  3. LoRA 参数量仅约 0.5%，效果损失有多大？
     对比实验：train_sft.py --lora_r 32，或换回全量微调（去掉 peft）。
  4. 生成式分类有 "无法解析" 的情况，判别式分类（BERT）没有。
     在生产系统中，这个差异如何处理？
""")

    # ── 保存结果 ──────────────────────────────────────────────────────────────
    out_path = RESULTS_DIR / "03_sft_lora_qwen2-0.5b_result.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "accuracy": acc, "total": total, "correct": correct,
            "unparseable": unparseable, "results": results,
        }, f, ensure_ascii=False, indent=2)
    print(f"结果已保存 → {out_path}")


if __name__ == "__main__":
    main()
