"""
Few-Shot LLM 分类（每类给 N 个示例），与 zero-shot 和 SFT 对照

和 classify_llm.py 的区别只有一个：在 system prompt 之后插入若干组
「用户提问 + 标准答案」的示范，让模型照猫画虎。

教学重点：
  1. In-Context Learning：不改一个参数，只靠 prompt 里的示例就能提升效果
  2. 示例必须来自训练集——从验证集抽会把待评估样本的答案直接写进 prompt（数据泄漏）
  3. few-shot 的代价：prompt 变长，prefill 变慢；示例的选择也会影响结果

使用方式：
  python src_llm/classify_llm_fewshot.py                    # 每类 2 例，评估 200 条
  python src_llm/classify_llm_fewshot.py --n_shot 1         # 每类 1 例
  python src_llm/classify_llm_fewshot.py --n_shot 4         # 每类 4 例
  python src_llm/classify_llm_fewshot.py --demo             # 只跑 5 条，快速看 prompt

输出：
  终端：准确率 + 与 zero-shot 的对比
  文件：outputs/results/02_fewshot_qwen2-0.5b_result.json

注意：验证集采样与 classify_llm.py 完全一致（seed=42 + random.sample），
      所以两份结果的 200 条是同一批样本，可以直接对比。
"""

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import torch

# 复用 zero-shot 脚本里的模型加载、类别表、prompt 构造、结果解析
from classify_llm import (
    LABEL_NAMES, SYSTEM_PROMPT, ROOT, DATA_DIR, MODEL_PATH,
    load_model, build_prompt, parse_prediction,
)


def build_few_shot_examples(train_data, id2name, n_per_class, seed):
    """
    从训练集里「每类抽 n_per_class 条」作为示范。

    两个细节：
      1. 只用 train_data。用 val_data 会造成数据泄漏——被评估的标题
         可能正好出现在示例里，等于把答案告诉了模型。
      2. 抽完要 shuffle。否则 30 个示例是按类别顺序排列的
         （故事、故事、文化、文化、…），模型可能学到"位置"而不是"语义"。
    """
    rng = random.Random(seed)

    by_label = defaultdict(list)
    for item in train_data:
        by_label[item["label"]].append(item)

    examples = []
    for label_id in sorted(by_label.keys()):
        pool = by_label[label_id]
        k = min(n_per_class, len(pool))
        if k < n_per_class:
            print(f"  [警告] 类别「{id2name[label_id]}」只有 {len(pool)} 条，"
                  f"不足 {n_per_class} 例，按实际数量取")
        examples.extend(rng.sample(pool, k))

    rng.shuffle(examples)
    return examples


def build_messages(text, examples, id2name):
    """
    组装 few-shot 的对话。

    结构：system → (user, assistant) × N → 最后一个 user
    每条示范的 user 部分和真正的问题格式完全一致（都用 build_prompt），
    assistant 部分只放类别名——这样模型学到的是「只输出类别名」这个格式。
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex in examples:
        messages.append({"role": "user",      "content": build_prompt(ex["sentence"])})
        messages.append({"role": "assistant", "content": id2name[ex["label"]]})
    messages.append({"role": "user", "content": build_prompt(text)})
    return messages


def classify_one_fewshot(messages, model, tokenizer, device, max_new_tokens=8):
    """和 zero-shot 的 classify_one 逻辑相同，只是 messages 由外部构造。"""
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
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip(), prompt_len


def main():
    parser = argparse.ArgumentParser(description="LLM Few-Shot 分类")
    parser.add_argument("--model_path",    default=str(MODEL_PATH))
    parser.add_argument("--data_dir",      default=str(DATA_DIR))
    parser.add_argument("--num_samples",   default=200, type=int,
                        help="从验证集采样的样本数（与 zero-shot 保持一致才能对比）")
    parser.add_argument("--n_shot",        default=2,   type=int,
                        help="每个类别给几个示例")
    parser.add_argument("--seed",          default=42,  type=int,
                        help="验证集采样种子；必须和 zero-shot 一致才是同一批样本")
    parser.add_argument("--example_seed",  default=1234, type=int,
                        help="示例抽样种子，和 --seed 分开，互不干扰")
    parser.add_argument("--demo",          action="store_true",
                        help="只跑 5 条，并打印完整 prompt 供检查")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # ── 加载数据 ────────────────────────────────────────────────────────────
    with open(data_dir / "val.json",   encoding="utf-8") as f:
        val_data = json.load(f)
    with open(data_dir / "train.json", encoding="utf-8") as f:
        train_data = json.load(f)
    with open(data_dir / "label_map.json", encoding="utf-8") as f:
        label_map = json.load(f)
    id2name = {int(k): v for k, v in label_map["id2name"].items()}

    # ── 构造示例（只用训练集）──────────────────────────────────────────────
    examples = build_few_shot_examples(
        train_data, id2name, args.n_shot, args.example_seed
    )
    print(f"示例数: {len(examples)} 条（{args.n_shot} 例/类 × 15 类，全部取自训练集）")

    # ── 验证集采样：必须和 classify_llm.py 逐条一致 ─────────────────────────
    # random.seed(s); random.sample(x, k) 等价于 random.Random(s).sample(x, k)，
    # 所以用独立 Random 实例不会因为前面抽过示例而改变结果。
    n = 5 if args.demo else args.num_samples
    samples = random.Random(args.seed).sample(val_data, min(n, len(val_data)))
    print(f"评估样本数: {len(samples)}")

    # ── 加载模型 ────────────────────────────────────────────────────────────
    model, tokenizer = load_model(args.model_path, device)

    # ── 推理 ────────────────────────────────────────────────────────────────
    correct, total, unparseable = 0, 0, 0
    results = []
    t0 = time.time()
    prompt_len_first = 0

    for i, item in enumerate(samples):
        text      = item["sentence"]
        true_name = id2name[item["label"]]

        messages = build_messages(text, examples, id2name)
        if args.demo and i == 0:
            print("\n" + "=" * 60)
            print("第一条样本的完整 messages（中间示范省略）")
            print("=" * 60)
            shown = messages[:7] + [{"role": "...", "content": f"（省略 {len(messages)-8} 条）"}] + messages[-1:]
            for m in shown:
                print(f"  [{m['role']:9s}] {m['content']!r}")

        raw_output, plen = classify_one_fewshot(messages, model, tokenizer, device)
        if i == 0:
            prompt_len_first = plen

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
        print(f"[{i+1:3d}/{len(samples)}] {status} "
              f"真实:{true_name:4s} 预测:{str(pred_name):4s} | {text[:35]}")

    elapsed = time.time() - t0
    acc = correct / total if total > 0 else 0

    # ── 结果 ────────────────────────────────────────────────────────────────
    print(f"\n{'='*56}")
    print(f"Few-Shot LLM 分类结果（{args.n_shot} 例/类）")
    print(f"{'='*56}")
    print(f"  样本数   : {total}")
    print(f"  准确率   : {correct}/{total} = {acc:.4f}")
    print(f"  无法解析 : {unparseable} 条 ({unparseable/total*100:.1f}%)")
    print(f"  prompt长 : 首条 {prompt_len_first} tokens")
    print(f"  总耗时   : {elapsed:.1f}s, 均值 {elapsed/total:.2f}s/条")

    # ── 与 zero-shot 对比 ───────────────────────────────────────────────────
    zs_path = ROOT / "outputs" / "results" / "01_zeroshot_qwen2-0.5b_result.json"
    if zs_path.exists():
        with open(zs_path, encoding="utf-8") as f:
            zs = json.load(f)
        print(f"\n{'方法':<22}{'准确率':>10}{'无法解析':>12}")
        print("-" * 44)
        print(f"{'zero-shot':<22}{zs['accuracy']:>10.4f}"
              f"{zs['unparseable']:>10} 条")
        print(f"{f'few-shot ({args.n_shot}例/类)':<22}{acc:>10.4f}"
              f"{unparseable:>10} 条")
        print(f"{'差值':<22}{acc - zs['accuracy']:>+10.4f}")
        if zs["total"] != total:
            print(f"\n[注意] zero-shot 是 {zs['total']} 条，本次 {total} 条，样本规模不同")

    # ── 保存 ────────────────────────────────────────────────────────────────
    out_path = ROOT / "outputs" / "results" / "02_fewshot_qwen2-0.5b_result.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "accuracy": acc, "total": total, "correct": correct,
            "unparseable": unparseable, "n_shot": args.n_shot,
            "example_seed": args.example_seed, "seed": args.seed,
            "prompt_len": prompt_len_first, "results": results,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存 → {out_path}")


if __name__ == "__main__":
    main()
