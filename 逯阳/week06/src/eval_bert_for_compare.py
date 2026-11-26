"""
用 BERT 微调后的 5 个模型，跑和 LLM 评测完全相同的那 200 条验证样本

5 个模型 = 3 种池化 × 2 种 loss（cls/mean 各有加权版，max 只有不加权版）：
  best_cls.pt            cls  不加权
  best_cls_weighted.pt   cls  加权
  best_max.pt            max  不加权
  best_mean.pt           mean 不加权
  best_mean_weighted.pt  mean 加权

为什么要单独写这个脚本：
  src/evaluate.py 跑的是整个验证集（10000 条），和 LLM 那边抽的 200 条对不上，
  没法逐条配对比较。这里用和 classify_llm.py 一模一样的抽样方式
  （random.seed(42) + random.sample），保证所有结果都是同一批样本、同一顺序。

教学重点：
  1. 判别式模型不需要「解析输出」—— argmax 出来的一定是 15 个类别之一，
     所以 unparseable 恒为 0。这是 BERT 相对生成式分类的一个结构性优势。
  2. 池化策略和加权 loss 的影响都很小（全量 val 上准确率差不到 0.005），
     200 条样本上更是分不出胜负 —— 只有做配对检验才知道是不是噪声。
  3. BERT 会「自信地答错」：错的时候 softmax 概率可能依然很高，
     所以置信度不能当正确率用，要看 logits 分布或做校准。

使用方式：
  python src/eval_bert_for_compare.py                 # 跑全部 5 个模型
  python src/eval_bert_for_compare.py --only cls      # 只跑 pool=cls 的（含加权版）
  python src/eval_bert_for_compare.py --only best_max.pt

输出（outputs/results/ 下，编号 04~08）：
  04_bert_cls_unweighted_result.json
  05_bert_cls_weighted_result.json
  06_bert_max_unweighted_result.json
  07_bert_mean_unweighted_result.json
  08_bert_mean_weighted_result.json
"""

import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import BertTokenizer

ROOT        = Path(__file__).parent.parent
DATA_DIR    = ROOT / "data"
BERT_PATH   = ROOT.parent.parent / "pretrain_models" / "bert-base-chinese"
CKPT_DIR    = ROOT / "outputs" / "checkpoints"
RESULTS_DIR = ROOT / "outputs" / "results"

# 和 classify_llm.py 保持一致，这样两边抽出来的是同一批样本
SEED = 42

# 编号决定文件名前缀，顺序固定，方便和 LLM 那几份结果一起看
BERT_MODELS = [
    {"idx": 4, "ckpt": "best_cls.pt"},
    {"idx": 5, "ckpt": "best_cls_weighted.pt"},
    {"idx": 6, "ckpt": "best_max.pt"},
    {"idx": 7, "ckpt": "best_mean.pt"},
    {"idx": 8, "ckpt": "best_mean_weighted.pt"},
]


def out_name(idx, pool, weighted):
    """结果文件名：编号 + 模型 + 池化 + 是否加权，见名知义。"""
    return f"{idx:02d}_bert_{pool}_{'weighted' if weighted else 'unweighted'}_result.json"


def eval_one(ckpt_name, texts, true_names, tokenizer, id2name, args, device):
    """加载一个 checkpoint，跑一遍 texts，返回结果 dict。"""
    from model import build_model

    ckpt_path = CKPT_DIR / ckpt_name
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    pool = ckpt["pool"]

    model = build_model(args.bert_path, len(id2name), pool=pool)
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device).eval()

    t0 = time.time()
    pred_ids, pred_probs = [], []
    with torch.no_grad():
        for i in range(0, len(texts), args.batch_size):
            enc = tokenizer(texts[i: i + args.batch_size],
                            max_length=args.max_length, truncation=True,
                            padding="max_length", return_tensors="pt")
            logits = model(enc["input_ids"].to(device),
                           enc["attention_mask"].to(device),
                           enc["token_type_ids"].to(device))
            probs = F.softmax(logits, dim=-1)
            top_probs, top_ids = probs.max(dim=-1)
            pred_ids.extend(top_ids.cpu().tolist())
            pred_probs.extend(top_probs.cpu().tolist())
    elapsed = time.time() - t0

    # 立刻释放，免得 5 个模型轮着占显存
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    results, correct = [], 0
    for text, true_name, lid, prob in zip(texts, true_names, pred_ids, pred_probs):
        pred_name = id2name[lid]
        is_correct = pred_name == true_name
        correct += is_correct
        results.append({
            "text":       text,
            "true_label": true_name,
            "pred_label": pred_name,
            "raw_output": pred_name,   # BERT 没有「原始输出」，和 pred_label 相同
            "prob":       round(prob, 6),
            "correct":    is_correct,
        })

    return {
        "method": "bert-finetune",
        "model": "bert-base-chinese",
        "pool": pool,
        "use_class_weight": ckpt["use_class_weight"],
        "ckpt": ckpt_name,
        "ckpt_epoch": ckpt["epoch"],
        "ckpt_val_acc_full": ckpt["val_acc"],                    # 完整 10000 条 val
        "ckpt_val_macro_f1_full": float(ckpt["val_macro_f1"]),   # 完整 10000 条 val
        "max_length": args.max_length,
        "seed": args.seed,
        "accuracy": correct / len(results),
        "total": len(results),
        "correct": correct,
        "unparseable": 0,
        "avg_confidence": round(sum(pred_probs) / len(pred_probs), 6),
        "elapsed_sec": round(elapsed, 2),
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description="用 BERT 的 5 个微调模型跑 LLM 评测的那批样本")
    parser.add_argument("--only",       default=None,
                        help="只跑某类：cls / mean / max，或直接写 ckpt 文件名")
    parser.add_argument("--num_samples", default=200, type=int)
    parser.add_argument("--seed",        default=SEED, type=int)
    parser.add_argument("--max_length",  default=64, type=int,
                        help="默认 64，和训练时的 args 一致；新闻标题都很短，64 足够，不会截断")
    parser.add_argument("--batch_size",  default=64, type=int)
    parser.add_argument("--bert_path",   default=str(BERT_PATH), type=str)
    parser.add_argument("--data_dir",    default=str(DATA_DIR), type=str)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── 挑要跑哪几个 ────────────────────────────────────────────────────────
    todo = BERT_MODELS
    if args.only:
        key = args.only.strip()
        if key.endswith(".pt"):
            todo = [m for m in BERT_MODELS if m["ckpt"] == key]
        else:
            todo = [m for m in BERT_MODELS if m["ckpt"].startswith(f"best_{key}")]
        if not todo:
            print(f"[错误] --only {key} 没匹配到任何 checkpoint")
            return

    missing = [m["ckpt"] for m in todo if not (CKPT_DIR / m["ckpt"]).exists()]
    if missing:
        for name in missing:
            print(f"[缺少] {CKPT_DIR / name}")
        print("       请先训练：python src/train.py --pool {pool} [--use_class_weight]")
        return

    # ── 抽样：必须和 classify_llm.py 完全一致 ────────────────────────────────
    with open(data_dir / "val.json", encoding="utf-8") as f:
        val_data = json.load(f)
    random.seed(args.seed)
    samples = random.sample(val_data, min(args.num_samples, len(val_data)))
    print(f"评估样本数: {len(samples)}（seed={args.seed}，与 LLM 评测同一批）")

    with open(data_dir / "label_map.json", encoding="utf-8") as f:
        label_map = json.load(f)
    id2name = {int(k): v for k, v in label_map["id2name"].items()}
    texts = [s["sentence"] for s in samples]
    true_names = [id2name[s["label"]] for s in samples]

    tokenizer = BertTokenizer.from_pretrained(args.bert_path)

    # ── 逐个跑 ──────────────────────────────────────────────────────────────
    print(f"\n{'模型':<24}{'池化':>6}{'加权':>6}{'准确率':>10}{'全量val':>10}{'ms/条':>8}")
    print("-" * 64)
    summaries = []
    for m in todo:
        data = eval_one(m["ckpt"], texts, true_names, tokenizer, id2name, args, device)
        out_path = RESULTS_DIR / out_name(m["idx"], data["pool"], data["use_class_weight"])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        ms = data["elapsed_sec"] / data["total"] * 1000
        print(f"{m['ckpt']:<24}{data['pool']:>6}"
              f"{'是' if data['use_class_weight'] else '否':>6}"
              f"{data['accuracy']:>10.4f}{data['ckpt_val_acc_full']:>10.4f}{ms:>8.1f}")
        summaries.append((out_path, data))

    # ── 和 LLM 结果对一下，确认是同一批样本 ──────────────────────────────────
    ref_path = RESULTS_DIR / "01_zeroshot_qwen2-0.5b_result.json"
    if ref_path.exists():
        with open(ref_path, encoding="utf-8") as f:
            ref_texts = [r["text"] for r in json.load(f)["results"]]
        if ref_texts == texts:
            print(f"\n样本校验通过：与 {ref_path.name} 逐条一致 ✓（可以做配对检验）")
        else:
            print(f"\n[警告] 样本与 {ref_path.name} 不一致，不能做配对检验")

    print(f"\n共保存 {len(summaries)} 份结果到 {RESULTS_DIR}")
    for p, _ in summaries:
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
