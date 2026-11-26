"""
单条和批量推理脚本

教学重点：
  1. 推理时 model.eval() + torch.no_grad() 缺一不可（关闭 dropout 和梯度计算）
  2. softmax 后的概率分布比 argmax 更有信息量（可以看模型的置信度）
  3. 批量推理时注意显存管理（每次处理固定大小的 batch）

使用方式：
  # 单条推理
  python predict.py --text "苹果发布了最新的 iPhone 17 系列手机"

  # 批量推理（从 JSON 文件读取）
  python predict.py --input_file ../data/val.json --output_file ../outputs/val_predictions.json

  # 指定池化策略
  python predict.py --pool mean --text "今天股市大幅下跌"

依赖：
  pip install torch transformers
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import BertTokenizer

ROOT      = Path(__file__).parent.parent
DATA_DIR  = ROOT / "data"
BERT_PATH = ROOT.parent.parent / "pretrain_models" / "bert-base-chinese"
CKPT_DIR  = ROOT / "outputs" / "checkpoints"


def load_model_and_tokenizer(bert_path: str, ckpt_path: Path,
                              num_labels: int, device: torch.device):
    from model import build_model
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    pool  = ckpt.get("pool", "cls")
    model = build_model(bert_path, num_labels, pool=pool)
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device)
    model.eval()
    tokenizer = BertTokenizer.from_pretrained(bert_path)
    return model, tokenizer, pool


def predict_single(text: str, model, tokenizer, id2name: dict,
                   max_length: int, device: torch.device,
                   top_k: int = 3) -> dict:
    """单条文本推理，返回 top-k 预测结果及概率。"""
    encoding = tokenizer(
        text, max_length=max_length, truncation=True,
        padding="max_length", return_tensors="pt"
    )
    input_ids      = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    token_type_ids = encoding["token_type_ids"].to(device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask, token_type_ids)  # [1, C]
        probs  = F.softmax(logits, dim=-1).squeeze(0)              # [C]

    top_probs, top_ids = probs.topk(top_k)
    results = [
        {"label_id": int(lid), "label_name": id2name[int(lid)], "prob": float(p)}
        for lid, p in zip(top_ids, top_probs)
    ]
    return {"text": text, "prediction": results[0], "top_k": results}


def predict_batch(texts: list[str], model, tokenizer, id2name: dict,
                  max_length: int, batch_size: int, device: torch.device) -> list[dict]:
    """批量推理，返回每条的最优预测。"""
    all_results = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i: i + batch_size]
        encoding = tokenizer(
            batch_texts, max_length=max_length, truncation=True,
            padding="max_length", return_tensors="pt"
        )
        input_ids      = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)
        token_type_ids = encoding["token_type_ids"].to(device)

        with torch.no_grad():
            logits = model(input_ids, attention_mask, token_type_ids)
            probs  = F.softmax(logits, dim=-1)
            top_probs, top_ids = probs.topk(1, dim=-1)

        for text, lid, prob in zip(batch_texts, top_ids[:, 0], top_probs[:, 0]):
            all_results.append({
                "text":       text,
                "label_id":   int(lid),
                "label_name": id2name[int(lid)],
                "prob":       float(prob),
            })

    return all_results


def main():
    parser = argparse.ArgumentParser(description="BERT 文本分类推理")
    parser.add_argument("--pool",        default="cls", choices=["cls", "mean", "max"])
    parser.add_argument("--ckpt_path",   default=None, type=str)
    parser.add_argument("--bert_path",   default=str(BERT_PATH), type=str)
    parser.add_argument("--data_dir",    default=str(DATA_DIR), type=str)
    parser.add_argument("--max_length",  default=128, type=int)
    parser.add_argument("--batch_size",  default=64, type=int)
    parser.add_argument("--top_k",       default=3,  type=int)
    parser.add_argument("--text",        default=None, type=str, help="单条推理文本")
    parser.add_argument("--input_file",  default=None, type=str, help="批量推理输入 JSON")
    parser.add_argument("--output_file", default=None, type=str, help="批量推理结果输出路径")
    args = parser.parse_args()

    data_dir  = Path(args.data_dir)
    ckpt_path = Path(args.ckpt_path) if args.ckpt_path \
                else CKPT_DIR / f"best_{args.pool}.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(data_dir / "label_map.json", encoding="utf-8") as f:
        label_map = json.load(f)
    num_labels = label_map["num_labels"]
    id2name    = {int(k): v for k, v in label_map["id2name"].items()}

    model, tokenizer, pool = load_model_and_tokenizer(
        args.bert_path, ckpt_path, num_labels, device
    )
    print(f"模型加载完成，池化策略: {pool}")

    # ── 单条推理 ──────────────────────────────────────────────────────────────
    if args.text:
        result = predict_single(args.text, model, tokenizer, id2name,
                                args.max_length, device, args.top_k)
        print(f"\n文本：{result['text']}")
        print(f"预测：{result['prediction']['label_name']} "
              f"(置信度 {result['prediction']['prob']:.4f})")
        print(f"Top-{args.top_k}：")
        for r in result["top_k"]:
            print(f"  [{r['label_id']:2d}] {r['label_name']:4s}  {r['prob']:.4f}")
        return

    # ── 批量推理 ──────────────────────────────────────────────────────────────
    if args.input_file:
        with open(args.input_file, encoding="utf-8") as f:
            data = json.load(f)
        texts = [item["sentence"] for item in data]
        print(f"批量推理 {len(texts)} 条 ...")
        results = predict_batch(texts, model, tokenizer, id2name,
                                args.max_length, args.batch_size, device)

        # 如果有真实标签，计算准确率
        true_labels = [item["label"] for item in data]
        correct = sum(1 for r, t in zip(results, true_labels)
                      if r["label_id"] == t and t != -1)
        valid   = sum(1 for t in true_labels if t != -1)
        if valid > 0:
            print(f"准确率: {correct}/{valid} = {correct/valid:.4f}")

        if args.output_file:
            out_path = Path(args.output_file)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"结果已保存 → {out_path}")
        return

    # 未提供参数时给出示例
    print("请使用 --text 进行单条推理，或 --input_file 进行批量推理")
    print("\n示例（单条）：")
    examples = [
        "苹果发布了最新的 iPhone 17，搭载 A19 芯片",
        "今天 A 股市场全线下跌，沪指跌幅超过 2%",
        "梅西在比赛中打入一粒世界波，全场沸腾",
        "教育部出台新政策，要求减轻学生课业负担",
    ]
    for text in examples:
        result = predict_single(text, model, tokenizer, id2name,
                                args.max_length, device, top_k=3)
        top1 = result["prediction"]
        print(f"  [{top1['label_name']}] ({top1['prob']:.3f}) {text[:30]}")


if __name__ == "__main__":
    main()
