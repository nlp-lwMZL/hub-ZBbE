"""
方法三（推理）：加载 BERT checkpoint，在干净验证集上评测

教学重点：
  1. 评测必须在**去重后的同一批样本**上跑，否则和 TF-IDF / LLM 不可比。
  2. 顺手测延迟：单条 vs batch=32 差 3 倍多（实测 394ms → 121ms/条）——
     「模型多大」不是成本，「能不能批处理」才是。
  3. 保存逐条预测，方便和规则/TF-IDF/LLM 做三方对照与错误分析。

用法：
  python bert/predict_bert.py                       # 默认 outputs/bert_ckpt.pt
  python bert/predict_bert.py --ckpt path/to/ckpt.pt --n 500
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.data import load_clean_val
from common.labels import LABELS
from common.metrics import classification_report, expected_calibration_error, to_markdown_table
from common.model_paths import bert_path
from train_bert import BertClassifier, TnewsDataset

CODE_ROOT = Path(__file__).resolve().parent.parent


@torch.no_grad()
def predict(model, tok, texts, device, max_length=64, batch_size=32):
    """返回 (preds, probs, 每样本毫秒数)"""
    preds, probs = [], []
    t0 = time.time()
    for i in range(0, len(texts), batch_size):
        enc = tok(texts[i:i + batch_size], truncation=True, max_length=max_length,
                  padding="max_length", return_tensors="pt")
        logits = model(enc["input_ids"].to(device), enc["attention_mask"].to(device))
        p = torch.softmax(logits.float(), dim=-1)
        preds.extend(p.argmax(-1).cpu().tolist())
        probs.extend(p.cpu().tolist())
    ms = (time.time() - t0) * 1000 / max(1, len(texts))
    return preds, probs, round(ms, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(CODE_ROOT / "outputs" / "bert_ckpt.pt"))
    ap.add_argument("--n", type=int, default=0, help="抽样条数，0=全部干净样本")
    ap.add_argument("--out", default="outputs/bert_pred.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)   # torch 2.6 默认 weights_only=True，会拒载字典
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(bert_path()))
    model = BertClassifier(bert_path(), num_labels=len(LABELS), pool=ckpt.get("pool", "cls"))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()

    samples, meta = load_clean_val(args.n or None)
    texts = [d["sentence"] for d in samples]
    preds, probs, ms = predict(model, tok, texts, device)

    y_true = [d["label"] for d in samples]
    report = classification_report(y_true, preds)
    report.update({
        "method": "bert_finetune",
        "pool": ckpt.get("pool"),
        "ckpt_val_f1": ckpt.get("val_macro_f1"),
        "ms_per_sample": ms,
        "ece": expected_calibration_error(y_true, probs),
    })

    out = CODE_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"meta": meta, "report": report,
                               "preds": [{"text": t, "true": y, "pred": p}
                                         for t, y, p in zip(texts, y_true, preds)]},
                              ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"评测 {meta['n_eval']} 条（干净集）| {ms} ms/条 | ECE={report['ece']}")
    print(to_markdown_table({"BERT 微调": report}))
    print(f"→ 结果写入 {out}")


if __name__ == "__main__":
    main()
