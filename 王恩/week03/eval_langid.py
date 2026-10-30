# -*- coding: utf-8 -*-
"""加载训练产物进行独立评测或单句预测。"""
import argparse

import torch
from torch.utils.data import DataLoader

from train_langid_cls import (
    DATA_DIR, MODEL_PATH, TextDataset, classification_report, encode, load_model,
    load_split, normalize, positive_int, predict, print_report, validate_labels,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--split", choices=["test", "dev"], default="test")
    parser.add_argument("--text", help="指定时仅预测该文本")
    parser.add_argument("--threads", type=positive_int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    model, checkpoint = load_model(args.model)
    vocab, labels, maxlen = checkpoint["vocab"], checkpoint["labels"], checkpoint["maxlen"]
    if args.text is not None:
        if not normalize(args.text):
            parser.error("文本必须至少含一个字母或组合标记")
        x = torch.tensor([encode(args.text, vocab, maxlen)], dtype=torch.long)
        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1)[0]
        for index in probs.argsort(descending=True)[:3].tolist():
            print(f"{labels[index]}: {probs[index].item():.2%}")
    else:
        rows = load_split(args.data_dir, args.split)
        validate_labels(rows, labels, args.split)
        loader = DataLoader(TextDataset(rows, vocab, {l: i for i, l in enumerate(labels)}, maxlen),
                            batch_size=128)
        preds, golds = predict(model, loader)
        print(f"{args.split}: {len(rows)} 条")
        print_report(classification_report(preds, golds, labels), labels)


if __name__ == "__main__":
    main()
