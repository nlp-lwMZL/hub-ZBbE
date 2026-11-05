# -*- coding: utf-8 -*-
"""12 语种识别训练，参考 courseware/week03/language_recognize。

字符级 Embedding → RNN → 掩码 MaxPool → LayerNorm → Dropout → Linear。
标签和字符表仅从训练集生成，验证集选最优模型，测试集用于最终评测。
运行：python train_langid_cls.py
"""

import argparse
import copy
import csv
import json
import random
import time
import unicodedata
from collections import Counter
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "v12"
MODEL_PATH = HERE / "langid_rnn_v12.pt"
MAX_LEN = 40
EMBED_DIM = 64
HIDDEN_DIM = 64
DROPOUT = 0.3
SEED = 42


def normalize(text):
    """与课堂一致，保留字母及组合标记（印地语、泰语等需要）。"""
    text = unicodedata.normalize("NFKC", text).lower()
    return "".join(c for c in text if unicodedata.category(c)[0] in ("L", "M"))


def build_char_vocab(sentences, min_freq=1):
    freq = Counter(c for text in sentences for c in normalize(text))
    chars = sorted(freq, key=lambda c: (-freq[c], c))
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for char in chars:
        if freq[char] >= min_freq:
            vocab[char] = len(vocab)
    return vocab


def encode(text, vocab, maxlen=MAX_LEN):
    ids = [vocab.get(c, 1) for c in normalize(text)[:maxlen]]
    return ids + [0] * (maxlen - len(ids))


def load_split(data_dir, name):
    path = Path(data_dir) / f"{name}.csv"
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"text", "label"}.issubset(reader.fieldnames or []):
            raise ValueError(f"{path} 必须包含 text、label 列")
        rows = []
        for line, row in enumerate(reader, 2):
            text, label = row.get("text"), row.get("label")
            if not text or not normalize(text) or not label or not label.strip():
                raise ValueError(f"{path}:{line} 文本或标签为空（或文本不含有效字符）")
            rows.append((text, label.strip()))
    if not rows:
        raise ValueError(f"{path} 没有样本")
    return rows


def validate_labels(rows, labels, split):
    unknown = sorted({label for _, label in rows} - set(labels))
    if unknown:
        raise ValueError(f"{split} 包含训练集未见标签：{', '.join(unknown)}")


class TextDataset(Dataset):
    def __init__(self, rows, vocab, label2id, maxlen=MAX_LEN):
        self.x = torch.tensor([encode(t, vocab, maxlen) for t, _ in rows], dtype=torch.long)
        self.y = torch.tensor([label2id[label] for _, label in rows], dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


class LangIDClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes, embed_dim=EMBED_DIM,
                 hidden_dim=HIDDEN_DIM, dropout=DROPOUT):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        mask = (x != 0).unsqueeze(-1)
        hidden, _ = self.rnn(self.embedding(x))
        hidden = hidden.masked_fill(~mask, -1e9)
        pooled = hidden.max(dim=1).values
        # 纯标点等输入全为 PAD 时，使用零向量避免池化异常。
        pooled = pooled.masked_fill(~mask.any(dim=1), 0.0)
        return self.fc(self.dropout(self.norm(pooled)))


def predict(model, loader):
    model.eval()
    preds, golds = [], []
    with torch.no_grad():
        for x, y in loader:
            preds.extend(model(x).argmax(dim=1).tolist())
            golds.extend(y.tolist())
    return preds, golds


def classification_report(preds, golds, labels):
    matrix = [[0] * len(labels) for _ in labels]
    for pred, gold in zip(preds, golds):
        matrix[gold][pred] += 1
    per_class = {}
    for i, label in enumerate(labels):
        support = sum(matrix[i])
        predicted = sum(row[i] for row in matrix)
        precision = matrix[i][i] / predicted if predicted else 0.0
        recall = matrix[i][i] / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = dict(support=support, precision=precision, recall=recall, f1=f1)
    return {
        "accuracy": sum(p == g for p, g in zip(preds, golds)) / len(golds),
        "macro_f1": sum(item["f1"] for item in per_class.values()) / len(labels),
        "per_class": per_class,
        "confusion_matrix": matrix,  # 行为真实标签，列为预测标签，顺序见 labels。
    }


def print_report(report, labels):
    print(f"accuracy={report['accuracy']:.4f}  macro-F1={report['macro_f1']:.4f}")
    print("语种    样本数  precision  recall     F1")
    for label, item in report["per_class"].items():
        print(f"{label:<6}{item['support']:>6}  {item['precision']:.4f}     "
              f"{item['recall']:.4f}  {item['f1']:.4f}")
    matrix = report["confusion_matrix"]
    errors = sorted(((matrix[g][p], labels[g], labels[p])
                     for g in range(len(labels)) for p in range(len(labels))
                     if g != p and matrix[g][p]), reverse=True)
    print("最常混淆（真实 → 预测）：")
    for count, gold, pred in errors[:5]:
        print(f"  {gold} → {pred}: {count} 条")


def load_model(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = LangIDClassifier(len(checkpoint["vocab"]), len(checkpoint["labels"]),
                             checkpoint["embed_dim"], checkpoint["hidden_dim"],
                             checkpoint["dropout"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("必须为正整数")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--save", type=Path, default=MODEL_PATH)
    parser.add_argument("--epochs", type=positive_int, default=8)
    parser.add_argument("--batch-size", type=positive_int, default=64)
    parser.add_argument("--max-len", type=positive_int, default=MAX_LEN)
    parser.add_argument("--threads", type=positive_int, default=1)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    train_rows = load_split(args.data_dir, "train")
    dev_rows = load_split(args.data_dir, "dev")
    labels = sorted({label for _, label in train_rows})
    validate_labels(dev_rows, labels, "dev")
    label2id = {label: index for index, label in enumerate(labels)}
    # 不复用 v8 字符表，也不把 dev/test 字符泄漏到词表。
    vocab = build_char_vocab(text for text, _ in train_rows)
    train_loader = DataLoader(TextDataset(train_rows, vocab, label2id, args.max_len),
                              batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(TextDataset(dev_rows, vocab, label2id, args.max_len),
                            batch_size=128)
    model = LangIDClassifier(len(vocab), len(labels))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    print(f"训练 {len(train_rows)} 条 / 验证 {len(dev_rows)} 条")
    print(f"类别 {len(labels)} 个：{' '.join(labels)}")
    print(f"字符表 {len(vocab)}，参数量 {sum(p.numel() for p in model.parameters()):,}")

    start = time.perf_counter()
    history, best_state = [], None
    best_accuracy, best_epoch = -1.0, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)
        preds, golds = predict(model, dev_loader)
        dev_report = classification_report(preds, golds, labels)
        history.append(dict(epoch=epoch, train_loss=total_loss / len(train_rows),
                            dev_accuracy=dev_report["accuracy"], dev_macro_f1=dev_report["macro_f1"]))
        print(f"Epoch {epoch:2d}/{args.epochs}  loss={history[-1]['train_loss']:.4f}  "
              f"dev_acc={dev_report['accuracy']:.4f}", flush=True)
        if dev_report["accuracy"] > best_accuracy:
            best_accuracy, best_epoch = dev_report["accuracy"], epoch
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    test_rows = load_split(args.data_dir, "test")
    validate_labels(test_rows, labels, "test")
    test_loader = DataLoader(TextDataset(test_rows, vocab, label2id, args.max_len), batch_size=128)
    preds, golds = predict(model, test_loader)
    report = classification_report(preds, golds, labels)
    elapsed = time.perf_counter() - start
    print(f"\n使用验证集最优模型（epoch={best_epoch}，dev_acc={best_accuracy:.4f}）评测测试集：")
    print_report(report, labels)

    args.save.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state_dict=best_state, vocab=vocab, labels=labels, maxlen=args.max_len,
                    embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, dropout=DROPOUT,
                    seed=args.seed, best_epoch=best_epoch, dev_accuracy=best_accuracy), args.save)
    log_path = args.save.with_suffix(".json")
    log = dict(labels=labels, vocab_size=len(vocab), seed=args.seed,
               config=dict(epochs=args.epochs, batch_size=args.batch_size, maxlen=args.max_len,
                           learning_rate=1e-3, threads=args.threads, embed_dim=EMBED_DIM,
                           hidden_dim=HIDDEN_DIM, dropout=DROPOUT),
               samples=dict(train=len(train_rows), dev=len(dev_rows), test=len(test_rows)),
               best_epoch=best_epoch, best_dev_accuracy=best_accuracy, elapsed_seconds=elapsed,
               history=history, test=report)
    log_path.write_text(json.dumps(log, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n耗时 {elapsed:.1f} 秒；模型：{args.save}；训练记录：{log_path}")


if __name__ == "__main__":
    main()
