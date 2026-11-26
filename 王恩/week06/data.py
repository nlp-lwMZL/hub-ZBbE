"""复用 week03 的 12 语种数据，统一去重、标签和抽样口径。"""
import csv
import hashlib
import json
import random
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT.parent / "week03" / "v12"


def normalize(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def load_data(data_dir=DEFAULT_DATA, seed=42, train_per_class=0, eval_per_class=0):
    """按 train→dev→test 顺序去重；冲突标签直接报错，不看测试集选样本。"""
    splits, seen, removed = {}, {}, {}
    for split in ("train", "dev", "test"):
        rows, duplicates = [], 0
        with (Path(data_dir) / f"{split}.csv").open(encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if not {"text", "label"}.issubset(reader.fieldnames or []):
                raise ValueError(f"{split}.csv 必须包含 text,label 列")
            for index, row in enumerate(reader):
                text, label = row["text"].strip(), row["label"].strip()
                key = normalize(text)
                if not key or not label:
                    raise ValueError(f"{split}.csv 第 {index + 2} 行文本或标签为空")
                if key in seen:
                    if seen[key] != label:
                        raise ValueError(f"重复文本标签冲突：{text!r}")
                    duplicates += 1
                    continue
                seen[key] = label
                rows.append({"id": f"{split}:{index}", "text": text, "label": label})
        splits[split], removed[split] = rows, duplicates
    labels = sorted({r["label"] for r in splits["train"]})
    if len(labels) < 2:
        raise ValueError("训练集至少需要两类")
    for split, rows in splits.items():
        if {r["label"] for r in rows} != set(labels):
            raise ValueError(f"{split} 去重后的标签须与训练集一致且每类非空")
        limit = train_per_class if split == "train" else eval_per_class
        groups = defaultdict(list)
        for row in rows:
            groups[row["label"]].append(row)
        rng = random.Random(seed)
        selected = []
        for label in labels:
            group = groups[label][:]
            rng.shuffle(group)
            selected.extend(group[:limit] if limit else group)
        rng.shuffle(selected)
        splits[split] = selected
    # 指纹包含标签、顺序和文本：不同运行结果只有指纹一致才可横向比较。
    fingerprint = hashlib.sha256(json.dumps(splits, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    meta = {"source": str(Path(data_dir).resolve()), "seed": seed, "labels": labels,
            "removed_duplicates": removed, "fingerprint": fingerprint,
            "counts": {s: dict(Counter(r["label"] for r in rows)) for s, rows in splits.items()}}
    return splits, labels, meta
