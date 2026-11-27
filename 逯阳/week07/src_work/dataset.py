"""
人民日报 NER 数据集（CoNLL 格式）→ BERT 输入

与 ../src/dataset.py（cluener2020）的区别：
  1. 数据已经是**逐字 tokens + BIO 标签**，不需要 span→BIO 转换
     cluener: {"text": "浙商银行...", "label": {"company": {"浙商银行": [[0,3]]}}}
     这里:    {"tokens": ["浙","商","银","行",...], "ner_tags": ["B-ORG","I-ORG",...]}
  2. 只有 3 类实体（PER / ORG / LOC），共 7 个标签，标签体系读 label_names.json
  3. test 集带标签（不像 CLUE cluener2020 的 test 是空的），可以直接在 test 上评估
  4. 保留了 word_ids() 子词对齐：虽然实测 tokens 全是单字，但代码不依赖这个前提

使用方式：
  from dataset import build_label_schema, build_dataloaders
"""

import json
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "peoples_daily"

# 兜底标签表：与 data/peoples_daily/label_names.json 一致，文件缺失时用它
DEFAULT_LABELS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]


def build_label_schema(data_dir: Optional[Path] = None) -> tuple[list[str], dict[str, int], dict[int, str]]:
    """读取标签表，返回 (labels, label2id, id2label)。

    标签顺序以 label_names.json 为准（O 在 0，后面是 B/I 成对出现）。
    顺序必须固定，否则 checkpoint 的 id2label 会和预测对不上。
    """
    d = data_dir or DATA_DIR
    path = d / "label_names.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            labels = json.load(f)
    else:
        labels = list(DEFAULT_LABELS)

    label2id = {lbl: i for i, lbl in enumerate(labels)}
    id2label = {i: lbl for lbl, i in label2id.items()}
    return labels, label2id, id2label


def entity_types(labels: list[str]) -> list[str]:
    """从 BIO 标签表里抽出实体类型，['B-PER','I-PER',...] → ['PER','ORG','LOC']（去重保序）。"""
    types = []
    for lbl in labels:
        if lbl.startswith(("B-", "I-")):
            t = lbl[2:]
            if t not in types:
                types.append(t)
    return types


class PeoplesDailyDataset(Dataset):
    """人民日报 NER 的 PyTorch Dataset。

    流程：
      tokens(逐字) + ner_tags(BIO 字符串)
        → BertTokenizer(is_split_into_words=True)
        → word_ids() 对齐：每个字的首个子词保留标签，其余设 -100
        → input_ids / attention_mask / token_type_ids / labels
    """

    def __init__(
        self,
        records: list,
        tokenizer: BertTokenizer,
        label2id: dict,
        max_length: int = 128,
    ):
        self.records = records
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        row = self.records[idx]
        tokens: list[str] = row["tokens"]
        tags: list[str] = row["ner_tags"]

        # 1. BIO 字符串 → id。未知标签按 O 处理，避免一条脏数据炸掉整个训练
        char_labels = [self.label2id.get(t, self.label2id["O"]) for t in tags]

        # 2. is_split_into_words=True：让 word_ids() 的下标等于字在 tokens 里的下标
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        # 3. 子词对齐
        #    word_ids() 形如 [None, 0, 1, 1, 2, ..., None]：
        #      None  → [CLS] / [SEP] / [PAD]，设 -100
        #      首次出现的字 → 用它的 BIO 标签
        #      同一个字的后续子词 → -100（中文几乎不会出现，保留正确性）
        word_ids = encoding.word_ids(batch_index=0)
        aligned_labels = []
        prev_word_id = None
        for wid in word_ids:
            if wid is None:
                aligned_labels.append(-100)
            elif wid != prev_word_id:
                aligned_labels.append(char_labels[wid] if wid < len(char_labels) else -100)
                prev_word_id = wid
            else:
                aligned_labels.append(-100)

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "token_type_ids": encoding["token_type_ids"].squeeze(0),
            "labels": torch.tensor(aligned_labels, dtype=torch.long),
        }


def load_records(split: str, data_dir: Optional[Path] = None) -> list:
    d = data_dir or DATA_DIR
    with open(d / f"{split}.json", "r", encoding="utf-8") as f:
        return json.load(f)


def build_dataloaders(
    tokenizer: BertTokenizer,
    label2id: dict,
    batch_size: int = 32,
    max_length: int = 128,
    data_dir: Optional[Path] = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """构建训练/验证/测试 DataLoader，返回 (train, val, test)。"""
    d = data_dir or DATA_DIR
    train_ds = PeoplesDailyDataset(load_records("train", d), tokenizer, label2id, max_length)
    val_ds = PeoplesDailyDataset(load_records("validation", d), tokenizer, label2id, max_length)
    test_ds = PeoplesDailyDataset(load_records("test", d), tokenizer, label2id, max_length)

    print(f"数据集规模：训练={len(train_ds)}，验证={len(val_ds)}，测试={len(test_ds)}")

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0),
    )
