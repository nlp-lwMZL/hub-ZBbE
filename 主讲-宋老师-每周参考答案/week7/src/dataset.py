# -*- coding: utf-8 -*-
"""
PyTorch Dataset：把「BIO 标签的 JSON」变成 BERT 能吃的 input_ids / labels。

这一层是 BIO 任务真正的工程核心，三个坑必须在这里处理掉：

1. **字符级标签 → 子词级标签**
   本数据集的 tokens 已经是「一个字一个 token」，看起来直接对齐就行；
   但 BERT 的 WordPiece 可能把一个字/词切成多个子词（中文少见，英文/数字常见）。
   规则：**只有第一个子词背标签**，后续子词和 [CLS]/[SEP]/[PAD] 一律 -100，
   这样 CrossEntropyLoss 会忽略它们，不会把一个实体重复算好几遍。

2. **截断**
   max_length=128 会切掉长句的尾巴。被切掉的实体在评估里就是「漏检」，
   所以 dataset 要把 `truncated` 标记带出去，评估时单独统计，不能假装没发生。

3. **padding 位置不能算 loss**
   本作业为了简单，统一 padding 到 max_length（而不是动态 padding）。
   所有 pad 位 labels=-100，attention_mask=0。

使用方式：
  from dataset import build_dataloaders
  train_loader, val_loader, test_loader = build_dataloaders(...)
"""

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader


class NerDataset(Dataset):
    """
    一条样本 = 一句「tokens + ner_tags」。

    __getitem__ 返回的字段：
      input_ids / attention_mask / token_type_ids：BERT 输入
      labels：子词级标签 id，-100 表示不参与 loss
      word_ids：每个子词对应的字符下标，-1 表示特殊 token / 非首子词
                （评估时用它把 token 预测映射回字符，再还原成 span）
      char_len：原句字符数（截断前）
      kept_chars：实际进入模型的字符数（截断后）
      truncated：是否被截断
      index：原记录下标，写预测文件时用来对齐
    """

    def __init__(self, records: list[dict], tokenizer, label2id: dict,
                 max_length: int = 128):
        self.records = records
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length

        # 启动时先校验一遍标签，避免训练到一半才发现某条数据有脏标签
        valid = set(label2id)
        for i, row in enumerate(records):
            bad = [t for t in row["ner_tags"] if t not in valid]
            if bad:
                raise ValueError(f"第 {i} 条数据含未知标签 {bad[:5]}（合法标签：{sorted(valid)}）")
            if len(row["tokens"]) != len(row["ner_tags"]):
                raise ValueError(
                    f"第 {i} 条数据 tokens 与 ner_tags 长度不一致："
                    f"{len(row['tokens'])} vs {len(row['ner_tags'])}"
                )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        row = self.records[idx]
        chars: list[str] = row["tokens"]
        tags: list[str] = row["ner_tags"]

        # 先把字符级 BIO 标签转成 id；后面按子词对齐时直接取 id
        char_label_ids = [self.label2id[t] for t in tags]

        # is_split_into_words=True：告诉 tokenizer「传进来的是已经切好的词/字」，
        # 这样它才会维护 word_ids()，让我们知道每个子词属于第几个字符。
        enc = self.tokenizer(
            chars,
            is_split_into_words=True,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        word_ids = enc.word_ids(0)  # list[int|None]，长度 == max_length

        aligned_labels: list[int] = []
        char_index: list[int] = []
        prev_word_id = None
        kept_chars = 0

        for wid in word_ids:
            if wid is None:
                # [CLS] / [SEP] / [PAD]：不参与 loss，也不映射到字符
                aligned_labels.append(-100)
                char_index.append(-1)
                continue

            kept_chars = max(kept_chars, wid + 1)
            if wid != prev_word_id:
                # 这个字符的第一个子词：背标签
                aligned_labels.append(char_label_ids[wid])
                char_index.append(wid)
                prev_word_id = wid
            else:
                # 同一个字符的后续子词：不背标签（否则一个实体被重复计 loss）
                aligned_labels.append(-100)
                char_index.append(-1)

        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        token_type_ids = enc.get("token_type_ids")
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        else:
            token_type_ids = token_type_ids.squeeze(0)

        char_len = len(chars)
        truncated = kept_chars < char_len

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "labels": torch.tensor(aligned_labels, dtype=torch.long),
            "word_ids": torch.tensor(char_index, dtype=torch.long),
            "char_len": torch.tensor(char_len, dtype=torch.long),
            "kept_chars": torch.tensor(kept_chars, dtype=torch.long),
            "truncated": torch.tensor(bool(truncated), dtype=torch.bool),
            "index": torch.tensor(idx, dtype=torch.long),
        }


def load_records(split: str, data_dir: str | Path) -> list[dict]:
    """读取 train / validation / test 三个 JSON 文件。"""
    path = Path(data_dir) / f"{split}.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_dataloaders(
    data_dir: str | Path,
    tokenizer,
    label2id: dict,
    batch_size: int = 32,
    max_length: int = 128,
    num_workers: int = 0,
    limit_train: int = 0,
    limit_eval: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    构建三个 DataLoader。

    limit_train / limit_eval：只取前 N 条，专为「冒烟测试」准备（几十秒看通不通），
    正式训练保持 0 = 全量。

    num_workers 在 Windows 上保持 0：多进程 worker 会把 tokenizer 再 pickle 一份，
    课堂机器上反而更容易卡住；本数据集只有 2 万条，单进程足够。
    """
    train_records = load_records("train", data_dir)
    val_records = load_records("validation", data_dir)
    test_records = load_records("test", data_dir)

    if limit_train > 0:
        train_records = train_records[:limit_train]
    if limit_eval > 0:
        val_records = val_records[:limit_eval]
        test_records = test_records[:limit_eval]

    train_ds = NerDataset(train_records, tokenizer, label2id, max_length)
    val_ds = NerDataset(val_records, tokenizer, label2id, max_length)
    test_ds = NerDataset(test_records, tokenizer, label2id, max_length)

    print(f"数据集规模：训练={len(train_ds)}，验证={len(val_ds)}，测试={len(test_ds)}")
    print(f"max_length={max_length}（含 [CLS]/[SEP]）")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
