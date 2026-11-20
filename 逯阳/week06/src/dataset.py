"""
TNEWS 数据集封装

教学重点：
  1. PyTorch Dataset 的标准写法（__len__ / __getitem__）
  2. BERT tokenizer 的参数含义：max_length、truncation、padding
  3. DataLoader 的 batch 组装：模型需要 input_ids、attention_mask、token_type_ids

使用方式：
  from dataset import TNEWSDataset, build_dataloaders
"""

import json
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer


class TNEWSDataset(Dataset):
    """
    每条样本经过 tokenizer 处理后返回：
      input_ids      : [max_length]  — token id 序列，含 [CLS] 和 [SEP]
      attention_mask : [max_length]  — 1=真实 token，0=padding
      token_type_ids : [max_length]  — 单句分类全为 0
      label          : scalar        — 类别 id，test 集为 -1
    """

    def __init__(
        self,
        data_path: Path,
        tokenizer: BertTokenizer,
        max_length: int = 128,
    ):
        with open(data_path, encoding="utf-8") as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        encoding = self.tokenizer(
            item["sentence"],
            max_length=self.max_length,
            truncation=True,       # 超出 max_length 时截断
            padding="max_length",  # 不足 max_length 时用 [PAD] 填充
            return_tensors="pt",   # 直接返回 PyTorch tensor
        )
        return {
            "input_ids":      encoding["input_ids"].squeeze(0),       # [max_length]
            "attention_mask": encoding["attention_mask"].squeeze(0),  # [max_length]
            "token_type_ids": encoding["token_type_ids"].squeeze(0),  # [max_length]
            "label":          torch.tensor(item["label"], dtype=torch.long),
        }


def build_dataloaders(
    data_dir: Path,
    tokenizer: BertTokenizer,
    max_length: int = 128,
    batch_size: int = 32,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    一次性构建 train / val / test 三个 DataLoader。
    num_workers=0 在 Windows 上更稳定（避免多进程 pickle 问题）。
    """
    train_ds = TNEWSDataset(data_dir / "train.json", tokenizer, max_length)
    val_ds   = TNEWSDataset(data_dir / "val.json",   tokenizer, max_length)
    test_ds  = TNEWSDataset(data_dir / "test.json",  tokenizer, max_length)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, num_workers=num_workers)

    print(f"DataLoader 构建完成")
    print(f"  train: {len(train_ds)} 条, {len(train_loader)} batch")
    print(f"  val  : {len(val_ds)} 条, {len(val_loader)} batch")
    print(f"  test : {len(test_ds)} 条, {len(test_loader)} batch")

    return train_loader, val_loader, test_loader
