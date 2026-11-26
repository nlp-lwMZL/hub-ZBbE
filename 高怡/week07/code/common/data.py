"""事件抽取（序列标注）共用数据层。

职责
----
1. 读取 split JSONL（含 event.trigger / event.args 的原始 start/end）；
2. 把长公告按滑动窗口切成 chunk，保证实体不被整段丢弃；
3. chunk 内把 trigger 与实体型槽位映射为字符级 BIO 标签；
4. 字符级词表（BiLSTM 用；BERT 训练时可以不用，但仍可复用 make_chunks）。

数据契约（与 build_event_annotation_dataset.py 对齐）
----------------------------------------------------
每条记录：{"text": str, "source": str, "event": {"type": str,
          "trigger": {"text","start","end"},
          "args": [{"role","text","start","end"}, ...]}}
所有 span 的 text 严格等于 text[start:end]，因此可直接切标。
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import torch
from torch.utils.data import DataLoader, Dataset

from .label import (
    BIO_LABEL2ID,
    ENTITY_ROLES,
    NUM_BIO_LABELS,
    OUTSIDE_LABEL,
    PAD_LABEL_ID,
    TRIGGER_LABEL,
)

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


# ---------------------------------------------------------------- 记录读取


def load_records(path: str | Path, limit: int = 0) -> list[dict]:
    """读取 JSONL。limit>0 时只取前 N 条（调试用）。"""
    records: list[dict] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
            if limit and len(records) >= limit:
                break
    return records


def extract_spans(rec: dict) -> list[tuple[int, int, str]]:
    """把一条记录的 trigger + 实体型槽位转成 [(start, end, base_label)]。

    - trigger 的基标签固定为 TRIGGER；
    - args 中 role 不在 ENTITY_ROLES 白名单的（说明/内容类）直接跳过；
    - 过滤越界或空 span。
    """
    text = rec.get("text") or ""
    n = len(text)
    spans: list[tuple[int, int, str]] = []

    ev = rec.get("event") or {}
    trig = ev.get("trigger") or {}
    s, e = int(trig.get("start", -1)), int(trig.get("end", -1))
    if 0 <= s < e <= n:
        spans.append((s, e, TRIGGER_LABEL))

    for arg in ev.get("args") or []:
        role = arg.get("role")
        if role not in ENTITY_ROLES:
            continue
        s, e = int(arg.get("start", -1)), int(arg.get("end", -1))
        if 0 <= s < e <= n:
            spans.append((s, e, role))
    return spans


# ---------------------------------------------------------------- 滑窗切块


@dataclass
class Chunk:
    text: str
    spans: list[tuple[int, int, str]] = field(default_factory=list)
    source: str = ""
    offset: int = 0  # chunk 在原文档中的起始字符位置


def make_chunks(
    text: str,
    spans: Sequence[tuple[int, int, str]],
    max_len: int = 512,
    stride: int = 384,
    source: str = "",
) -> list[Chunk]:
    """把长文本切成多个窗口，窗口内实体按交集裁剪并转成相对 offset。

    stride < max_len 时相邻窗口有重叠，保证跨界实体至少在一个窗口内完整出现。
    """
    n = len(text)
    if n == 0:
        return []
    if max_len <= 0:
        raise ValueError("max_len 必须为正数")
    if stride <= 0 or stride > max_len:
        stride = max_len

    if n <= max_len:
        windows = [(0, n)]
    else:
        windows = []
        start = 0
        while True:
            end = min(start + max_len, n)
            windows.append((start, end))
            if end >= n:
                break
            start += stride

    chunks: list[Chunk] = []
    # 预判每个 span 是否存在「完整包含它」的窗口。存在则只在该窗口完整标注，
    # 避免同一个实体在相邻窗口被裁成两段（会让文档级评估重复计数）。
    full_covered: list[bool] = [
        any(ws <= a and b <= we for ws, we in windows) for a, b, _ in spans
    ]
    for ws, we in windows:
        local_spans: list[tuple[int, int, str]] = []
        for idx, (a, b, base) in enumerate(spans):
            if b <= ws or a >= we:
                continue
            if full_covered[idx]:
                # 只在完整包含的窗口里标注，片段一律跳过
                if ws <= a and b <= we:
                    local_spans.append((a - ws, b - ws, base))
            else:
                # 实体比窗口还长：无窗口能完整包含，退化为按交集裁剪
                ca, cb = max(a, ws), min(b, we)
                if cb > ca:
                    local_spans.append((ca - ws, cb - ws, base))
        chunks.append(Chunk(text[ws:we], local_spans, source, ws))
    return chunks


def chunk_to_label_ids(chunk: Chunk) -> list[int]:
    """chunk -> 字符级 BIO label id 序列（长度 == len(chunk.text)）。

    重叠实体处理：按 span 长度升序填充（短实体优先，通常是更精确的槽位），
    已被占用的字符跳过，避免一个字对应多个标签。
    """
    L = len(chunk.text)
    labels = [BIO_LABEL2ID[OUTSIDE_LABEL]] * L
    occupied = bytearray(L)
    for a, b, base in sorted(chunk.spans, key=lambda x: (x[1] - x[0], x[0])):
        if a < 0 or b > L or b <= a:
            continue
        if any(occupied[a:b]):
            continue
        labels[a] = BIO_LABEL2ID[f"B-{base}"]
        for i in range(a + 1, b):
            labels[i] = BIO_LABEL2ID[f"I-{base}"]
        occupied[a:b] = b"\x01" * (b - a)
    return labels


# ---------------------------------------------------------------- 词表


class Vocab:
    """字符级词表。0=<pad>，1=<unk>，其余按词频排序。"""

    def __init__(self, token2id: dict[str, int]):
        self.token2id = token2id
        self.id2token = {i: t for t, i in token2id.items()}
        self.pad_id = token2id[PAD_TOKEN]
        self.unk_id = token2id[UNK_TOKEN]

    def __len__(self) -> int:
        return len(self.token2id)

    def encode(self, text: str) -> list[int]:
        return [self.token2id.get(ch, self.unk_id) for ch in text]

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.token2id, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> "Vocab":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))


def build_vocab(chunks: Iterable[Chunk], min_freq: int = 2) -> Vocab:
    """从训练集 chunks 构建字符词表。"""
    from collections import Counter

    counter: Counter[str] = Counter()
    for ch in chunks:
        counter.update(ch.text)
    token2id: dict[str, int] = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for ch, freq in counter.most_common():
        if freq < min_freq:
            continue
        token2id[ch] = len(token2id)
    return Vocab(token2id)


# ---------------------------------------------------------------- Dataset


class EventBIODataset(Dataset):
    """把记录列表切块并编码为 (input_ids, labels, attention_mask) 样本。"""

    def __init__(
        self,
        records: Sequence[dict],
        vocab: Vocab,
        max_len: int = 512,
        stride: int = 384,
    ):
        self.vocab = vocab
        self.examples: list[dict] = []
        for rec in records:
            text = rec.get("text") or ""
            spans = extract_spans(rec)
            for chunk in make_chunks(
                text, spans, max_len=max_len, stride=stride,
                source=str(rec.get("source", "")),
            ):
                if not chunk.text:
                    continue
                input_ids = vocab.encode(chunk.text)
                labels = chunk_to_label_ids(chunk)
                self.examples.append(
                    {
                        "input_ids": input_ids,
                        "labels": labels,
                        "source": chunk.source,
                        "offset": chunk.offset,
                        "length": len(input_ids),
                        "event_type": (rec.get("event") or {}).get("type", ""),
                    }
                )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        return self.examples[idx]


def collate_fn(batch: Sequence[dict], pad_id: int = 0) -> dict:
    """把变长样本 pad 到 batch 内最大长度。labels 用 PAD_LABEL_ID 填充。"""
    max_len = max(item["length"] for item in batch)
    bsz = len(batch)
    input_ids = torch.full((bsz, max_len), pad_id, dtype=torch.long)
    labels = torch.full((bsz, max_len), PAD_LABEL_ID, dtype=torch.long)
    mask = torch.zeros((bsz, max_len), dtype=torch.bool)
    meta: list[dict] = []
    for i, item in enumerate(batch):
        L = item["length"]
        input_ids[i, :L] = torch.tensor(item["input_ids"], dtype=torch.long)
        labels[i, :L] = torch.tensor(item["labels"], dtype=torch.long)
        mask[i, :L] = True
        meta.append(
            {
                "source": item["source"],
                "offset": item["offset"],
                "length": L,
                "event_type": item.get("event_type", ""),
            }
        )
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": mask,
        "meta": meta,
    }


def compute_label_counts(dataset: EventBIODataset) -> list[int]:
    """统计训练集每个 BIO 标签的出现次数（用于类别不平衡的先验/权重）。"""
    counts = [0] * NUM_BIO_LABELS
    for ex in dataset.examples:
        for lid in ex["labels"]:
            if 0 <= lid < NUM_BIO_LABELS:
                counts[lid] += 1
    return counts


def build_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 0,
    pad_id: int = 0,
    seed: int = 42,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda b: collate_fn(b, pad_id=pad_id),
        generator=generator if shuffle else None,
    )
