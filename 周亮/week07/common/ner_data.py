"""
统一数据层 —— cdtier NER，三类方法（BiLSTM / RoBERTa / Qwen-SFT）共用同一份标签体系与读取。

迁移自 homework/week7/ner_data.py，关键改造：
  1. 默认数据目录改到 cdtier/clean（3091/387/387，无文本泄漏），不再是人民日报 PER/ORG/LOC；
  2. tokenizer 默认 hfl/chinese-roberta-wwm-ext，不再假设 bert-base-chinese「一字一 token」；
     仍然是逐字 is_split_into_words + word_ids() 对齐，因为 clean 数据的 tokens 就是字符序列；
  3. 新增 CharDataset：字符级 BiLSTM 用（查字表、不做子词），与 BERT 分支共用 LabelScheme；
  4. 保留四类防坑：-100 忽略位、UNK 屏蔽、截断残段屏蔽、动态 padding。

【BIO 标签体系】来自 clean/label_names.json：
  O, B-Attacker, I-Attacker, B-Campaign, I-Campaign,
  B-Industry, I-Industry, B-Region, I-Region, B-Tools, I-Tools   （共 11 个）

【数据记录契约】clean/{train,validation,test}.json 是 JSON 数组，每条：
  {id, tokens:[str], ner_tags:[str], text:str, entities:[{type,start,end,text,...}]}
  start/end 是**字符级**左闭右开区间（text 已去掉空格，tokens 拼起来 == text）。

【自检】python common/ner_data.py
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Iterable, Sequence

import sys
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

_CODE_DIR = Path(__file__).resolve().parent.parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))
from common.paths import DEFAULT_DATA_DIR as _FALLBACK_DATA_DIR, get_data_dir

IGNORE_INDEX: int = -100
DEFAULT_TOKENIZER: str = "hfl/chinese-roberta-wwm-ext"
TRAIN_SPLITS = frozenset({"train", "training"})

__all__ = [
    "IGNORE_INDEX", "DEFAULT_TOKENIZER", "LabelScheme", "DataConfig", "NERDataset",
    "CharDataset", "DataLoaderFactory", "create_tokenizer", "load_records",
    "pad_collate", "collate_char", "build_dataloaders", "decode_bio_ids", "make_char_vocab",
]


# --------------------------------------------------------------------------- #
# BIO 标签体系（沿用现有实现，逻辑未改，仅默认数据不同）
# --------------------------------------------------------------------------- #
class LabelScheme:
    """BIO 标签体系：O 恒为下标 0，B-/I- 成对，附 CRF 转移/起始约束。"""

    def __init__(self, labels: Sequence[str]) -> None:
        labels = list(labels)
        if not labels:
            raise ValueError("标签体系不能为空")
        if len(set(labels)) != len(labels):
            raise ValueError(f"标签体系存在重复项: {labels}")
        if labels[0] != "O":
            raise ValueError(f"BIO 体系要求 O 位于下标 0，实际首项是 {labels[0]!r}")
        for tag in labels:
            if tag == "O":
                continue
            prefix, _, etype = tag.partition("-")
            if prefix not in ("B", "I") or not etype:
                raise ValueError(f"非法标签 {tag!r}，应为 'O' / 'B-XXX' / 'I-XXX'")
        for tag in labels:
            if tag.startswith("I-") and ("B-" + tag[2:]) not in labels:
                raise ValueError(f"标签 {tag!r} 缺少配对的 B-{tag[2:]}")

        self.labels: list[str] = labels
        self.label2id: dict[str, int] = {t: i for i, t in enumerate(labels)}
        self.id2label: dict[int, str] = {i: t for t, i in self.label2id.items()}
        seen: list[str] = []
        for tag in labels:
            if tag == "O":
                continue
            if tag[2:] not in seen:
                seen.append(tag[2:])
        self.entity_types: list[str] = seen
        self._transition_mask: torch.Tensor | None = None
        self._start_mask: torch.Tensor | None = None

    @classmethod
    def from_entity_types(cls, entity_types: Sequence[str]) -> "LabelScheme":
        labels = ["O"]
        for etype in entity_types:
            labels.extend([f"B-{etype}", f"I-{etype}"])
        return cls(labels)

    @classmethod
    def from_json(cls, path: str | Path) -> "LabelScheme":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = data["labels"] if "labels" in data else [t for t, _ in sorted(data.items(), key=lambda kv: kv[1])]
        return cls(list(data))

    @property
    def num_labels(self) -> int:
        return len(self.labels)

    @property
    def o_index(self) -> int:
        return self.label2id["O"]

    def __len__(self) -> int:
        return len(self.labels)

    def encode_tags(self, tags: Sequence[str]) -> list[int]:
        out = []
        for i, t in enumerate(tags):
            try:
                out.append(self.label2id[t])
            except KeyError:
                raise KeyError(f"第 {i} 个标签 {t!r} 不在体系内：{self.labels}") from None
        return out

    def decode(self, label_id: int) -> str:
        return self.id2label[int(label_id)]

    def transition_mask(self) -> torch.Tensor:
        """合法转移 mask：I-X 只能接 B-X / I-X，其余 True。"""
        if self._transition_mask is None:
            n = self.num_labels
            mask = torch.zeros(n, n, dtype=torch.bool)
            for i, src in enumerate(self.labels):
                for j, dst in enumerate(self.labels):
                    if dst == "O" or dst.startswith("B-"):
                        mask[i, j] = True
                    else:  # I-X
                        etype = dst[2:]
                        mask[i, j] = src in (f"B-{etype}", f"I-{etype}")
            self._transition_mask = mask
        return self._transition_mask

    def start_mask(self) -> torch.Tensor:
        if self._start_mask is None:
            self._start_mask = torch.tensor(
                [t == "O" or t.startswith("B-") for t in self.labels], dtype=torch.bool
            )
        return self._start_mask

    def describe(self) -> str:
        lines = [f"{'id':>4}  {'tag':<14} 类型"]
        lines.append("-" * 34)
        for i, t in enumerate(self.labels):
            lines.append(f"{i:>4}  {t:<14} {t[2:] if t != 'O' else '-'}")
        return "\n".join(lines)

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.labels, ensure_ascii=False, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------- #
# 配置
# --------------------------------------------------------------------------- #
@dataclass
class DataConfig:
    data_dir: Path = _FALLBACK_DATA_DIR
    tokenizer_name_or_path: str = DEFAULT_TOKENIZER
    #: RoBERTa-wwm 对中文也是（基本）一字一 token，但数字/英文可能被切开；
    #: clean 数据句长 p50≈65、max 2232，取 256 覆盖绝大多数；正式训练会报截断率。
    max_length: int = 256
    label_all_subtokens: bool = False
    ignore_unk: bool = True
    ignore_truncated_entity: bool = True
    batch_size: int = 16
    eval_batch_size: int = 32
    num_workers: int = 0
    pin_memory: bool = False
    return_token_type_ids: bool = False   # RoBERTa 不需要 token_type_ids
    return_word_ids: bool = False

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        if self.max_length < 4:
            raise ValueError(f"max_length 至少为 4，实际 {self.max_length}")


# --------------------------------------------------------------------------- #
# 记录读取
# --------------------------------------------------------------------------- #
def load_records(split: str, data_dir: str | Path | None = None) -> list[dict]:
    """读取一个 split。split: train / validation / test。"""
    d = Path(data_dir) if data_dir is not None else get_data_dir()
    path = d / f"{split}.json"
    if not path.exists():
        available = sorted(p.stem for p in d.glob("*.json"))
        raise FileNotFoundError(f"找不到 {path}；目录下有: {available}")
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"{path} 顶层应为数组，实际 {type(records).__name__}")
    return records


# --------------------------------------------------------------------------- #
# BERT/RoBERTa 序列标注数据集
# --------------------------------------------------------------------------- #
class NERDataset(Dataset):
    """逐字对齐的序列标注数据集，构造时一次性分词，之后每 epoch 只做取数。"""

    def __init__(self, records: Sequence[dict], tokenizer: PreTrainedTokenizerBase,
                 scheme: LabelScheme, config: DataConfig | None = None, verbose: bool = True) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.scheme = scheme
        self.config = config or DataConfig()
        if not getattr(tokenizer, "is_fast", False):
            raise TypeError("需要 fast tokenizer（提供 word_ids() 做子词对齐）")
        self.pad_token_id = int(tokenizer.pad_token_id)
        self.unk_token_id = int(tokenizer.unk_token_id)
        self._continuation = {
            scheme.label2id[f"B-{t}"]: scheme.label2id[f"I-{t}"] for t in scheme.entity_types
        }
        self.stats = {k: 0 for k in (
            "num_samples", "num_tokens", "num_subtokens", "truncated_samples",
            "truncated_chars", "unk_subtokens", "dropped_chars", "ignored_entity_tokens")}
        if verbose:
            print(f"  分词并对齐标签：{len(records):,} 条 ...")
        self.features = [self._encode(r) for r in records]
        self.stats["num_samples"] = len(self.features)
        if verbose:
            s = self.stats
            print(f"  完成：平均 {s['num_subtokens'] / max(1, len(self.features)):.1f} 子词/条，"
                  f"截断 {s['truncated_samples']:,} ({s['truncated_samples'] / max(1, len(self.features)) * 100:.2f}%)")

    def _encode(self, record: dict) -> dict[str, list[int]]:
        tokens, tags = record["tokens"], record["ner_tags"]
        if len(tokens) != len(tags):
            raise ValueError(f"tokens/ner_tags 长度不一致：{len(tokens)} vs {len(tags)}，id={record.get('id')}")
        tag_ids = self.scheme.encode_tags(tags)
        cfg = self.config
        enc = self.tokenizer(tokens, is_split_into_words=True, add_special_tokens=True,
                             truncation=False, return_attention_mask=True, verbose=False)
        word_ids_full = list(enc.word_ids(0))
        truncated = len(enc["input_ids"]) > cfg.max_length
        if truncated:
            enc = self.tokenizer(tokens, is_split_into_words=True, add_special_tokens=True,
                                 truncation=True, max_length=cfg.max_length,
                                 return_attention_mask=True, verbose=False)
        input_ids = list(enc["input_ids"])
        word_ids = list(enc.word_ids(0))

        labels: list[int] = []
        prev_wid: int | None = None
        unk_hit = 0
        for i, wid in enumerate(word_ids):
            if wid is None:
                labels.append(IGNORE_INDEX); prev_wid = None; continue
            if wid >= len(tag_ids):
                labels.append(IGNORE_INDEX); prev_wid = wid; continue
            if wid != prev_wid:
                labels.append(tag_ids[wid]); prev_wid = wid
            else:
                labels.append(self._continuation.get(tag_ids[wid], tag_ids[wid]) if cfg.label_all_subtokens else IGNORE_INDEX)
            if cfg.ignore_unk and input_ids[i] == self.unk_token_id:
                if labels[-1] != IGNORE_INDEX:
                    unk_hit += 1
                labels[-1] = IGNORE_INDEX

        ignored = 0
        if truncated and cfg.ignore_truncated_entity:
            ignored = self._mask_truncated_entity(labels, word_ids, tags)

        seen_full = {w for w in word_ids_full if w is not None}
        seen_kept = {w for w in word_ids if w is not None}
        self.stats["num_tokens"] += len(tokens)
        self.stats["num_subtokens"] += len(input_ids)
        self.stats["truncated_samples"] += int(truncated)
        self.stats["truncated_chars"] += len(seen_full) - len(seen_kept)
        self.stats["dropped_chars"] += len(tokens) - len(seen_full)
        self.stats["unk_subtokens"] += unk_hit
        self.stats["ignored_entity_tokens"] += ignored

        feat: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": list(enc["attention_mask"]),
            "labels": labels,
            "word_ids": [-1 if w is None else w for w in word_ids],
        }
        tti = enc.get("token_type_ids")
        if cfg.return_token_type_ids and tti is not None:
            feat["token_type_ids"] = list(tti)
        return feat

    def _mask_truncated_entity(self, labels: list[int], word_ids: list[int | None], tags: list[str]) -> int:
        last = max((w for w in word_ids if w is not None), default=None)
        if last is None or last >= len(tags) or tags[last] == "O":
            return 0
        etype = tags[last][2:]
        start = last
        if tags[last].startswith("I-"):
            while start > 0 and tags[start - 1] == f"I-{etype}":
                start -= 1
            if start > 0 and tags[start - 1] == f"B-{etype}":
                start -= 1
        n = 0
        for i, w in enumerate(word_ids):
            if w is not None and start <= w <= last and labels[i] != IGNORE_INDEX:
                labels[i] = IGNORE_INDEX; n += 1
        return n

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, i: int) -> dict[str, list[int]]:
        return self.features[i]

    def original(self, i: int) -> dict:
        return self.records[i]


# --------------------------------------------------------------------------- #
# 字符级数据集（BiLSTM 用，无子词、无预训练）
# --------------------------------------------------------------------------- #
def make_char_vocab(records: Sequence[dict]) -> dict[str, int]:
    """字表只从训练集字符去重构建（用 val/test 建表 = 数据泄漏）。<PAD>=0, <UNK>=1。"""
    chars: set[str] = set()
    for r in records:
        chars.update(r["tokens"])
    return {"<PAD>": 0, "<UNK>": 1, **{c: i + 2 for i, c in enumerate(sorted(chars))}}


class CharDataset(Dataset):
    """字符级 BiLSTM 数据集：一个汉字一个 id，定长由 collate_char 在 batch 内补齐。"""

    def __init__(self, records: Sequence[dict], char2id: dict[str, int], scheme: LabelScheme,
                 max_length: int = 256, verbose: bool = True) -> None:
        self.records = records
        self.scheme = scheme
        self.max_length = max_length
        self.features = []
        n_trunc = 0
        for r in records:
            ids = [char2id.get(c, 1) for c in r["tokens"]][:max_length]
            tags = scheme.encode_tags(r["ner_tags"])[:max_length]
            n_trunc += int(len(r["tokens"]) > max_length)
            self.features.append({"input_ids": ids, "labels": tags})
        if verbose:
            print(f"  字符化 {len(records):,} 条，字表 {len(char2id)}，截断 {n_trunc} "
                  f"({n_trunc / max(1, len(records)) * 100:.2f}%)")

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, i: int) -> dict[str, list[int]]:
        return self.features[i]

    def original(self, i: int) -> dict:
        return self.records[i]


# --------------------------------------------------------------------------- #
# 动态 padding
# --------------------------------------------------------------------------- #
def pad_collate(batch: Sequence[dict[str, list[int]]], pad_token_id: int,
                label_pad_id: int = IGNORE_INDEX, return_token_type_ids: bool = False,
                return_word_ids: bool = False) -> dict[str, Any]:
    max_len = max(len(x["input_ids"]) for x in batch)
    bs = len(batch)
    input_ids = torch.full((bs, max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((bs, max_len), dtype=torch.long)
    labels = torch.full((bs, max_len), label_pad_id, dtype=torch.long)
    tti = torch.zeros((bs, max_len), dtype=torch.long) if return_token_type_ids else None
    for i, x in enumerate(batch):
        n = len(x["input_ids"])
        input_ids[i, :n] = torch.tensor(x["input_ids"], dtype=torch.long)
        attention_mask[i, :n] = torch.tensor(x["attention_mask"], dtype=torch.long)
        labels[i, :n] = torch.tensor(x["labels"], dtype=torch.long)
        if tti is not None and "token_type_ids" in x:
            tti[i, :n] = torch.tensor(x["token_type_ids"], dtype=torch.long)
    out: dict[str, Any] = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
    if tti is not None:
        out["token_type_ids"] = tti
    if return_word_ids:
        out["word_ids"] = [x["word_ids"] for x in batch]
    return out


def collate_char(batch: Sequence[dict[str, list[int]]], pad_id: int = 0,
                 label_pad_id: int = IGNORE_INDEX) -> dict[str, torch.Tensor]:
    """字符级 collate：补到 batch 内最长，attention_mask = 连续前缀。"""
    max_len = max(len(x["input_ids"]) for x in batch)
    bs = len(batch)
    input_ids = torch.full((bs, max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((bs, max_len), dtype=torch.long)
    labels = torch.full((bs, max_len), label_pad_id, dtype=torch.long)
    for i, x in enumerate(batch):
        n = len(x["input_ids"])
        input_ids[i, :n] = torch.tensor(x["input_ids"], dtype=torch.long)
        attention_mask[i, :n] = 1
        labels[i, :n] = torch.tensor(x["labels"], dtype=torch.long)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


# --------------------------------------------------------------------------- #
# tokenizer / loader 工厂
# --------------------------------------------------------------------------- #
def create_tokenizer(name_or_path: str | Path | PreTrainedTokenizerBase | None = None) -> PreTrainedTokenizerBase:
    if isinstance(name_or_path, PreTrainedTokenizerBase):
        tok = name_or_path
    else:
        tok = AutoTokenizer.from_pretrained(str(name_or_path or DEFAULT_TOKENIZER), use_fast=True)
    if not getattr(tok, "is_fast", False):
        raise TypeError(f"{type(tok).__name__} 不是 fast tokenizer，无法用 word_ids() 对齐")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token or "[PAD]"
    return tok


class DataLoaderFactory:
    """统一构造 train/validation/test 的 DataLoader，同实例缓存 dataset。"""

    def __init__(self, tokenizer: str | Path | PreTrainedTokenizerBase | None,
                 scheme: LabelScheme, config: DataConfig | None = None) -> None:
        self.config = config or DataConfig()
        self.tokenizer = create_tokenizer(tokenizer or self.config.tokenizer_name_or_path)
        self.scheme = scheme
        self._datasets: dict[str, NERDataset] = {}
        self.collate_fn = partial(
            pad_collate, pad_token_id=int(self.tokenizer.pad_token_id),
            label_pad_id=IGNORE_INDEX,
            return_token_type_ids=self.config.return_token_type_ids,
            return_word_ids=self.config.return_word_ids,
        )

    def dataset(self, split: str) -> NERDataset:
        if split not in self._datasets:
            recs = load_records(split, self.config.data_dir)
            self._datasets[split] = NERDataset(recs, self.tokenizer, self.scheme, self.config)
        return self._datasets[split]

    def create(self, split: str, shuffle: bool | None = None,
               batch_size: int | None = None) -> DataLoader:
        is_train = split in TRAIN_SPLITS
        shuffle = is_train if shuffle is None else shuffle
        bs = batch_size or (self.config.batch_size if is_train else self.config.eval_batch_size)
        return DataLoader(self.dataset(split), batch_size=bs, shuffle=shuffle,
                          collate_fn=self.collate_fn, num_workers=self.config.num_workers,
                          pin_memory=self.config.pin_memory)

    def create_all(self, splits: Iterable[str] = ("train", "validation", "test")) -> dict[str, DataLoader]:
        return {s: self.create(s) for s in splits}


def build_dataloaders(tokenizer: str | Path | None = None, data_dir: str | Path | None = None,
                      batch_size: int = 16, max_length: int = 256, **kw: Any) -> dict[str, DataLoader]:
    d = Path(data_dir) if data_dir is not None else get_data_dir()
    cfg = DataConfig(data_dir=d, max_length=max_length, batch_size=batch_size, **kw)
    scheme = LabelScheme.from_json(d / "label_names.json")
    return DataLoaderFactory(tokenizer or cfg.tokenizer_name_or_path, scheme, cfg).create_all()


# --------------------------------------------------------------------------- #
# 解码（评估 / SFT 对齐共用）
# --------------------------------------------------------------------------- #
def decode_bio_ids(label_ids: Sequence[int], scheme: LabelScheme,
                   ignore_index: int = IGNORE_INDEX) -> list[dict[str, Any]]:
    """标签 id 序列 -> [{type,start,end,text_len}]；-100 处断开实体。"""
    spans: list[dict[str, Any]] = []
    cur_type: str | None = None
    cur_start = 0

    def close(end: int) -> None:
        nonlocal cur_type
        if cur_type is not None:
            spans.append({"type": cur_type, "start": cur_start, "end": end,
                          "text_len": end - cur_start + 1})
            cur_type = None

    for i, lid in enumerate(label_ids):
        lid = int(lid)
        if lid == ignore_index:
            close(i - 1); continue
        tag = scheme.decode(lid)
        if tag == "O":
            close(i - 1); continue
        prefix, _, etype = tag.partition("-")
        if prefix == "B":
            close(i - 1); cur_type, cur_start = etype, i
        else:
            if cur_type != etype:
                close(i - 1); cur_type, cur_start = etype, i
    close(len(label_ids) - 1)
    return spans


# --------------------------------------------------------------------------- #
# 自检
# --------------------------------------------------------------------------- #
def _run_self_test() -> None:
    print("=" * 74)
    print("common/ner_data 自检（cdtier/clean）")
    print("=" * 74)
    d = get_data_dir()
    scheme = LabelScheme.from_json(d / "label_names.json")
    print(f"\n【1】标签体系 ({scheme.num_labels})")
    print(scheme.describe())
    print(f"  合法起始: {[scheme.labels[i] for i in scheme.start_mask().nonzero().flatten().tolist()]}")

    print("\n【2】读取三个 split 并统计")
    for split in ("train", "validation", "test"):
        recs = load_records(split, d)
        lens = [len(r["tokens"]) for r in recs]
        n_ent = sum(len(r["entities"]) for r in recs)
        print(f"  {split:<11} {len(recs):>5} 句 | token min/med/max = "
              f"{min(lens)}/{sorted(lens)[len(lens)//2]}/{max(lens)} | 实体 {n_ent}")

    print("\n【3】字符级 BiLSTM 数据集")
    train = load_records("train", d)
    cv = make_char_vocab(train)
    cds = CharDataset(train[:64], cv, scheme, verbose=False)
    b = collate_char([cds[i] for i in range(4)])
    print(f"  字表 {len(cv)}，batch 形状 "
          f"{ {k: tuple(v.shape) for k, v in b.items()} }")

    print("\n【4】子词对齐自检（用本机可用的 tokenizer，验证 word_ids 对齐机制）")
    # 对齐逻辑与具体 tokenizer 无关；优先用环境变量指定的本地 tokenizer，
    # 再退回默认 RoBERTa。绝不在这里触发联网下载（本机常无外网，会卡住）。
    import os
    tok_name = os.environ.get("WEEK7_SMOKE_TOKENIZER", "").strip() or DEFAULT_TOKENIZER
    try:
        tok = create_tokenizer(tok_name)
        cfg = DataConfig(data_dir=d, max_length=128, return_word_ids=True)
        ds = NERDataset(train[:32], tok, scheme, cfg, verbose=False)
        feat = ds[0]
        print(f"  tokenizer={tok_name} 加载成功，"
              f"首条 {len(feat['input_ids'])} token / {len(feat['word_ids'])} word_ids")
        assert all(feat["labels"][i] == IGNORE_INDEX for i, w in enumerate(feat["word_ids"]) if w < 0)
        print("  ✓ 特殊位置（CLS/SEP）标签为 -100")
    except Exception as exc:                                   # noqa: BLE001
        print(f"  [跳过] tokenizer 加载失败：{type(exc).__name__}: {exc}")
        print("         本机冒烟可设 WEEK7_SMOKE_TOKENIZER=/本机/bert-base-chinese；"
              "正式训练在 Colab 用 RoBERTa")

    print("\n自检结束 ✓")


if __name__ == "__main__":
    _run_self_test()
