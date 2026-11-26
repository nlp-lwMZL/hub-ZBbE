"""RoBERTa 事件抽取训练脚本（序列标注，字符级 BIO 评估）。

特性（全部参数可配）
--------------------
- 分层学习率：encoder 逐层衰减 ``--layer-lr-decay``（=1.0 退化为 2 组）+ 分类头 ``--head-lr-mult``
- 权重衰减：默认 0.01，自动排除 bias 与 LayerNorm（可配）
- 学习率 warmup：``--warmup-ratio``（默认 0.1）+ 线性衰减，可被 ``--warmup-steps`` 覆盖
- 标签平滑：``--label-smoothing``（默认 0 = 关）
- Focal Loss：``--focal-gamma``（默认 0 = 关），与标签平滑共用同一 per-class 权重向量
- 对抗训练：``--adv {none,fgm}`` + ``--fgm-eps``（默认 none）
- CRF 头：``--crf`` 开启（默认关，纯 Softmax）；可叠加 ``--constrained`` 解码期 BIO 硬约束
- AMP：CUDA fp16+GradScaler；CPU bf16 autocast
- 梯度累积：``--grad-accum-steps``（CPU 上放大有效 batch）
- monitor 选优：f1 / event_exact / type_macro_f1

数据：data/dataset/split/{train,val}.jsonl（与 BiLSTM 共用同一 split）
输出：extract/output/roberta/<run>/

示例
----
# 快速冒烟（CPU）
python -m extract.roberta.train --limit 200 --epochs 1 --max-steps 5 --batch-size 4

# 纯 Softmax + 标签平滑 + FGM
python -m extract.roberta.train --label-smoothing 0.1 --adv fgm --monitor event_exact

# RoBERTa + CRF
python -m extract.roberta.train --crf --constrained --monitor event_exact
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.data import (  # noqa: E402
    extract_spans,
    load_records,
    make_chunks,
)
from common.label import (  # noqa: E402
    BIO_ID2LABEL,
    BIO_LABEL2ID,
    NUM_BIO_LABELS,
    OUTSIDE_LABEL,
    PAD_LABEL_ID,
    TRIGGER_LABEL,
    split_bio_label,
)
from common.logging_utils import setup_logger  # noqa: E402
from common.metrics import (  # noqa: E402
    compute_event_exact_match,
    compute_metrics,
    compute_norm_metrics,
    compute_type_metrics,
    spans_from_bio,
)
from common.schema import infer_event_type  # noqa: E402

try:
    from .loss import TokenClassificationLoss  # noqa: E402
    from .model import RobertaForEventExtraction  # noqa: E402
except ImportError:  # 直接 python extract/roberta/train.py 运行
    from loss import TokenClassificationLoss  # type: ignore  # noqa: E402
    from model import RobertaForEventExtraction  # type: ignore  # noqa: E402

DEFAULT_PRETRAIN = PROJECT_ROOT / "pretrain_models" / "chinese-roberta-wwm-ext"


# ---------------------------------------------------------------- 数据：子词对齐


def _aggregate_token_label(char_labels: list[int], s: int, e: int) -> int:
    """子词覆盖字符区间 [s, e) -> 该子词的字符级标签。

    BPE 可能把 "519005"、"2017" 这类串合成一个 token，覆盖多个字符。
    训练标签必须按"区间内是否落在实体里"聚合，而不是只取起点字符的标签，
    否则 token 后半段的字符没有任何监督，解码时 span 会被截断。

    规则（与 model._fold_to_char 的展开逻辑互为逆运算）：
    1. 区间内取**最靠左**的非 O 标签（B-X 保留 B，I-X 保留 I）；
    2. 全为 O 时返回 O；
    3. token 起点在 O、但区间内部已进入实体时，仍返回该实体标签，
       避免实体末尾被 token 边界切掉。
    """
    e = min(e, len(char_labels))
    for c in range(max(s, 0), e):
        lid = char_labels[c]
        if lid == PAD_LABEL_ID:
            continue
        if BIO_ID2LABEL[lid] != OUTSIDE_LABEL:
            return lid
    if 0 <= s < len(char_labels):
        return char_labels[s]
    return PAD_LABEL_ID


class RobertaChunkDataset(torch.utils.data.Dataset):
    """把记录切块后，用 RoBERTa fast tokenizer 做子词对齐。

    每个 example：
      input_ids / labels / attention_mask / content_mask : 子词级（pad 到 batch max）
      content_offsets : 内容子词的 (start,end) 字符 offset（用于解码折叠回字符）
      char_labels     : 字符级金标（评估用，长度 = char_len）
      source / offset / event_type / char_len : 元信息
    """

    def __init__(self, records, tok, max_len=512, stride=384):
        self.tok = tok
        self.max_len = max_len
        self.examples: list[dict] = []
        from common.data import chunk_to_label_ids

        for rec in records:
            text = rec.get("text") or ""
            spans = extract_spans(rec)
            source = str(rec.get("source", ""))
            event_type = (rec.get("event") or {}).get("type", "")
            for chunk in make_chunks(
                text, spans, max_len=max_len, stride=stride, source=source
            ):
                if not chunk.text:
                    continue
                char_labels = chunk_to_label_ids(chunk)  # 字符级 BIO
                enc = tok(
                    chunk.text,
                    truncation=True,
                    max_length=max_len,
                    return_offsets_mapping=True,
                    add_special_tokens=True,
                )
                offsets = enc["offset_mapping"]
                input_ids = enc["input_ids"]
                sub_labels: list[int] = []
                content_mask: list[bool] = []
                content_offsets: list[tuple[int, int]] = []
                seen_starts: set[int] = set()
                for s, e in offsets:
                    if s == 0 and e == 0:  # <[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]> / [SEP]
                        sub_labels.append(PAD_LABEL_ID)
                        content_mask.append(False)
                        continue
                    if s >= len(char_labels):  # 截断越界保护
                        sub_labels.append(PAD_LABEL_ID)
                        content_mask.append(False)
                        continue
                    cl = _aggregate_token_label(char_labels, s, e)
                    # 同一个字符被拆成多个子词时：续接子词若拿到 B-X 要提升为 I-X，
                    # 否则 BIO 会在该字符处断开成两个 span。
                    if s in seen_starts:
                        prefix, base = split_bio_label(BIO_ID2LABEL[cl])
                        if prefix == "B":
                            cl = BIO_LABEL2ID[f"I-{base}"]
                    else:
                        seen_starts.add(s)
                    sub_labels.append(cl)
                    content_mask.append(True)
                    content_offsets.append((int(s), int(e)))
                self.examples.append(
                    {
                        "input_ids": input_ids,
                        "labels": sub_labels,
                        "attention_mask": [True] * len(input_ids),
                        "content_mask": content_mask,
                        "content_offsets": content_offsets,
                        "char_labels": char_labels,
                        "char_len": len(char_labels),
                        "source": chunk.source,
                        "offset": chunk.offset,
                        "event_type": event_type,
                    }
                )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        return self.examples[idx]


def collate_fn(batch):
    B = len(batch)
    max_t = max(len(x["input_ids"]) for x in batch)
    pad_id = 0  # RoBERTa pad_token_id=0
    input_ids = torch.full((B, max_t), pad_id, dtype=torch.long)
    labels = torch.full((B, max_t), PAD_LABEL_ID, dtype=torch.long)
    attn = torch.zeros((B, max_t), dtype=torch.bool)
    cmask = torch.zeros((B, max_t), dtype=torch.bool)
    content_offsets, char_labels, char_lengths = [], [], []
    meta = []
    for i, x in enumerate(batch):
        L = len(x["input_ids"])
        input_ids[i, :L] = torch.tensor(x["input_ids"], dtype=torch.long)
        labels[i, :L] = torch.tensor(x["labels"], dtype=torch.long)
        attn[i, :L] = True
        cmask[i, :L] = torch.tensor(x["content_mask"], dtype=torch.bool)
        content_offsets.append(x["content_offsets"])
        char_labels.append(x["char_labels"])
        char_lengths.append(x["char_len"])
        meta.append(
            {
                "source": x["source"],
                "offset": x["offset"],
                "event_type": x["event_type"],
            }
        )
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attn,
        "content_mask": cmask,
        "content_offsets": content_offsets,
        "char_labels": char_labels,
        "char_lengths": char_lengths,
        "meta": meta,
    }


# ---------------------------------------------------------------- 训练技巧


class FGM:
    """FGM 对抗训练：扰动 word_embeddings。"""

    def __init__(self, model, emb_name: str = "word_embeddings", eps: float = 1.0):
        self.model = model
        self.emb_name = emb_name
        self.eps = eps
        self.backup: dict[str, torch.Tensor] = {}

    def attack(self) -> None:
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name and param.grad is not None:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    param.data.add_(self.eps * param.grad / norm)

    def restore(self) -> None:
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


def build_layerwise_param_groups(
    model: RobertaForEventExtraction,
    base_lr: float,
    head_lr_mult: float,
    layer_lr_decay: float,
    weight_decay: float,
) -> list[dict]:
    """分层学习率 + weight decay（排除 bias / LayerNorm）。

    encoder 第 i 层（0=底）lr = base_lr * decay^(num_layers-1-i)；
    embedding 与最底层同 lr；分类头 lr = base_lr * head_lr_mult。
    decay=1.0 时 encoder 全部同 lr（退化为 classify 的 2 组模式）。
    """
    no_decay_substrings = ("bias", "LayerNorm", "layer_norm")
    groups: list[dict] = []

    encoder = model.encoder
    num_layers = encoder.config.num_hidden_layers

    def add_params(params, lr, decay):
        decay_params, nodecay_params = [], []
        for n, p in params:
            if any(s in n for s in no_decay_substrings):
                nodecay_params.append(p)
            else:
                decay_params.append(p)
        if decay_params:
            groups.append({"params": decay_params, "lr": lr, "weight_decay": decay})
        if nodecay_params:
            groups.append({"params": nodecay_params, "lr": lr, "weight_decay": 0.0})

    # embedding（最底层 lr）
    emb_lr = base_lr * (layer_lr_decay ** (num_layers - 1))
    add_params(
        [(n, p) for n, p in encoder.embeddings.named_parameters() if p.requires_grad],
        emb_lr, weight_decay,
    )
    # 每层
    layers = encoder.encoder.layer
    for i, layer in enumerate(layers):
        lr_i = base_lr * (layer_lr_decay ** (num_layers - 1 - i))
        add_params(
            [(n, p) for n, p in layer.named_parameters() if p.requires_grad],
            lr_i, weight_decay,
        )
    # 分类头
    add_params(
        [(n, p) for n, p in model.classifier.named_parameters() if p.requires_grad],
        base_lr * head_lr_mult, weight_decay,
    )
    # CRF 转移参数（若有）走 head 档
    if model.use_crf:
        add_params(
            [(n, p) for n, p in model.crf.named_parameters() if p.requires_grad],
            base_lr * head_lr_mult, weight_decay,
        )
    return groups


# ---------------------------------------------------------------- 评估（自包含，复用 common.metrics）


@torch.no_grad()
def evaluate(model, loader, device, criterion, doc_texts: dict[str, str] | None = None):
    """验证集：字符级 gold/pred 按文档聚合后调用 common.metrics。

    doc_texts: {source: 原文}。传入后额外计算"空白归一化"口径（arg_norm），
    用于排除 "2017 年 6 月 30 日" 与 "2017年6月30日" 这类写法差异的干扰。
    """
    model.eval()
    total_loss, total_steps = 0.0, 0
    gold_by_doc: dict[str, set] = defaultdict(set)
    pred_by_doc: dict[str, set] = defaultdict(set)
    gold_type_by_doc: dict[str, str] = {}

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        cmask = batch["content_mask"].to(device)
        loss, pred_ids = model(
            input_ids, attn,
            labels=labels, content_mask=cmask,
            content_offsets=batch["content_offsets"],
            char_lengths=batch["char_lengths"],
            criterion=criterion,
        )
        if loss is not None:
            total_loss += float(loss.item())
            total_steps += 1

        for b, meta in enumerate(batch["meta"]):
            src = meta["source"]
            off = int(meta["offset"])
            gold_type_by_doc.setdefault(src, meta.get("event_type", ""))
            for s, e, base in spans_from_bio(batch["char_labels"][b]):
                gold_by_doc[src].add((s + off, e + off, base))
            for s, e, base in spans_from_bio(pred_ids[b]):
                pred_by_doc[src].add((s + off, e + off, base))

    docs = sorted(set(gold_by_doc) | set(pred_by_doc))
    gold_list = [sorted(gold_by_doc[d]) for d in docs]
    pred_list = [sorted(pred_by_doc[d]) for d in docs]
    report = compute_metrics(gold_list, pred_list)
    if doc_texts is not None:
        norm = compute_norm_metrics(
            gold_list, pred_list, [doc_texts.get(d, "") for d in docs]
        )
        report.overall_norm, report.arg_norm = norm.overall, norm.arg

    gold_types = [gold_type_by_doc.get(d, "") for d in docs]
    pred_types = [infer_event_type({base for _, _, base in pred_by_doc[d]}) for d in docs]
    report.event_type = compute_type_metrics(gold_types, pred_types)

    def _to_event(spans, ev_type):
        trigger = None
        args = set()
        for s, e, base in sorted(spans):
            if base == TRIGGER_LABEL:
                if trigger is None:
                    trigger = (s, e, base)
            else:
                args.add((s, e, base))
        return {"type": ev_type, "trigger": trigger, "args": frozenset(args)}

    gold_events = [_to_event(gold_by_doc[d], gold_type_by_doc.get(d)) for d in docs]
    pred_events = [_to_event(pred_by_doc[d], pred_types[i]) for i, d in enumerate(docs)]
    report.event_match = compute_event_exact_match(gold_events, pred_events)
    return total_loss / max(total_steps, 1), report


def score_of(report, monitor: str) -> float:
    if monitor == "f1":
        return report.f1
    d = report.as_dict()
    if monitor == "event_exact":
        return (d.get("event_match") or {}).get("accuracy", 0.0)
    if monitor == "type_macro_f1":
        return (d.get("event_type") or {}).get("macro_f1", 0.0)
    raise ValueError(f"未知 monitor: {monitor}")


# ---------------------------------------------------------------- 主流程


def parse_args():
    p = argparse.ArgumentParser(description="RoBERTa 事件抽取训练")
    # 数据
    p.add_argument("--train-file", default=str(PROJECT_ROOT / "data/dataset/split/train.jsonl"))
    p.add_argument("--val-file", default=str(PROJECT_ROOT / "data/dataset/split/val.jsonl"))
    p.add_argument("--pretrain-dir", default=str(DEFAULT_PRETRAIN))
    p.add_argument("--max-len", type=int, default=512, help="滑窗字符数 / tokenizer max_length")
    p.add_argument("--stride", type=int, default=384)
    p.add_argument("--limit", type=int, default=0, help="每集只取前 N 条（调试）")

    # 模型
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--crf", action="store_true", help="启用 CRF 头（默认纯 Softmax）")
    p.add_argument("--constrained", action="store_true", help="CRF 解码期 BIO 硬约束")

    # 损失 / 类别不平衡（weight 单一真源）
    p.add_argument("--class-weight", default="none", choices=["none", "inverse", "balanced"])
    p.add_argument("--prior-scale", type=float, default=1.0, help="inverse 权重缩放")
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--focal-gamma", type=float, default=0.0)

    # 优化 / 学习率
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum-steps", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-5, help="encoder 顶层学习率")
    p.add_argument("--head-lr-mult", type=float, default=5.0)
    p.add_argument("--layer-lr-decay", type=float, default=0.95, help="逐层衰减（1.0=不衰减）")
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--warmup-steps", type=int, default=0, help=">0 时覆盖 warmup-ratio")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--monitor", default="event_exact",
                   choices=["f1", "event_exact", "type_macro_f1"])
    p.add_argument("--max-steps", type=int, default=0, help="每 epoch 最多 N 步（调试）")

    # 对抗 / 精度
    p.add_argument("--adv", default="none", choices=["none", "fgm"])
    p.add_argument("--fgm-eps", type=float, default=1.0)
    p.add_argument("--amp", action="store_true", default=True, help="混合精度（默认开）")
    p.add_argument("--no-amp", dest="amp", action="store_false")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--output-root", default=str(PROJECT_ROOT / "extract/output/roberta"))
    p.add_argument("--run-name", default="")
    p.add_argument("--log-every", type=int, default=20)
    return p.parse_args()


def resolve_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def set_seed(seed):
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_run_name(args, ts):
    if args.run_name:
        return args.run_name
    head = "crf" if args.crf else "softmax"
    loss = "focal" if args.focal_gamma > 0 else ("ls" if args.label_smoothing > 0 else "ce")
    cw = args.class_weight
    adv = args.adv
    mon = {"event_exact": "m-exact", "type_macro_f1": "m-macro", "f1": "m-f1"}[args.monitor]
    return f"{ts}_{head}_{loss}_cw-{cw}_adv-{adv}_{mon}"


EXPERIMENT_FIELDS = [
    "run_name", "start_time", "head", "loss", "class_weight", "label_smoothing",
    "focal_gamma", "adv", "lr", "layer_lr_decay", "head_lr_mult", "weight_decay",
    "batch_size", "grad_accum", "best_epoch", "best_val_f1", "event_exact",
]


def append_experiment_row(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=EXPERIMENT_FIELDS)
        if write_header:
            w.writeheader()
        w.writerow(row)


def compute_class_weight(option, train_ds, device, prior_scale):
    if option == "none":
        return None
    counts = [0] * NUM_BIO_LABELS
    for ex in train_ds.examples:
        for l in ex["char_labels"]:
            if 0 <= l < NUM_BIO_LABELS:
                counts[l] += 1
    total = max(sum(counts), 1)
    if option == "inverse":
        inv = [total / (c + 1) for c in counts]
        mean_inv = sum(inv) / len(inv)
        w = [min(10.0, max(0.1, (x / mean_inv))) * prior_scale for x in inv]
    else:  # balanced
        n_cls = NUM_BIO_LABELS
        w = [total / (n_cls * (c + 1)) for c in counts]
        m = sum(w) / len(w)
        w = [min(10.0, max(0.1, x / m)) for x in w]
    return torch.tensor(w, dtype=torch.float, device=device)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    is_cuda = device.type == "cuda"

    start_time = time.strftime("%Y-%m-%d %H:%M:%S")
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_name = make_run_name(args, ts)
    run_dir = Path(args.output_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(run_dir / f"roberta_{run_name}.log")

    (run_dir / "config.json").write_text(
        json.dumps({**vars(args), "run_name": run_name, "start_time": start_time},
                   ensure_ascii=False, indent=2), encoding="utf-8")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.pretrain_dir, use_fast=True)

    logger.info("=" * 70)
    logger.info("RoBERTa 事件抽取训练")
    logger.info("run 名称: %s", run_name)
    logger.info("输出目录: %s", run_dir)
    logger.info("设备: %s | amp=%s", device, args.amp)
    logger.info("参数: %s", json.dumps(vars(args), ensure_ascii=False))

    # ---------- 数据 ----------
    logger.info("加载数据 ...")
    train_records = load_records(args.train_file, limit=args.limit)
    val_records = load_records(args.val_file, limit=args.limit)
    logger.info("训练集 %d 条，验证集 %d 条", len(train_records), len(val_records))

    train_ds = RobertaChunkDataset(train_records, tok, args.max_len, args.stride)
    val_ds = RobertaChunkDataset(val_records, tok, args.max_len, args.stride)
    logger.info("切块后：训练 %d 块，验证 %d 块", len(train_ds), len(val_ds))

    g = torch.Generator(); g.manual_seed(args.seed)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0,
        collate_fn=collate_fn, generator=g,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0,
        collate_fn=collate_fn,
    )
    val_texts = {str(r.get("source", "")): r.get("text") or "" for r in val_records}

    # ---------- 类别权重（单一真源：CE/LS/Focal 共用；CRF 路径不使用） ----------
    class_weight = compute_class_weight(args.class_weight, train_ds, device, args.prior_scale)
    if class_weight is not None:
        top = sorted(zip([BIO_ID2LABEL[i] for i in range(NUM_BIO_LABELS)],
                         class_weight.tolist()), key=lambda kv: kv[1], reverse=True)[:5]
        logger.info("类别权重(%s): 最高 5 个: %s", args.class_weight,
                    json.dumps([(k, round(v, 3)) for k, v in top], ensure_ascii=False))

    # ---------- 模型 + 损失 ----------
    model = RobertaForEventExtraction(
        args.pretrain_dir, num_tags=NUM_BIO_LABELS, dropout=args.dropout,
        use_crf=args.crf, constrained=args.constrained,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("模型可训练参数: %d", n_params)

    criterion = None
    if not args.crf:
        criterion = TokenClassificationLoss(
            NUM_BIO_LABELS, weight=class_weight,
            label_smoothing=args.label_smoothing, focal_gamma=args.focal_gamma,
            ignore_index=PAD_LABEL_ID,
        ).to(device)
        if args.focal_gamma > 0 and class_weight is None and args.class_weight != "none":
            logger.warning("class_weight=%s 但权重计算为 None，Focal 将不带 per-class 加权",
                           args.class_weight)
    else:
        if args.label_smoothing > 0 or args.focal_gamma > 0:
            logger.warning("CRF 路径下 label_smoothing / focal_gamma 不生效（CRF 使用 NLL）")
        if class_weight is not None:
            logger.warning("CRF 路径下 per-class 权重不生效（如需类别加权请用 inverse prior 偏置，本版未接入）")
    loss_desc = ("CRF" if args.crf else
                 ("Focal" if args.focal_gamma > 0 else
                  ("CE+LS" if args.label_smoothing > 0 else "CE")))
    logger.info("损失: %s | 类别权重: %s", loss_desc,
                "on" if class_weight is not None else "off")

    # ---------- 优化器（分层 lr + wd） ----------
    param_groups = build_layerwise_param_groups(
        model, args.lr, args.head_lr_mult, args.layer_lr_decay, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups)
    logger.info("优化器: AdamW，参数组 %d 个 | base_lr=%.2e decay=%.3f head_mult=%.1f wd=%.2e",
                len(param_groups), args.lr, args.layer_lr_decay,
                args.head_lr_mult, args.weight_decay)

    # ---------- 调度器：warmup + 线性衰减 ----------
    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum_steps)
    total_updates = steps_per_epoch * args.epochs
    warmup = args.warmup_steps if args.warmup_steps > 0 else int(total_updates * args.warmup_ratio)
    warmup = max(0, min(warmup, total_updates))

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        return max(0.0, (total_updates - step) / max(1, total_updates - warmup))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    logger.info("调度器: warmup %d / 总 %d updates（线性衰减）", warmup, total_updates)

    # ---------- 对抗 / AMP ----------
    fgm = FGM(model, eps=args.fgm_eps) if args.adv == "fgm" else None
    amp_dtype = torch.float16 if is_cuda else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda", enabled=(is_cuda and args.amp))
    logger.info("对抗训练: %s%s", args.adv, f" (eps={args.fgm_eps})" if fgm else "")
    logger.info("混合精度: %s", f"{'fp16' if is_cuda else 'bf16'}+scaler" if args.amp else "off")

    # ---------- 训练循环 ----------
    best_path = run_dir / "roberta_best.pt"
    best_score, best_f1, best_epoch = -1.0, -1.0, -1
    no_improve = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running, n_micro, n_update = 0.0, 0, 0
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader, start=1):
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            cmask = batch["content_mask"].to(device)

            def forward_loss():
                with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=args.amp):
                    loss, _ = model(
                        input_ids, attn, labels=labels, content_mask=cmask,
                        content_offsets=batch["content_offsets"],
                        char_lengths=batch["char_lengths"], criterion=criterion,
                    )
                return loss / args.grad_accum_steps

            loss = forward_loss()
            if is_cuda and args.amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            if fgm is not None:
                fgm.attack()
                loss_adv = forward_loss()
                if is_cuda and args.amp:
                    scaler.scale(loss_adv).backward()
                else:
                    loss_adv.backward()
                fgm.restore()

            running += float(loss.item()) * args.grad_accum_steps
            n_micro += 1

            if step % args.grad_accum_steps == 0 or step == len(train_loader):
                if is_cuda and args.amp:
                    scaler.unscale_(optimizer)
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                if is_cuda and args.amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                n_update += 1

            if n_micro % args.log_every == 0:
                lr = optimizer.param_groups[0]["lr"]
                logger.info("    step %d | loss=%.4f | lr=%.2e | %.1fs",
                            n_micro, running / n_micro, lr, time.time() - t0)
            if args.max_steps and n_micro >= args.max_steps:
                break

        train_loss = running / max(n_micro, 1)
        val_loss, report = evaluate(model, val_loader, device, criterion, doc_texts=val_texts)
        score = score_of(report, args.monitor)
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_loss": val_loss, "val_f1": report.f1,
                        "monitor_score": score})
        logger.info(
            "Epoch %d/%d | train_loss=%.4f | val_loss=%.4f | %.1fs | %s",
            epoch, args.epochs, train_loss, val_loss, time.time() - t0, report.summary(),
        )
        d = report.as_dict()
        logger.info("    槽位 F1: %s", json.dumps(d["per_type_f1"], ensure_ascii=False))
        et = d.get("event_type")
        if et:
            logger.info("    事件类型: acc=%.4f macroF1=%.4f microF1=%.4f | %s",
                        et["accuracy"], et["macro_f1"], et["micro_f1"],
                        json.dumps(et["per_class_f1"], ensure_ascii=False))
        em = d.get("event_match")
        if em:
            logger.info("    事件级完全匹配: %s", json.dumps(em, ensure_ascii=False))

        if score > best_score:
            best_score, best_f1, best_epoch = score, report.f1, epoch
            no_improve = 0
            torch.save({
                "model_state": model.state_dict(),
                "model_config": model.get_config(),
                "best_f1": best_f1, "best_score": best_score,
                "monitor": args.monitor, "best_epoch": best_epoch,
                "args": vars(args),
            }, best_path)
            logger.info("    ★ 新最优 epoch=%d, %s=%.4f (val_F1=%.4f) -> %s",
                        epoch, args.monitor, score, report.f1, best_path.name)
        else:
            no_improve += 1
            logger.info("    %s 未提升（%d/%d），最优 epoch=%d %s=%.4f",
                        args.monitor, no_improve, args.patience,
                        best_epoch, args.monitor, best_score)
            if no_improve >= args.patience:
                logger.info("    触发 early stopping。")
                break

    # ---------- 最终复评 + 落盘 ----------
    logger.info("=" * 70)
    logger.info("训练完成：最优 epoch=%d，%s=%.4f，val_F1=%.4f",
                best_epoch, args.monitor, best_score, best_f1)
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    final_loss, final_report = evaluate(model, val_loader, device, criterion)
    fd = final_report.as_dict()
    logger.info("最终验证: val_loss=%.4f | %s", final_loss, final_report.summary())
    (run_dir / "val_report.json").write_text(
        json.dumps({
            "args": vars(args), "best_epoch": best_epoch, "best_val_f1": best_f1,
            "val_report": fd, "history": history,
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    em = fd.get("event_match")
    append_experiment_row(Path(args.output_root) / "experiments.csv", {
        "run_name": run_name, "start_time": start_time,
        "head": "crf" if args.crf else "softmax",
        "loss": loss_desc, "class_weight": args.class_weight,
        "label_smoothing": args.label_smoothing, "focal_gamma": args.focal_gamma,
        "adv": args.adv, "lr": args.lr, "layer_lr_decay": args.layer_lr_decay,
        "head_lr_mult": args.head_lr_mult, "weight_decay": args.weight_decay,
        "batch_size": args.batch_size, "grad_accum": args.grad_accum_steps,
        "best_epoch": best_epoch, "best_val_f1": round(best_f1, 4),
        "event_exact": round(em["accuracy"], 4) if em else "",
    })
    logger.info("验证报告: %s", run_dir / "val_report.json")
    logger.info("实验汇总: %s", Path(args.output_root) / "experiments.csv")
    logger.info("最优模型: %s", best_path)


if __name__ == "__main__":
    main()
