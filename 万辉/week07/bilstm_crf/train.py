"""
L1 基线：字符级 BiLSTM-CRF（不用任何预训练模型，从零训）

为什么课里要放这个"弱模型"：
  它是 BERT 的**同任务对照**。同样的标签体系（149 个）、同样的 CRF、同样的解码与评测，
  唯一差别是「特征编码器」：BiLSTM 从零学字向量，BERT 带着上亿语料预训练好的知识。
  两者一减，就是「预训练到底值多少个点」——这正是学生简历里最想写的那种对比数字。

结构：字向量(128) → BiLSTM(hidden 256×2=512) → Linear(149) → CRF
  - embedding 128：字表约 4000 字，128 维足够，再大在小数据上只会过拟合
  - BiLSTM 256：双向各 256，拼起来 512；单向 LSTM 拿不到右侧上下文，
    而事件抽取里「触发词」常常要靠后面的词才能确定（如「回购**股份**」）
  - CRF：和 BERT+CRF 共用 crf.py 里手写的那一份，保证公平

和 BERT 版本的两个工程差异（课上会说）：
  1. 没有 tokenizer / 子词，一个汉字直接查字表 → 不需要 word_ids 对齐（简单，但没有词表先验）
  2. 序列必须**补齐到 batch 内最长**（LSTM 要求定长张量），所以用 collate_char + mask

使用方式：
  python bilstm_crf/train.py                    # 默认 5 epoch
  python bilstm_crf/train.py --epochs 3 --max_len 256
  python bilstm_crf/train.py --decoder linear   # 消融：去掉 CRF（对照非法 BIO 数）
  python bilstm_crf/train.py --mock             # 20 条手造数据跑通流程

产物：outputs/checkpoints/bilstm_crf.pt、outputs/metrics/disc/bilstm_crf.json 等
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]   # 项目根目录（方法目录的上一级）
sys.path.insert(0, str(ROOT / "common"))     # 公共模块：event_data / crf / eval_utils / models / log_utils

from crf import CRFClassifier
from eval_utils import CKPT_DIR, LOG_DIR, Timer, evaluate_predictions
from log_utils import TrainLogger, peak_gpu_mb
from event_data import (CharDataset, DATA_DIR, ROOT, build_joint_schema,

                        build_bio_constraint_mask, build_start_constraint_mask,
                        collate_char, decode_bio_spans, illegal_bio_stats, load_enums,
                        load_extract_split, load_schema, make_pred_row, make_vocab,
                        row_event, spans_to_event)

MOCK_DIR = ROOT / "data" / "_mock"

class BiLSTMEncoder(nn.Module):
    """
    字符级 BiLSTM 编码器（输出 (B, L, 512)），接口和 train_bio.py 的 BertEncoder 一致。

    接口一致的好处：CRFClassifier / CRF / 解码 / 指标统计全都能复用，
    「BiLSTM vs BERT」的差别就被严格限制在编码器这一个模块里 —— 对比才公平。
    """

    def __init__(self, vocab_size: int, emb_dim: int = 128, hidden: int = 256,
                 num_layers: int = 1, dropout: float = 0.3):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(emb_dim, hidden, num_layers=num_layers, batch_first=True,
                            bidirectional=True,
                            # nn.LSTM 的 dropout 只在 num_layers>1 时生效，否则会告警
                            dropout=dropout if num_layers > 1 else 0.0)
        self.hidden_size = hidden * 2

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        emb = self.dropout(self.emb(input_ids))
        out, _ = self.lstm(emb)
        return out

def build_vocab_and_datasets(data_dir: Path, args):
    """建字表 + 切内层验证集 + 造三个 Dataset（训练/内层验证/评测 dev）。"""
    train_rows = load_extract_split("train", data_dir)
    dev_rows = load_extract_split("dev", data_dir)
    label_scheme = build_joint_schema(load_schema(data_dir / "event_schema.json"))

    # 字表只从「训练集」建：用 dev 建字表等于提前看了测试数据（数据泄漏）
    vocab = make_vocab(train_rows)
    print(f"字表大小：{len(vocab)}（训练集字符去重；<PAD>=0, <UNK>=1）")

    rng = random.Random(args.seed)
    idxs = list(range(len(train_rows)))
    rng.shuffle(idxs)
    n_inner = max(1, int(len(train_rows) * args.inner_val))
    inner_idx = set(idxs[:n_inner])
    fit_rows = [r for i, r in enumerate(train_rows) if i not in inner_idx]
    val_rows = [r for i, r in enumerate(train_rows) if i in inner_idx]

    fit_ds = CharDataset(fit_rows, vocab, label_scheme, args.max_len)
    val_ds = CharDataset(val_rows, vocab, label_scheme, args.max_len)
    dev_ds = CharDataset(dev_rows, vocab, label_scheme, args.max_len)
    print(f"训练 {len(fit_ds)} 句 / 内层验证 {len(val_ds)} 句 / 评测 dev {len(dev_ds)} 句")
    return vocab, label_scheme, fit_ds, val_ds, dev_ds, dev_rows

def predict_char_tags(model, loader, id2label: dict, device) -> list[list[str]]:
    """推理：把路径还原成字符级标签序列（BiLSTM 没有子词，直接一一对应）。"""
    model.eval()
    all_tags = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            paths = model.predict(input_ids, mask)
            for b, path in enumerate(paths):
                all_tags.append([id2label.get(t, "O") for t in path])
    return all_tags

def span_f1(rows: list[dict], char_tags: list[list[str]]) -> float:
    """训练中做模型选择的轻量指标（角色+文本精确匹配的 F1）。"""
    tp = fp = fn = 0
    for row, tags in zip(rows, char_tags):
        ge = row_event(row) or {}
        gold = {(a["role"], a["text"]) for a in ge.get("args") or []}
        gt = (ge.get("trigger") or {}).get("text", "")
        if gt:
            gold.add(("__trig__", gt))
        ev = spans_to_event(decode_bio_spans(tags), row["text"])
        pred = {(a["role"], a["text"]) for a in (ev or {}).get("args", [])}
        if (ev or {}).get("trigger", {}).get("text"):
            pred.add(("__trig__", ev["trigger"]["text"]))
        tp += len(gold & pred)
        fp += len(pred - gold)
        fn += len(gold - pred)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0

def eval_loader(model, loader, id2label, device) -> float:
    """算平均 loss + span F1（只做模型选择，不产出对外指标）。"""
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            inp = batch["input_ids"].to(device)
            lab = batch["labels"].to(device)
            msk = batch["attention_mask"].to(device)
            _, loss = model(inp, msk, lab)
            total += float(loss.item())
            n += 1
    tags = predict_char_tags(model, loader, id2label, device)
    return total / max(n, 1), span_f1(loader.dataset.rows, tags)

def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"设备：{device}")

    data_dir = (Path(args.data_dir) if args.data_dir
                else (MOCK_DIR if args.mock else DATA_DIR))
    enums = load_enums(data_dir / "event_schema.json")
    vocab, label_scheme, fit_ds, val_ds, dev_ds, dev_rows = build_vocab_and_datasets(
        data_dir, args)
    id2label = label_scheme["id2label"]
    labels = label_scheme["labels"]

    fit_loader = DataLoader(fit_ds, batch_size=args.batch_size, shuffle=True,
                            num_workers=0, collate_fn=collate_char)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, collate_fn=collate_char)
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, collate_fn=collate_char)

    encoder = BiLSTMEncoder(len(vocab), emb_dim=args.emb_dim, hidden=args.hidden,
                            num_layers=args.num_layers, dropout=args.dropout)
    constraint = start_c = None
    if args.hard_constraint:
        constraint = build_bio_constraint_mask(label_scheme)
        start_c = build_start_constraint_mask(label_scheme)
    model = CRFClassifier(encoder=encoder, hidden_size=encoder.hidden_size,
                          num_tags=len(labels), dropout=args.dropout,
                          use_crf=(args.decoder == "crf"),
                          constraint=constraint, start_constraint=start_c).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数：{n_params / 1e6:.2f}M（从零训练，无预训练），"
          f"解码器={args.decoder}")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1.0, (step + 1) / max(1, args.warmup_steps)))

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CKPT_DIR / f"{args.run_name}.pt"

    logger = TrainLogger(method=args.run_name, args=args, root=ROOT, params=n_params,
                         notes="BiLSTM-CRF（字符级，无预训练）训练过程")

    log, best_f1, best_epoch = [], -1.0, 0
    train_start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.perf_counter()
        total, n = 0.0, 0
        for step, batch in enumerate(fit_loader):
            inp = batch["input_ids"].to(device)
            lab = batch["labels"].to(device)
            msk = batch["attention_mask"].to(device)
            _, loss = model(inp, msk, lab)
            optimizer.zero_grad()
            loss.backward()
            # 梯度裁剪：BiLSTM + CRF 在长序列上前几步梯度很猛，不裁剪会偶发 NaN
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            logger.log_step(epoch=epoch, step=step, loss=float(loss.item()),
                            lr=scheduler.get_last_lr()[0])
            total += float(loss.item())
            n += 1
        train_loss = total / max(n, 1)
        val_loss, val_f1 = eval_loader(model, val_loader, id2label, device)
        elapsed = time.perf_counter() - t0
        print(f"epoch {epoch}/{args.epochs} | train_loss={train_loss:.4f} | "
              f"inner_loss={val_loss:.4f} inner_spanF1={val_f1:.4f} | {elapsed:.0f}s",
              flush=True)
        logger.log_epoch(epoch=epoch, train_loss=train_loss, elapsed=elapsed,
                         val_metrics={"inner_loss": val_loss, "inner_span_f1": val_f1},
                         extra={"best_so_far": round(best_f1, 6)})
        log.append({"epoch": epoch, "train_loss": round(train_loss, 6),
                    "inner_val_loss": round(val_loss, 6),
                    "inner_val_span_f1": round(val_f1, 6), "elapsed_s": round(elapsed, 1)})
        if val_f1 > best_f1:
            best_f1, best_epoch = val_f1, epoch
            torch.save({"state_dict": model.state_dict(), "vocab": vocab,
                        "labels": labels, "args": vars(args), "epoch": epoch,
                        "inner_val_span_f1": val_f1}, ckpt_path)
            print(f"  ★ 内层验证最优 → 保存 {ckpt_path.name}", flush=True)

    train_s = time.perf_counter() - train_start
    with open(LOG_DIR / f"train_{args.run_name}.json", "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    with Timer() as timer:
        dev_tags = predict_char_tags(model, dev_loader, id2label, device)
    dev_illegal = illegal_bio_stats(dev_tags)

    preds, n_empty = [], 0
    for row, tags in zip(dev_rows, dev_tags):
        ev = spans_to_event(decode_bio_spans(tags), row["text"], enums=enums)
        if ev is None:
            n_empty += 1
        preds.append(make_pred_row(row, [ev] if ev else []))
    lat = timer.summary(len(dev_rows))

    meta = {
        "display_name": f"BiLSTM-CRF（{'CRF' if args.decoder == 'crf' else 'Linear'}）",
        "train_s": round(train_s, 1),
        "params_m": round(n_params / 1e6, 2),
        "ms_per_sentence": lat["ms_per_sentence"],
        "peak_gpu_mb": lat["peak_gpu_mb"],
        "decoder": args.decoder, "epochs": args.epochs, "max_len": args.max_len,
        "emb_dim": args.emb_dim, "hidden": args.hidden, "lr": args.lr,
        "best_epoch": best_epoch, "inner_val_span_f1": round(best_f1, 4),
        "num_labels": len(labels), "vocab_size": len(vocab),
        "illegal_dev": dev_illegal,
        "n_truncated": dev_ds.n_truncated,
        "pretrained": False,
        "seed": args.seed,
    }
    res = evaluate_predictions(args.run_name, preds, meta=meta,
                              gold_path=data_dir / "dev_extract.jsonl")

    print("\n【非法 BIO 统计（dev，模型原始输出）】")
    print(f"  序列数 {dev_illegal['total_seqs']}：含非法序列 {dev_illegal['illegal_seqs']} 条"
          f"（{dev_illegal['illegal_seq_rate']:.1%}），"
          f"非法开头 {dev_illegal['illegal_start']} 处 / "
          f"非法转移 {dev_illegal['illegal_transition']} 处")
    print(f"\n训练耗时 {train_s:.0f}s | 延迟 {lat['ms_per_sentence']} ms/句")

    # 训练日志收尾：把「最好的 epoch / 验证指标 / dev 三级 F1 / 峰值显存」一起写进 _train.json，
    # 这样学生只看日志就能复盘「训练过程 → 最终效果」的因果，而不用重跑。
    logger.finalize(
        total_s=train_s,
        best_epoch=best_epoch,
        best_inner_val_span_f1=round(best_f1, 6),
        dev_metrics={k: res[k]["f1"] for k in ("trigger", "argument", "event_exact")},
        inference_latency_ms=lat["ms_per_sentence"],
        peak_gpu_mb=peak_gpu_mb(),
        illegal_dev=dev_illegal,
    )

def parse_args():
    ap = argparse.ArgumentParser(description="字符级 BiLSTM-CRF 事件抽取基线")
    ap.add_argument("--decoder", choices=["crf", "linear"], default="crf")
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--emb_dim", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--num_layers", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--max_grad_norm", type=float, default=5.0)
    ap.add_argument("--warmup_steps", type=int, default=100)
    ap.add_argument("--inner_val", type=float, default=0.1)
    ap.add_argument("--hard_constraint", action="store_true")
    ap.add_argument("--device", default="", help="cuda / cpu；留空自动选（mock 冒烟建议 cpu，别抢卡）")
    ap.add_argument("--log_every", type=int, default=50,
                    help="每多少步往 steps.jsonl 写一行（供学生画 loss 曲线）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--run_name", default="")
    ap.add_argument("--mock", action="store_true", help="用 data/_mock 的 20 条手造数据（离线冒烟）")
    ap.add_argument("--data_dir", default="", help="数据目录；留空=全量 data/processed（--mock 时用 data/_mock）")
    args = ap.parse_args()
    if not args.run_name:
        args.run_name = f"bilstm_{args.decoder}" + ("_mock" if args.mock else "")
    return args

if __name__ == "__main__":
    main()
