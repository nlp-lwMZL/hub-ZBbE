"""
L2 两段式流水线（pipeline）事件抽取：检测器 → 类型分类 → 类型条件论元抽取

这是本周的**核心教学对比**：同样是 BERT，一次联合抽取（train_bio.py，149 标签）
vs 拆成三段各管一件事，哪个更高？

三段的分工：
  (a) 事件检测器     句子里到底有没有事件？（二分类，负样本占 ~78%，这是真实公告的分布）
  (b) 事件类型 13 分类 有事件的话是 13 类里的哪一类？
  (c) 类型条件论元抽取 已知类型 t，抽「触发词 + 该类型的角色」
                      → 标签空间从 149 缩到 11~19 个（每个类型一个专家模型）

为什么流水线常常更高（本次实测见 NOTES）：
  - (c) 的标签空间小一个数量级，每个标签的样本密度高，论元边界学得准
  - 联合模型的 149 个标签里，长尾角色只有几十个样本，基本学不动
为什么流水线也常常更低（课上要讲的反面）：
  - 误差会**累积**：(b) 类型错 → (c) 一定全错；而且三段各跑一次，延迟是 3 倍
  - 训练要维护 15 个模型（1 检测 + 1 类型 + 13 专家），工程复杂度高
所以本项目把两个数字都实测出来，让学生看到「不是越拆越好」。

这一版刻意用**线性头**而不是 CRF：(c) 段每类标签只有十几个，非法序列问题本来就小，
加 CRF 的收益不抵训练/推理成本 —— 这个取舍也写进 NOTES。

使用方式：
  python bert_pipeline/train.py                        # 完整训练 + 组合评测（约 15 分钟）
  python bert_pipeline/train.py --arg_epochs 2         # 少训一点
  python bert_pipeline/train.py --skip_train           # 复用已有 checkpoint，只重新组合评测
  python bert_pipeline/train.py --mock                 # 20 条手造数据跑通流程

产物：
  outputs/checkpoints/pipe_detector.pt / pipe_type.pt / pipe_arg_<类型>.pt
  outputs/metrics/disc/pipe_detector.json      检测器 P/R/F1（dev_detect 全量 + clean）
  outputs/metrics/disc/pipe_type.json          13 分类准确率
  outputs/metrics/disc/pipeline.json           端到端（dev_extract 930）
  outputs/metrics/disc/pipeline_oracle_type.json  类型给对时的上界（定位误差来源）
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModel, BertTokenizer, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[1]   # 项目根目录（方法目录的上一级）
sys.path.insert(0, str(ROOT / "common"))     # 公共模块：event_data / crf / eval_utils / models / log_utils

from log_utils import TrainLogger, peak_gpu_mb
from eval_utils import (CKPT_DIR, LOG_DIR, METRIC_DIR, Timer, evaluate_predictions,
                        evaluate_nested_subset)
from event_data import (BioDataset, ClsDataset, DATA_DIR, ROOT, build_typed_schema,
                        collate_bio, collate_cls, decode_bio_spans, illegal_bio_stats,
                        load_detect_split, load_enums, load_extract_split, load_jsonl,
                        load_schema, make_pred_row, spans_to_event, union_roles)
from model_paths import BERT_MODEL_DIR, resolve_model_path

from models import BertCls, _Layers, _layers_of, build_arg_model

MOCK_DIR = ROOT / "data" / "_mock"

# ─────────────────────────── BERT 分类模型 ───────────────────────────

def train_classifier(rows, labels_of, num_labels, tokenizer, args, device,
                     ckpt_name: str, task: str, logger=None) -> tuple[BertCls, dict]:
    """通用的 BERT 分类训练（检测器 / 类型分类共用，避免两套几乎一样的训练代码）。"""
    ds = ClsDataset(rows, tokenizer, args.cls_max_len, task=task,
                    type2id=labels_of if isinstance(labels_of, dict) else None)
    loader = DataLoader(ds, batch_size=args.cls_batch_size, shuffle=True,
                        num_workers=0, collate_fn=collate_cls)
    model = BertCls(str(args.bert_path), num_labels, args.dropout).to(device)
    opt = AdamW(model.parameters(), lr=args.cls_lr, weight_decay=0.01)
    total = max(1, len(loader) * args.cls_epochs)
    sched = get_linear_schedule_with_warmup(opt, int(total * 0.1), total)

    t0 = time.perf_counter()
    for epoch in range(1, args.cls_epochs + 1):
        model.train()
        tot, n = 0.0, 0
        for step, batch in enumerate(loader):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            tti = batch["token_type_ids"].to(device)
            lab = batch["labels"].to(device)
            _, loss = model(ids, mask, tti, lab)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += float(loss.item())
            n += 1
            if logger is not None:
                logger.log_step(epoch=epoch, step=step, loss=float(loss.item()),
                                lr=sched.get_last_lr()[0], extra={"stage": task})
        print(f"  [{ckpt_name}] epoch {epoch}/{args.cls_epochs} loss={tot / max(n, 1):.4f}",
              flush=True)
    train_s = time.perf_counter() - t0
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "args": vars(args)}, CKPT_DIR / ckpt_name)
    return model, {"train_s": round(train_s, 1), "epochs": args.cls_epochs}

@torch.no_grad()
def predict_classifier(model, rows, tokenizer, args, device, task="detect",
                       type2id=None, batch_size=64) -> list[int]:
    """批量推理分类模型，返回每行的预测 label id（或 0/1）。"""
    ds = ClsDataset(rows, tokenizer, args.cls_max_len, task=task, type2id=type2id or {})
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=0, collate_fn=collate_cls)
    model.eval()
    out = []
    for batch in loader:
        logits, _ = model(batch["input_ids"].to(device),
                          batch["attention_mask"].to(device),
                          batch["token_type_ids"].to(device))
        out.extend(logits.argmax(-1).tolist())
    return out

# ─────────────────────────── (c) 类型条件论元抽取 ───────────────────────────

def train_arg_experts(extract_rows, schema, tokenizer, args, device,
                      logger=None) -> dict:
    """
    为每个事件类型训一个「该类型的专家」序列标注模型。

    训练数据：只用该类型的句子（平均 400 条，最少的「被约谈」只有 53 条）
    标签空间：TRG + roles(t)，即 11~19 个标签
    关键：13 个模型是**依次训练**的，每个训完就释放显存，
         否则 13 × BERT 同时驻留显存，8GB 显卡直接爆。
    """
    by_type = defaultdict(list)
    for r in extract_rows:
        ev = r.get("event")
        if ev:
            by_type[ev["type"]].append(r)

    infos = {}
    t_all = time.perf_counter()
    for etype in sorted(schema):
        rows = by_type.get(etype, [])
        if not rows:
            print(f"  [arg:{etype}] 没有训练样本，跳过")
            continue
        scheme = build_typed_schema(schema, etype)
        ds = BioDataset(rows, tokenizer, scheme, args.max_len, typed=True)
        loader = DataLoader(ds, batch_size=args.arg_batch_size, shuffle=True,
                            num_workers=0, collate_fn=collate_bio)
        encoder_path = str(args.bert_path)

        model = build_arg_model(encoder_path, scheme, args).to(device)
        opt = AdamW(model.parameters(), lr=args.arg_lr, weight_decay=0.01)
        total = max(1, len(loader) * args.arg_epochs)
        sched = get_linear_schedule_with_warmup(opt, int(total * 0.1), total)

        t0 = time.perf_counter()
        for epoch in range(1, args.arg_epochs + 1):
            model.train()
            tot, n = 0.0, 0
            for step, batch in enumerate(loader):
                inp = batch["input_ids"].to(device)
                msk = batch["attention_mask"].to(device)
                tti = batch["token_type_ids"].to(device)
                lab = batch["labels"].to(device)
                _, loss = model(inp, msk, lab, tti)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                tot += float(loss.item())
                n += 1
                if logger is not None:
                    logger.log_step(epoch=epoch, step=step, loss=float(loss.item()),
                                    lr=sched.get_last_lr()[0],
                                    extra={"stage": f"arg:{etype}"})
            if epoch == args.arg_epochs:
                print(f"  [arg:{etype}] {len(rows)} 句 × {args.arg_epochs} epoch "
                      f"loss={tot / max(n, 1):.4f}", flush=True)
        torch.save({"state_dict": model.state_dict(),
                    "labels": scheme["labels"], "etype": etype,
                    "args": vars(args)}, CKPT_DIR / f"pipe_arg_{etype}.pt")
        infos[etype] = {"n_rows": len(rows), "num_labels": len(scheme["labels"]),
                        "train_s": round(time.perf_counter() - t0, 1)}
        # 立刻释放：8GB 卡上绝不能同时留 13 个 BERT
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    infos["_total_train_s"] = round(time.perf_counter() - t_all, 1)
    return infos

# ─────────────────────────── 组合：检测 → 类型 → 论元 ───────────────────────────

def compose(dev_extract_rows, dev_detect_rows, tokenizer, args, device,
            oracle_type: bool = False) -> tuple[list[dict], dict]:
    """
    把三段串起来，对 dev_extract 的 930 句出端到端预测。

    oracle_type=True 时用**金标类型**代替类型分类器的输出 —— 这不是作弊，
    而是标准的「上界分析」：如果类型给对能到多少，就能判断端到端的主要损失
    是来自「类型判错」还是「论元抽不准」，PPT 上定位瓶颈很有用。
    """
    from models import BertEncoder, build_arg_model, predict_char_tags

    schema = load_schema()
    enums = load_enums()

    # (a) 检测器：在 dev_extract 这 930 句上判断有没有事件
    det_model = BertCls(str(args.bert_path), 2, args.dropout).to(device)
    det_ckpt = torch.load(CKPT_DIR / "pipe_detector.pt", map_location=device,
                          weights_only=False)
    det_model.load_state_dict(det_ckpt["state_dict"])
    det = predict_classifier(det_model, dev_extract_rows, tokenizer, args, device, task="detect")
    del det_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # (b) 类型 13 分类
    types_sorted = sorted(schema)
    type2id = {t: i for i, t in enumerate(types_sorted)}
    id2type = {i: t for t, i in type2id.items()}
    type_model = BertCls(str(args.bert_path), len(types_sorted), args.dropout).to(device)
    ty_ckpt = torch.load(CKPT_DIR / "pipe_type.pt", map_location=device, weights_only=False)
    type_model.load_state_dict(ty_ckpt["state_dict"])
    type_pred = predict_classifier(type_model, dev_extract_rows, tokenizer, args,
                                   device, task="type", type2id=type2id)
    del type_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # 决定每句用哪个类型
    final_types = []
    n_type_from_gold = 0
    for row, tp in zip(dev_extract_rows, type_pred):
        if oracle_type:
            ev = row.get("event") or {}
            final_types.append(ev.get("type", id2type[tp]))
            n_type_from_gold += 1
        else:
            final_types.append(id2type[tp])

    # (c) 按「预测类型」分组，每组的专家模型只加载一次
    preds: list[dict] = [None] * len(dev_extract_rows)
    idx_by_type = defaultdict(list)
    for i, (row, t) in enumerate(zip(dev_extract_rows, final_types)):
        if det[i] == 0:
            preds[i] = make_pred_row(row, [], extra={"has_event": 0})
        else:
            idx_by_type[t].append(i)

    n_arg_trunc = 0
    for etype, idxs in idx_by_type.items():
        ckpt_path = CKPT_DIR / f"pipe_arg_{etype}.pt"
        if not ckpt_path.exists():
            for i in idxs:
                preds[i] = make_pred_row(dev_extract_rows[i], [], extra={"has_event": 1})
            continue
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        scheme = {"labels": ckpt["labels"],
                  "label2id": {l: k for k, l in enumerate(ckpt["labels"])},
                  "id2label": {k: l for k, l in enumerate(ckpt["labels"])}}
        model = build_arg_model(str(args.bert_path), scheme,
                                _layers_of(ckpt, args.dropout)).to(device)
        model.load_state_dict(ckpt["state_dict"])
        rows = [dev_extract_rows[i] for i in idxs]
        ds = BioDataset(rows, tokenizer, scheme, args.max_len, typed=True)
        n_arg_trunc += ds.n_truncated
        loader = DataLoader(ds, batch_size=args.arg_batch_size, shuffle=False,
                            num_workers=0, collate_fn=collate_bio)
        # 注意 predict_char_tags（train_bio 里）返回 (tags, info)，这里要用解包拿到 tags
        tags, _ = predict_char_tags(model, loader, scheme["id2label"], device)
        for i, row, tg in zip(idxs, rows, tags):
            ev = spans_to_event(decode_bio_spans(tg), row["text"], typed_type=etype,
                                enums=enums)
            preds[i] = make_pred_row(row, [ev] if ev else [], extra={"has_event": 1})
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # 保险：任何没被填上的行都写成空预测，绝不丢行（丢行会让评测分母变小、F1 虚高）
    for i, p in enumerate(preds):
        if p is None:
            preds[i] = make_pred_row(dev_extract_rows[i], [], extra={"has_event": 0})

    # 统计三段各错多少，便于定位瓶颈
    n_det_miss = sum(1 for d in det if d == 0)
    n_type_wrong = sum(1 for row, t in zip(dev_extract_rows, final_types)
                       if t != (row.get("event") or {}).get("type"))
    diag = {
        "n_detect_dropped": n_det_miss,
        "n_type_wrong": n_type_wrong,
        "type_accuracy_on_dev_extract": round(
            1 - n_type_wrong / max(len(dev_extract_rows), 1), 4),
        "n_arg_truncated": n_arg_trunc,
    }
    return preds, diag

# ─────────────────────────── 主流程 ───────────────────────────

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    data_dir = (Path(args.data_dir) if args.data_dir
                else (MOCK_DIR if args.mock else DATA_DIR))
    suffix = "_mock" if args.mock else ""
    print(f"设备：{device} | 数据：{data_dir}")

    schema = load_schema(data_dir / "event_schema.json")
    types_sorted = sorted(schema)
    type2id = {t: i for i, t in enumerate(types_sorted)}
    tokenizer = BertTokenizer.from_pretrained(str(args.bert_path))

    train_extract = load_extract_split("train", data_dir)
    dev_extract = load_extract_split("dev", data_dir)
    train_detect = load_detect_split("train", data_dir)
    dev_detect_path = data_dir / "dev_detect.jsonl"
    dev_detect = load_jsonl(dev_detect_path)
    clean_path = data_dir / "dev_detect_clean.jsonl"
    dev_detect_clean = load_jsonl(clean_path) if clean_path.exists() else None
    print(f"数据：train_extract {len(train_extract)} / dev_extract {len(dev_extract)} / "
          f"train_detect {len(train_detect)} / dev_detect {len(dev_detect)}"
          + (f" / clean {len(dev_detect_clean)}" if dev_detect_clean else ""))

    # 两段式是 16 个模型（1 检测器 + 1 类型分类 + 13 论元专家）的组合。
    # 日志按「阶段」记录，学生能看出训练时间花在哪一段。
    logger = TrainLogger(method="bert_pipeline", args=args, root=ROOT,
                         notes="两段式：检测器 + 13 分类 + 13 个类型条件论元专家")

    t_all = time.perf_counter()

    # --only_experts：只重训 13 个论元专家、复用已训好的 detector / type classifier。
    # 用途：修了「typed 标签把论元刷成触发词」的 bug 后，只有专家的标签受影响，
    # detector/type 走的是 ClsDataset，完全没被这个 bug 污染 —— 没必要重训它们（省 20 分钟）。
    if args.only_experts:
        print("  [only_experts] 跳过检测器/类型分类训练，复用现有 checkpoint")

    # ── (a) 检测器 ──
    # --detect_limit：时间紧时只用一部分 train_detect 训练检测器（按正负比例分层抽样，
    # 否则检测器会退化成只猜多数类）。用了它就必须在 meta 里如实写训练条数，不能假装是全量。
    det_train_rows = train_detect
    if args.detect_limit and args.detect_limit < len(train_detect):
        import random as _random
        pos = [r for r in train_detect if int(r.get("n_events", 0)) > 0]
        neg = [r for r in train_detect if int(r.get("n_events", 0)) <= 0]
        _rng = _random.Random(args.seed)
        n_pos = max(1, round(args.detect_limit * len(pos) / len(train_detect)))
        _rng.shuffle(pos); _rng.shuffle(neg)
        det_train_rows = pos[:n_pos] + neg[:max(1, args.detect_limit - n_pos)]
        print(f"    [检测器] 只用 {len(det_train_rows)}/{len(train_detect)} 条训练"
              f"（正样本 {n_pos} 条，保持负样本比例）")
    det_train_s = 0.0
    if not args.skip_train and not args.only_experts:
        print("\n(a) 训练事件检测器（二分类，负样本占大头）...")
        _, info = train_classifier(det_train_rows, None, 2, tokenizer, args, device,
                                   "pipe_detector.pt", task="detect", logger=logger)
        det_train_s = info["train_s"]

    det_model = BertCls(str(args.bert_path), 2, args.dropout).to(device)
    det_model.load_state_dict(torch.load(CKPT_DIR / "pipe_detector.pt",
                                         map_location=device,
                                         weights_only=False)["state_dict"])
    t0 = time.perf_counter()
    det_pred = predict_classifier(det_model, dev_detect, tokenizer, args, device, task="detect")
    det_ms = (time.perf_counter() - t0) * 1000 / max(len(dev_detect), 1)
    del det_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # 检测指标：全量 dev_detect + clean 版各报一次（数据泄漏素材）
    from evaluate_events import detection_metrics
    rows_pred = [{"id": r["id"], "sent_id": r["sent_id"], "has_event": int(p)}
                 for r, p in zip(dev_detect, det_pred)]
    det_full = detection_metrics(dev_detect, rows_pred)
    det_clean = None
    if dev_detect_clean:
        keep = {(r["id"], r["sent_id"]) for r in dev_detect_clean}
        sub = [x for x in rows_pred if (x["id"], x["sent_id"]) in keep]
        det_clean = detection_metrics(dev_detect_clean, sub)
    METRIC_DIR.mkdir(parents=True, exist_ok=True)
    det_res = {
        "method": f"pipe_detector{suffix}",
        "dev_detect_full": det_full,
        "dev_detect_clean": det_clean,
        "meta": {"display_name": "检测器（BERT 二分类）",
                 "train_s": det_train_s, "ms_per_sentence": round(det_ms, 2),
                 "dev_detect_n": len(dev_detect),
                 "detect_train_n": len(det_train_rows),
                 "dev_detect_clean_n": len(dev_detect_clean) if dev_detect_clean else 0,
                 "note": "dev_detect 与 train 有 9.2% 文本重复（真实泄漏），"
                         "clean 版去掉重复后指标更能代表真实泛化"},
    }
    with open(METRIC_DIR / f"pipe_detector{suffix}.json", "w", encoding="utf-8") as f:
        json.dump(det_res, f, ensure_ascii=False, indent=2)
    print(f"   检测器 dev_detect：P={det_full['precision']:.4f} R={det_full['recall']:.4f} "
          f"F1={det_full['f1']:.4f} acc={det_full['accuracy']:.4f}")
    if det_clean:
        print(f"   检测器 clean({len(dev_detect_clean)})：P={det_clean['precision']:.4f} "
              f"R={det_clean['recall']:.4f} F1={det_clean['f1']:.4f} "
              f"acc={det_clean['accuracy']:.4f}")

    # ── (b) 类型 13 分类 ──
    if not args.skip_train and not args.only_experts:
        print("\n(b) 训练事件类型 13 分类器...")
        train_classifier(train_extract, type2id, len(types_sorted), tokenizer, args, device,
                         "pipe_type.pt", task="type", logger=logger)

    # ── (c) 13 个类型条件论元专家 ──
    arg_infos = {}
    if not args.skip_train:
        print("\n(c) 训练 13 个类型条件论元抽取专家（依次训练，训完即释放显存）...")
        _t_experts = time.perf_counter()
    arg_infos = train_arg_experts(train_extract, schema, tokenizer, args, device,
                                  logger=logger)
    logger.log_epoch(epoch=args.arg_epochs, train_loss=0.0,
                     elapsed=time.perf_counter() - _t_experts,
                     val_metrics={k: v.get("train_s") for k, v in arg_infos.items()
                                  if isinstance(v, dict)},
                     extra={"stage": "13_argument_experts"})

    # ── 组合评测 ──
    print("\n组合三段 → dev_extract 端到端评测...")
    with Timer() as timer:
        preds, diag = compose(dev_extract, dev_detect, tokenizer, args, device)
    lat = timer.summary(len(dev_extract))
    print(f"   诊断：检测器丢掉 {diag['n_detect_dropped']} 句 / "
          f"类型判错 {diag['n_type_wrong']} 句 / "
          f"dev_extract 上类型准确率 {diag['type_accuracy_on_dev_extract']:.4f}")

    meta = {
        "display_name": "两段式流水线（检测→类型→类型条件论元）",
        "train_s": round(time.perf_counter() - t_all, 1),
        "ms_per_sentence": lat["ms_per_sentence"],
        "peak_gpu_mb": lat["peak_gpu_mb"],
        "max_len": args.max_len, "arg_epochs": args.arg_epochs,
        "pipeline_stages": 3,
        "num_models": 1 + 1 + len(arg_infos),
        "diag": diag,
        "arg_experts": arg_infos,
        "note": "三段各跑一次；延迟约是单模型的 3 倍（见 meta.ms_per_sentence）",
    }
    res = evaluate_predictions(f"pipeline{suffix}", preds, meta=meta,
                              gold_path=data_dir / "dev_extract.jsonl", with_detect=True)

    # ── 上界分析：类型给金标 ──
    print("\n上界分析：类型用金标（隔离出「类型判错」造成的损失）...")
    preds_o, diag_o = compose(dev_extract, dev_detect, tokenizer, args, device,
                              oracle_type=True)
    meta_o = dict(meta)
    meta_o["display_name"] = "两段式（类型给金标的上界）"
    meta_o["note"] = "类型用金标，只保留检测+论元两段的错误 → 用来定位瓶颈在类型还是论元"
    meta_o["diag"] = diag_o
    res_o = evaluate_predictions(f"pipeline_oracle_type{suffix}", preds_o, meta=meta_o,
                                gold_path=data_dir / "dev_extract.jsonl")

    print("\n" + "=" * 70)
    print(f"{'方法':<26}{'trigger F1':>12}{'arg F1':>10}{'event F1':>10}")
    print("-" * 70)
    for tag, r in (("流水线（端到端）", res), ("流水线（类型金标上界）", res_o)):
        print(f"{tag:<26}{r['trigger']['f1']:>12.4f}{r['argument']['f1']:>10.4f}"
              f"{r['event_exact']['f1']:>10.4f}")
    print("=" * 70)
    print(f"总耗时 {meta['train_s']:.0f}s，延迟 {lat['ms_per_sentence']} ms/句")
    logger.finalize(total_s=meta["train_s"], diag=diag,
                    dev_metrics={k: res[k]["f1"] for k in ("trigger", "argument", "event_exact")},
                    inference_latency_ms=lat["ms_per_sentence"], peak_gpu_mb=peak_gpu_mb(),
                    pipeline_stages=3, num_models=meta.get("num_models"))

def parse_args():
    ap = argparse.ArgumentParser(description="两段式流水线事件抽取")
    ap.add_argument("--bert_path", default="")
    ap.add_argument("--cls_epochs", type=int, default=2)
    ap.add_argument("--detect_limit", type=int, default=0,
                    help="检测器只用前 N 条 train_detect 训练（0=全量；时间紧时用，meta 里会如实标注）")
    ap.add_argument("--cls_lr", type=float, default=2e-5)
    ap.add_argument("--cls_batch_size", type=int, default=64)
    ap.add_argument("--cls_max_len", type=int, default=192)
    ap.add_argument("--arg_epochs", type=int, default=3)
    ap.add_argument("--arg_lr", type=float, default=3e-5)
    ap.add_argument("--arg_batch_size", type=int, default=32)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--arg_layers", type=int, default=4,
                    help="论元专家保留几层 BERT（默认 4：13 个模型省磁盘、快 3 倍）")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--device", default="", help="cuda / cpu；留空自动选（mock 冒烟建议 cpu，别抢卡）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip_train", action="store_true")
    ap.add_argument("--log_every", type=int, default=50,
                    help="每多少步往 steps.jsonl 写一行（供学生画 loss 曲线）")
    ap.add_argument("--only_experts", action="store_true",
                    help="只重训 13 个论元专家，复用已有 detector/type")
    ap.add_argument("--mock", action="store_true", help="用 data/_mock 的 20 条手造数据（离线冒烟）")
    ap.add_argument("--data_dir", default="", help="数据目录；留空=全量 data/processed（--mock 时用 data/_mock）")
    args = ap.parse_args()
    if not args.bert_path:
        args.bert_path = str(resolve_model_path(BERT_MODEL_DIR))
    return args

if __name__ == "__main__":
    main()
