"""
CNVD 漏洞描述 severity 三分类 —— BERT 微调（消融实验版）

参照课程 code/bert/train_bert.py 的骨架，所有消融旋钮全部参数化；
每次训练在 logs/ 下生成「文件名 = 训练参数」的日志与同名 .json 指标。

代码组织（本作业统一结构，BERT 与后续 LLM 对齐）：
  dataset.py      数据层（BERT / LLM 共用同一份 jsonl）
  model_bert.py   模型 + 池化 + FGM + 损失
  train_bert.py   训练（本文件）
  predict_bert.py 评测
  paths.py        路径解析（本机 / Colab 挂载 Drive）

消融旋钮：
  --pool {cls,mean,max}                 池化方式
  --class_weight {none,sqrt,balanced}   类别加权
  --focal_gamma 2.0                     >0 启用 Focal Loss
  --label_smoothing 0.1                 >0 启用软标签（与 Focal 同时给时按 Focal 跑）
  --fgm --fgm_eps 1.0                   FGM 对抗训练（耗时约 ×1.5）
  --use_title                           输入改为 title：description
  --train_file                          换训练集规模（数据量消融）
  --max_length                          截断长度（128 可作消融档，192 近无损）

用法：
  python train_bert.py                                   # 基线
  python train_bert.py --pool mean                        # 池化消融
  python train_bert.py --class_weight sqrt                # 加权消融
  python train_bert.py --fgm                              # 对抗训练
  python train_bert.py --train_file data/ablation/train_20k.jsonl
  # 冒烟（本机 CPU）：
  python train_bert.py --max_steps 20 --max_eval 500 --no_save --skip_test
  # Colab 上产物写 Drive：
  WEEK6_OUT_ROOT=/content/drive/MyDrive/week6_runs python train_bert.py
"""

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import paths                                                    # noqa: E402
from dataset import (BertDataset, build_text, class_counts,      # noqa: E402
                     load_jsonl, load_label_map, make_collate_fn,
                     seed_everything, token_length_stats)
from metrics import brief, classification_report                 # noqa: E402
from model_bert import (FGM, build_loss_fn, build_model,         # noqa: E402
                        compute_class_weight, get_param_groups)


# ─────────────────────────────────────────────────────────────────────────────
# 模型获取：本地目录优先，否则从 ModelScope 下载到 MODEL_ROOT（代码目录同级的 models/）
# ─────────────────────────────────────────────────────────────────────────────

def ensure_local_model(name_or_path: str, logger=None) -> Path:
    """
    本地目录优先；否则按 ModelScope 的文件接口逐文件下载（带进度、可断点复用）。
    不用 snapshot_download：某些版本会漏拉权重，也不利于断线重连后复用已下文件。
    """
    import urllib.request

    log = logger.info if logger else print
    p = Path(name_or_path)
    if p.exists():
        log(f"使用本地模型目录: {p}")
        return p
    root = Path(paths.MODEL_ROOT)
    cache = root / str(name_or_path).split("/")[-1]
    needed = ["config.json", "vocab.txt", "pytorch_model.bin"]
    # 旧版默认把底座下在代码目录的 pretrain_models（Colab 上一度在 Drive）；
    # 换了默认位置（同级 models/）不该把 412MB 重下一遍，先看那两处有没有现成的
    for old in paths.legacy_model_roots():
        cand = old / cache.name
        if cand.is_dir() and all((cand / f).is_file() and (cand / f).stat().st_size > 0
                                 for f in needed):
            log(f"复用旧缓存: {cand}（新默认位置是 {root}，确认无误可自行挪过去）")
            return cand
    base = (f"https://www.modelscope.cn/api/v1/models/{name_or_path}"
            f"/repo?Revision=master&FilePath=")
    cache.mkdir(parents=True, exist_ok=True)
    for fname in needed:
        dst = cache / fname
        if dst.exists() and dst.stat().st_size > 0:
            log(f"已缓存 {fname}（{dst.stat().st_size:,} B），跳过下载")
            continue
        tmp = dst.with_suffix(dst.suffix + ".tmp")
        log(f"从 ModelScope 下载 {name_or_path}/{fname} …")
        req = urllib.request.Request(base + fname, headers={"User-Agent": "Mozilla/5.0"})
        t0, got, last_log = time.time(), 0, 0.0
        with urllib.request.urlopen(req, timeout=180) as r, open(tmp, "wb") as f:
            total = int(r.headers.get("Content-Length", 0))
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                got += len(chunk)
                if time.time() - last_log > 5:
                    speed = got / max(1e-9, time.time() - t0) / (1 << 20)
                    pct = f"（{got / total * 100:.0f}%）" if total else ""
                    log(f"  {got / (1 << 20):.0f}/{total / (1 << 20):.0f} MB{pct} @ {speed:.1f} MB/s")
                    last_log = time.time()
        tmp.rename(dst)
        log(f"  完成 {fname}（{dst.stat().st_size:,} B，{time.time() - t0:.0f}s）")
    # 词表配置：该 backbone 与 Google bert-base-chinese 同源（vocab 21128）
    tc = cache / "tokenizer_config.json"
    if not tc.exists():
        tc.write_text(json.dumps({"do_lower_case": True, "model_max_length": 512,
                                  "tokenizer_class": "BertTokenizer"},
                                 ensure_ascii=False, indent=2), encoding="utf-8")
    return cache


# ─────────────────────────────────────────────────────────────────────────────
# 评测（口径统一在 metrics.classification_report）
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, device, id2name):
    model.eval()
    ys, ps = [], []
    for batch in loader:
        tt = batch.get("token_type_ids")
        logits = model(batch["input_ids"].to(device),
                       batch["attention_mask"].to(device),
                       tt.to(device) if tt is not None else None)
        ps.extend(logits.argmax(-1).cpu().tolist())
        ys.extend(batch["labels"].tolist())
    return classification_report(ys, ps, id2name=id2name)


# ─────────────────────────────────────────────────────────────────────────────
# 日志命名：文件名 = 训练参数串
# ─────────────────────────────────────────────────────────────────────────────

def build_run_name(args, n_train: int) -> str:
    """
    参数串即身份：调度器靠它判断「这组跑过没有」，predict 靠它找回 ckpt。
    改动这里会让历史日志对不上，谨慎。
    """
    parts = [
        f"pool-{args.pool}",
        f"cw-{args.class_weight}",
        f"focal-{args.focal_gamma:g}" if args.focal_gamma > 0 else "focal-0",
        f"ls-{args.label_smoothing:g}",
        f"fgm-{int(args.fgm)}" + (f"-eps{args.fgm_eps:g}" if args.fgm else ""),
        f"len-{args.max_length}",
        "text-titledesc" if args.use_title else "text-desc",
        f"n-{n_train}",
        f"bs-{args.batch_size}",
        f"lr-{args.lr:g}",
        f"ep-{args.epochs}",
        f"seed-{args.seed}",
    ]
    if args.max_steps:
        parts.append("smoke")
    return "_".join(parts)


def unique_path(path: Path) -> Path:
    """同参数重跑：加 _r2/_r3 后缀，不覆盖历史"""
    if not path.exists():
        return path
    for k in range(2, 100):
        cand = path.with_stem(f"{path.stem}_r{k}")
        if not cand.exists():
            return cand
    raise FileExistsError(path)


def setup_logger(log_path: Path):
    logger = logging.getLogger("train_bert")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)
    logger.propagate = False
    return logger


# ─────────────────────────────────────────────────────────────────────────────
# 参数
# ─────────────────────────────────────────────────────────────────────────────

def build_arg_parser() -> argparse.ArgumentParser:
    """单独暴露成函数：调度器 run_experiments.py 复用它来预测 run_name"""
    ap = argparse.ArgumentParser(description="CNVD severity 三分类 BERT 微调（消融实验版）")
    # —— 消融旋钮 ——
    ap.add_argument("--pool", default="cls", choices=["cls", "mean", "max"])
    ap.add_argument("--class_weight", default="none", choices=["none", "sqrt", "balanced"])
    ap.add_argument("--focal_gamma", type=float, default=0.0)
    ap.add_argument("--label_smoothing", type=float, default=0.0)
    ap.add_argument("--fgm", action="store_true")
    ap.add_argument("--fgm_eps", type=float, default=1.0)
    ap.add_argument("--use_title", action="store_true")
    # —— 数据 ——
    ap.add_argument("--data_dir", default=str(paths.DATA_DIR), type=Path)
    ap.add_argument("--train_file", default=None, type=Path)
    ap.add_argument("--val_file", default=None, type=Path)
    ap.add_argument("--test_file", default=None, type=Path)
    # —— 训练超参 ——
    ap.add_argument("--model_name_or_path", default="iic/nlp_roberta_backbone_base_std",
                    help="本地目录或 ModelScope 模型 id（首跑自动下载到 MODEL_ROOT）")
    ap.add_argument("--max_length", type=int, default=192)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--eval_batch_size", type=int, default=64)
    ap.add_argument("--grad_accum", type=int, default=1, help="梯度累积，显存不足时设 2/4")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--head_lr_mult", type=float, default=5.0)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--patience", type=int, default=0, help="早停耐心值，0=不早停")
    ap.add_argument("--seed", type=int, default=42)
    # —— 输出 ——
    ap.add_argument("--out_root", default=None, type=Path,
                    help="产物根目录；默认取 WEEK6_OUT_ROOT 或本目录（Colab 上指向 Drive）")
    ap.add_argument("--log_dir", default=None, type=Path)
    ap.add_argument("--ckpt_dir", default=None, type=Path)
    # —— 冒烟/调试 ——
    ap.add_argument("--max_steps", type=int, default=0, help="每 epoch 最多跑多少步（冒烟）")
    ap.add_argument("--max_eval", type=int, default=0, help="验证集最多评多少条（冒烟，等距抽样）")
    ap.add_argument("--no_save", action="store_true")
    ap.add_argument("--skip_test", action="store_true")
    ap.add_argument("--force_name", default=None,
                    help="强制指定 run_name（调度器重试时用，覆盖同名日志）")
    ap.add_argument("--print_run_name", action="store_true",
                    help="只打印 run_name 后退出，供调度器预演")
    return ap


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    from transformers import AutoTokenizer

    # ── 输出路径：命令行 > 环境变量 > 本目录 ──
    out_root = Path(args.out_root) if args.out_root else paths.OUT_ROOT
    log_dir = Path(args.log_dir) if args.log_dir else out_root / "logs"
    ckpt_dir = Path(args.ckpt_dir) if args.ckpt_dir else out_root / "checkpoints"

    # ── 数据与映射 ──
    data_dir = Path(args.data_dir)
    train_file = Path(args.train_file) if args.train_file else data_dir / "train_5k.jsonl"
    val_file = Path(args.val_file) if args.val_file else data_dir / "val_2k.jsonl"
    test_file = Path(args.test_file) if args.test_file else data_dir / "test_3k.jsonl"
    name2id, id2name, _ = load_label_map(data_dir)

    n_train = sum(1 for _ in open(train_file, encoding="utf-8"))
    run_name = args.force_name or build_run_name(args, n_train)
    if args.print_run_name:
        print(run_name)
        return 0

    # ── 日志 ──
    log_dir.mkdir(parents=True, exist_ok=True)
    stem = run_name if args.force_name else unique_path(log_dir / f"{run_name}.log").stem
    log_path = log_dir / f"{stem}.log"
    logger = setup_logger(log_path)
    logger.info(f"run_name: {stem}")
    logger.info(f"路径: {paths.summary()} | log_dir={log_dir} ckpt_dir={ckpt_dir}")
    logger.info("args: " + json.dumps({k: str(v) for k, v in vars(args).items()}, ensure_ascii=False))

    t_start = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"device = {device}"
                + (f"（{torch.cuda.get_device_name(0)}）" if device == "cuda"
                   else "（CPU，冒烟可以，全量请上 Colab）"))
    seed_everything(args.seed)

    # ── 模型与 tokenizer ──
    model_dir = ensure_local_model(args.model_name_or_path, logger)
    tok = AutoTokenizer.from_pretrained(str(model_dir))

    train_rows = load_jsonl(train_file)
    train_texts = [build_text(r, args.use_title) for r in train_rows]
    train_labels = []
    for r in train_rows:
        if r["label_id"] != name2id[r["label"]]:
            raise ValueError(f"{train_file}：label_id 与 label_map 不一致（{r['id']}）")
        train_labels.append(r["label_id"])
    data_sha256 = hashlib.sha256(train_file.read_bytes()).hexdigest()
    logger.info(f"加载 {train_file.name}: {len(train_rows)} 条（use_title={args.use_title}）")
    logger.info(f"训练数据 sha256={data_sha256[:16]}…（与 LLM 同规模臂核对此值即知是否同一份数据）")

    val_rows = load_jsonl(val_file)
    val_texts = [build_text(r, args.use_title) for r in val_rows]
    val_labels = [r["label_id"] for r in val_rows]
    logger.info(f"加载 {val_file.name}: {len(val_rows)} 条")

    if args.max_eval:
        step = max(1, len(val_rows) // args.max_eval)
        idx = list(range(0, len(val_rows), step))[:args.max_eval]
        val_texts = [val_texts[i] for i in idx]
        val_labels = [val_labels[i] for i in idx]
        logger.info(f"冒烟截断 val 至 {len(val_texts)} 条（等距抽样，保持类别分布）")

    # tokenizer 健康检查：[UNK] 占比异常说明词表/do_lower_case 配置有问题
    probe = tok(train_texts[:200], truncation=True, max_length=args.max_length)["input_ids"]
    n_tok = sum(len(x) for x in probe)
    n_unk = sum(x.count(tok.unk_token_id) for x in probe)
    logger.info(f"tokenizer [UNK] 占比（train 前 200 条）: {n_unk}/{n_tok} = {n_unk / max(1, n_tok):.3%}")
    if n_unk / max(1, n_tok) > 0.05:
        logger.warning("[UNK] 占比 > 5%，检查 tokenizer 配置（do_lower_case / 词表版本）")
    st = token_length_stats(train_texts, tok, args.max_length)
    logger.info(f"token 长度: 均值 {st['mean']} 中位 {st['median']} P95 {st['p95']} "
                f"最长 {st['max']} | max_length={args.max_length} 截断 {st['truncation_pct']}%")

    train_ds = BertDataset(train_texts, train_labels, tok, args.max_length)
    val_ds = BertDataset(val_texts, val_labels, tok, args.max_length)
    g = torch.Generator().manual_seed(args.seed)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, generator=g,
                          collate_fn=make_collate_fn(tok))
    val_dl = DataLoader(val_ds, batch_size=args.eval_batch_size, shuffle=False,
                        collate_fn=make_collate_fn(tok))

    model = build_model(model_dir, num_labels=len(name2id), pool=args.pool,
                        dropout=args.dropout, logger=logger).to(device)
    opt = torch.optim.AdamW(get_param_groups(model, args.lr, args.head_lr_mult),
                            weight_decay=args.weight_decay)

    # ── 类别加权：本任务 6.2x 不均衡（中 55.3% vs 低 8.9%），低类 recall 是观测重点 ──
    counts = class_counts(train_labels, len(name2id))
    weight = compute_class_weight(train_labels, len(name2id), args.class_weight)
    if weight is not None:
        logger.info("class_weight({}) = ".format(args.class_weight)
                    + json.dumps({id2name[i]: round(float(x), 3) for i, x in enumerate(weight)},
                                 ensure_ascii=False))
    logger.info("train 分布: " + json.dumps({id2name[i]: c for i, c in enumerate(counts)},
                                            ensure_ascii=False))

    if args.focal_gamma > 0 and args.label_smoothing > 0:
        logger.warning("同时给了 --focal_gamma 和 --label_smoothing，按 Focal 跑")
    crit, crit_desc = build_loss_fn(args, weight, device)
    logger.info(f"loss = {crit_desc}, class_weight = {args.class_weight}")

    steps_per_epoch = min(len(train_dl), args.max_steps) if args.max_steps else len(train_dl)
    total_steps = max(1, steps_per_epoch * args.epochs // max(1, args.grad_accum))
    warmup = int(total_steps * args.warmup_ratio)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda step: step / max(1, warmup) if step < warmup
        else max(0.0, (total_steps - step) / max(1, total_steps - warmup)))
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    fgm = FGM(model, eps=args.fgm_eps) if args.fgm else None
    logger.info(f"[cfg] pool={args.pool} loss={crit_desc} cw={args.class_weight} "
                f"fgm={'on(eps=%s)' % args.fgm_eps if fgm else 'off'} epochs={args.epochs} "
                f"steps/epoch={steps_per_epoch} grad_accum={args.grad_accum} "
                f"train_n={len(train_ds)} eval_n={len(val_ds)} max_length={args.max_length}")

    # ── 训练循环 ──
    best_f1, best_state, best_epoch = -1.0, None, 0
    no_improve, epochs_log = 0, []
    opt.zero_grad(set_to_none=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, running, n_step = time.time(), 0.0, 0
        for i, batch in enumerate(train_dl):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            tt = batch.get("token_type_ids")
            tt = tt.to(device) if tt is not None else None
            y = batch["labels"].to(device)

            with torch.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
                loss = crit(model(ids, mask, tt), y)
            # 梯度累积：loss 先除以累积步数，等效于更大 batch
            scaler.scale(loss / args.grad_accum).backward()

            if fgm is not None:
                fgm.attack()
                with torch.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
                    loss_adv = crit(model(ids, mask, tt), y)
                scaler.scale(loss_adv / args.grad_accum).backward()
                fgm.restore()

            if (i + 1) % args.grad_accum == 0 or (i + 1) == steps_per_epoch:
                scaler.unscale_(opt)      # 裁剪梯度前必须反缩放，否则阈值失去意义
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()

            running += loss.item()
            n_step += 1
            if i % 200 == 0:
                logger.info(f"  epoch {epoch} step {i}/{steps_per_epoch} loss={loss.item():.4f}")
            if args.max_steps and n_step >= args.max_steps:
                break

        rep = evaluate(model, val_dl, device, id2name)
        is_best = rep["macro_f1"] > best_f1
        if is_best:
            best_f1, best_epoch = rep["macro_f1"], epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if not args.no_save:
                ck = ckpt_dir / stem
                ck.mkdir(parents=True, exist_ok=True)
                # args 序列化成纯类型再存：torch 2.6+ 的 torch.load 默认
                # weights_only=True，Path 对象会直接报错。
                # 用 default=str 而不是逐项 str(v)：后者会把 None 变成 "None"、
                # 0 变成 "0"，两者都是真值，会让 predict 侧「是否指定了 test_file」
                # 之类的判断全部误判。
                safe_args = json.loads(json.dumps(vars(args), default=str))
                torch.save({"state_dict": best_state, "pool": args.pool,
                            "val_macro_f1": best_f1, "args": safe_args}, ck / "best.pt")
        epochs_log.append({"epoch": epoch,
                           "train_loss": round(running / max(1, n_step), 4),
                           "lr": round(opt.param_groups[0]["lr"], 8),
                           "val": rep, "is_best": is_best,
                           "elapsed_s": round(time.time() - t0, 1)})
        star = " ★best" if is_best else ""
        logger.info(f"epoch {epoch}: train_loss={running / max(1, n_step):.4f} "
                    f"elapsed={time.time() - t0:.0f}s | val {brief(rep, id2name)}{star}")

        if args.patience > 0:
            no_improve = 0 if is_best else no_improve + 1
            if no_improve >= args.patience:
                logger.info(f"早停：{args.patience} 个 epoch 无提升（best epoch = {best_epoch}）")
                break

    # ── best ckpt 在 test 上只评一次（消融口径：所有组都报 test）──
    test_rep = None
    if best_state is not None:
        model.load_state_dict(best_state)
    if not args.skip_test:
        test_rows = load_jsonl(test_file)
        test_texts = [build_text(r, args.use_title) for r in test_rows]
        test_labels = [r["label_id"] for r in test_rows]
        test_dl = DataLoader(BertDataset(test_texts, test_labels, tok, args.max_length),
                             batch_size=args.eval_batch_size, shuffle=False,
                             collate_fn=make_collate_fn(tok))
        test_rep = evaluate(model, test_dl, device, id2name)
        logger.info("=" * 70)
        logger.info(f"TEST（best epoch {best_epoch}, val macro_f1={best_f1:.4f}）:")
        logger.info(f"  acc={test_rep['acc']} macro_f1={test_rep['macro_f1']} "
                    f"weighted_f1={test_rep['weighted_f1']}")
        for name, v in test_rep["per_class"].items():
            if v:
                logger.info(f"    {name}: P={v['P']:.4f} R={v['R']:.4f} F1={v['F1']:.4f} (n={v['support']})")
        logger.info(f"  混淆矩阵（行=真实 高/中/低，列=预测）: {test_rep['confusion_matrix']}")

    # ── 结果 JSON（调度器与汇总表直接读这里）──
    result = {"run_name": stem,
              "started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t_start)),
              "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
              "total_seconds": round(time.time() - t_start, 1),
              "device": device, "out_root": str(out_root),
              "args": json.loads(json.dumps(vars(args), default=str)),
              "train_file": str(train_file), "train_n": len(train_rows),
              "train_sha256": data_sha256,
              "train_dist": {id2name[i]: c for i, c in enumerate(counts)},
              "class_weights": None if weight is None else [round(float(x), 4) for x in weight.tolist()],
              "loss": crit_desc, "epochs": epochs_log,
              "best_epoch": best_epoch, "best_val_macro_f1": best_f1,
              "test": test_rep}
    json_path = log_dir / f"{stem}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    logger.info(f"日志   → {log_path}")
    logger.info(f"指标   → {json_path}")
    logger.info(f"ckpt   → {'（--no_save 未落盘）' if args.no_save else str(ckpt_dir / stem / 'best.pt')}")
    logger.info(f"总耗时 {time.time() - t_start:.0f}s（best epoch {best_epoch}, "
                f"val macro_f1={best_f1:.4f}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
