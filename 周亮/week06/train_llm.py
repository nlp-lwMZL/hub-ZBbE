"""
LLM 指令微调（SFT）：全量微调 / LoRA / QLoRA 三模式，共用一套训练循环

参照老师 llm_lora/train_lora.py，教学重点原样保留：
  1. SFT 的关键不是模型，而是 chat 模板、Loss Masking、数据格式
     （模板渲染统一走 dataset.render_messages，训练与评测不可能错位）。
  2. 两种微调模式的成本差一个量级：
       lora   只训低秩旁路（~0.4% 参数），adapter 几十 MB
       full   全参数更新，T4 上限 1.7B：fp16 主权重 + 静态 loss scaling + adamw8bit
              + 梯度检查点（fp16 梯度进不了 GradScaler.unscale_，见 unscale_and_check）
  3. Qwen3 混合思考模型（0.6B/1.7B）必须 enable_thinking=False（渲染层已统一处理）。

与我们 BERT 管道同构的部分（复用已有纪律）：
  · 数据：dataset.LLMSFTDataset 与 BertDataset 同一份 jsonl、同一 split、同一 seed
  · 指标：metrics.classification_report（主指标 macro-F1），logprob 打分而非生成
  · 产物：logs/{run_name}.log + 同名 .json；ckpt 落 CKPT_DIR/{run_name}/
  · 每 epoch 在 val 上打分选最佳 epoch（老师只看 val_loss，我们用 macro-F1，
    与 BERT 的选点口径一致），test 只在最后报一次

用法（本机 CPU 冒烟，用 0.6B）：
  python train_llm.py --model qwen3-0.6b --mode lora --max_steps 3 --val_n 24 --skip_test
  python train_llm.py --model qwen3-0.6b --mode full --max_steps 2 --val_n 16 --skip_test

用法（Colab T4，SFT 三臂对比，底座默认 HuggingFace 优先）：
  # ① 0.6B LoRA —— 同尺寸对照的 LoRA 臂
  python train_llm.py --model qwen3-0.6b --mode lora
  # ② 0.6B full —— 与 ① 只差「训多少参数」，变量唯一；fp32 主权重，显存宽松
  python train_llm.py --model qwen3-0.6b --mode full
  # ③ 1.7B LoRA —— 与 ① 只差参数量（0.6B → 1.7B），看规模收益
  python train_llm.py --model qwen3-1.7b --mode lora
  # ④⑤ 数据量消融：用 BERT 同一份嵌套子集 train_10k（不能用 --num_train 代替）
  python train_llm.py --model qwen3-0.6b --mode lora --train_file data/ablation/train_10k.jsonl
  python train_llm.py --model qwen3-0.6b --mode full --train_file data/ablation/train_10k.jsonl

  # 可选：4B 规模探针（QLoRA，需另下 ~8GB 底座）
  python train_llm.py --model qwen3-4b-instruct-2507 --mode qlora
"""

import argparse
import hashlib
import json
import logging
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import paths
from dataset import (LLMSFTDataset, load_label_map, load_jsonl, make_pad_collate_fn,
                    stratified_subset, render_chat_prompt, seed_everything)
from metrics import classification_report, brief, to_markdown_table, format_confusion
from model_llm import (REGISTRY, build_optimizer, describe_device, label_first_tokens,
                       load_base_model, logprob_classify, pick_amp, preflight_memory,
                       resolve_optimizer, wrap_peft)

HERE = paths.HERE


# ─────────────────────────────────────────────────────────────────────────────
# 参数
# ─────────────────────────────────────────────────────────────────────────────

def build_arg_parser():
    ap = argparse.ArgumentParser(description="LLM SFT：全量微调 / LoRA / QLoRA")
    ap.add_argument("--model", default="qwen3-1.7b",
                    help=f"注册表键（{list(REGISTRY)}）或本地目录")
    ap.add_argument("--source", default="auto", choices=["auto", "hf", "ms"],
                    help="底座下载源：auto=Colab走HuggingFace/本机走ModelScope；"
                         "也可用环境变量 WEEK6_LLM_SOURCE")
    ap.add_argument("--mode", choices=["lora", "full", "qlora"], default="lora")
    ap.add_argument("--train_file", default=None,
                    help="缺省 data/train_5k.jsonl；数据量消融用 "
                         "data/ablation/train_10k.jsonl（与 BERT 同一嵌套抽样链）")
    ap.add_argument("--val_file", default=None, help="缺省 data/val_2k.jsonl")
    ap.add_argument("--test_file", default=None, help="缺省 data/test_3k.jsonl（--skip_test 时不用）")
    ap.add_argument("--num_train", type=int, default=None,
                    help="train 截取条数（等距分层，不掐头）。**缺省 = 用 --train_file 的全部行**："
                         "传 10k 文件就是 10000 条；显式给值只能 ≤ 文件行数")
    ap.add_argument("--val_n", type=int, default=500, help="每 epoch 打分的 val 条数")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=None,
                    help="缺省按模式：lora/qlora=4，full=2")
    ap.add_argument("--grad_accum", type=int, default=None,
                    help="缺省按模式：lora/qlora=8，full=16")
    ap.add_argument("--lr", type=float, default=None,
                    help="缺省按模式：lora/qlora=2e-4，full=2e-5（全量必须小 lr，否则冲掉预训练知识）")
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--lora_targets", choices=["qkvo", "all"], default="qkvo",
                    help="qkvo=注意力四投影（老师口径）；all=+MLP 三投影（消融用）")
    ap.add_argument("--optimizer", choices=["adamw", "adamw8bit"], default=None,
                    help="缺省按模式：full=adamw8bit（T4 显存关键），lora/qlora=adamw")
    ap.add_argument("--master_dtype", choices=["auto", "fp32", "fp16"], default="auto",
                    help="full 模式的主权重精度：auto=fp32（正确，但 1.7B 在 T4 上装不下）；"
                         "fp16=省一半显存，代价是更新精度只有 ~1e-5（loss 可能压着不动）")
    ap.add_argument("--max_length", type=int, default=384,
                    help="prompt+答案的 token 上限（title+desc 平均 ~230，384 近无损）")
    ap.add_argument("--use_title", action="store_true", default=True)
    ap.add_argument("--no_title", dest="use_title", action="store_false",
                    help="只用 description（与 BERT 的 desc 口径对齐）")
    ap.add_argument("--no_gc", action="store_true",
                    help="关梯度检查点（缺省 full/qlora 开、lora 关）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_steps", type=int, default=0, help=">0 时冒烟：只跑这么多优化步")
    ap.add_argument("--no_save", action="store_true")
    ap.add_argument("--skip_test", action="store_true")
    ap.add_argument("--run_suffix", default="", help="run_name 附加后缀（避免特殊配置重名）")
    return ap


def resolve_defaults(a):
    if a.batch_size is None:
        a.batch_size = 2 if a.mode == "full" else 4
    if a.grad_accum is None:
        a.grad_accum = 16 if a.mode == "full" else 8
    if a.lr is None:
        a.lr = 2e-5 if a.mode == "full" else 2e-4
    if a.optimizer is None:
        a.optimizer = "adamw8bit" if a.mode == "full" else "adamw"
    return a


def model_short(name: str) -> str:
    if name in REGISTRY:
        return name
    return Path(name).name or name


def build_run_name(a, n_train: int) -> str:
    parts = [f"llm-{model_short(a.model)}", f"mode-{a.mode}"]
    # 非默认训练集要写进 run_name：同样 n=10000 也可能来自不同的抽样/文件，名字里说清楚
    if a.train_file:
        parts.append(f"data-{Path(a.train_file).stem}")
    if a.mode in ("lora", "qlora"):
        parts.append(f"r{a.lora_r}a{a.lora_alpha}tgt-{a.lora_targets}")
    if a.mode == "full" and getattr(a, "master_dtype", "auto") == "fp16":
        parts.append("master-fp16")          # 与 fp32 主权重是两个不同的实验，不能同名
    parts += [f"len-{a.max_length}", f"text-{'titledesc' if a.use_title else 'desc'}",
              f"n-{n_train}", f"bs-{a.batch_size}ga{a.grad_accum}",
              f"lr-{a.lr:g}", f"ep-{a.epochs}", f"seed-{a.seed}"]
    name = "_".join(parts)
    if a.max_steps:
        name += "_smoke"
    if a.run_suffix:
        name += f"_{a.run_suffix}"
    return name


def unique_run_name(base: str) -> str:
    """同名参数重跑不覆盖：加 _r2/_r3（与 train_bert 同规则）"""
    if not (paths.LOG_DIR / f"{base}.json").exists():
        return base
    k = 2
    while paths.LOG_DIR / f"{base}_r{k}.json" .exists():
        k += 1
    return f"{base}_r{k}"


def unscale_and_check(params, scale: float) -> bool:
    """
    fp16 主权重路径的「静态 loss scaling」还原：梯度先除以 scale，再检查 inf/nan。

    为什么不能直接用 GradScaler：torch 的 GradScaler.unscale_() 只接受 **fp32 梯度**；
    full + --master_dtype fp16 时参数本身就是 fp16，反向得到的梯度也是 fp16，
    scaler.step() 第一次调用就抛 "Attempting to unscale FP16 gradients."。
    返回 True 表示这批梯度不可用，调用方必须跳过本次更新（一次 nan 会污染全部权重）。
    """
    bad = False
    for p in params:
        if p.grad is None:
            continue
        p.grad.div_(scale)
        if not bool(torch.isfinite(p.grad).all()):
            bad = True
    return bad


# ─────────────────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = resolve_defaults(build_arg_parser().parse_args())
    paths.ensure_dirs()
    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    name2id, id2name, label_names = load_label_map(paths.DATA_DIR)
    train_path = Path(args.train_file) if args.train_file else paths.DATA_DIR / "train_5k.jsonl"
    val_path = Path(args.val_file) if args.val_file else paths.DATA_DIR / "val_2k.jsonl"
    test_path = Path(args.test_file) if args.test_file else paths.DATA_DIR / "test_3k.jsonl"

    train_rows = load_jsonl(train_path)
    # 缺省用整个文件：传 --train_file data/ablation/train_10k.jsonl 时绝不能被旧的
    # --num_train 5000 默认值偷偷砍回 5k（那会让「数据量消融」白跑）。
    if args.num_train is None:
        args.num_train = len(train_rows)
    elif args.num_train > len(train_rows):
        raise SystemExit(
            f"--num_train={args.num_train} 超过 {train_path} 的 {len(train_rows)} 行。"
            f"要扩训练集请传 --train_file data/ablation/train_10k.jsonl（BERT 用的同一份），"
            f"不能用 --num_train 从 5k 里抽出更多数据。")
    if args.num_train < len(train_rows):
        train_rows = stratified_subset(train_rows, args.num_train)
    val_rows = stratified_subset(load_jsonl(val_path), args.val_n)

    run_base = build_run_name(args, len(train_rows))
    run_name = unique_run_name(run_base)

    logger = logging.getLogger("train_llm")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(message)s", "%H:%M:%S")
    for h in (logging.FileHandler(paths.LOG_DIR / f"{run_name}.log", encoding="utf-8"),
              logging.StreamHandler()):
        h.setFormatter(fmt)
        logger.addHandler(h)
    logger.info(f"run_name = {run_name}")
    logger.info(f"device={device} | {describe_device(device)}")
    data_sha256 = hashlib.sha256(train_path.read_bytes()).hexdigest()
    logger.info(f"train={len(train_rows)}（{train_path.name}） val={len(val_rows)}"
                f" | mode={args.mode} | labels={label_names}")
    logger.info(f"训练数据 sha256={data_sha256[:16]}…（与 BERT 数据量消融共用同一份文件时可核对此值）")
    logger.info(f"{paths.summary()}")

    # ── 模型：三种模式，同一套后续流程 ──
    model, tok, meta = load_base_model(args.model, args.mode, device, logger,
                                       source=args.source, master_dtype=args.master_dtype)
    if args.mode in ("lora", "qlora"):
        if args.mode == "qlora":
            from peft import prepare_model_for_kbit_training
            model = prepare_model_for_kbit_training(model)
        model = wrap_peft(model, args.lora_r, args.lora_alpha, args.lora_dropout,
                          args.lora_targets, logger)
    if not args.no_gc and args.mode in ("full", "qlora"):
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        logger.info("梯度检查点开（时间换显存）")

    # ── 显存预算：**在 model.to(cuda) 之前**算清楚 ──
    # 参数侧的账是确定的（权重+梯度+优化器状态），算完就知道这张卡能不能装下。
    # 放到 .to(cuda) 之前是为了「装不下就直接说」，而不是占满显存跑三分钟再 OOM。
    meta.update({"model": model, "name": args.model})
    opt_name = resolve_optimizer(args.optimizer, logger)
    preflight_memory(meta, args.mode, opt_name, device, logger,
                     batch_size=args.batch_size, max_length=args.max_length,
                     master_dtype=args.master_dtype)
    if args.mode != "qlora":                 # qlora 加载时已经 device_map 上卡
        model = model.to(device)

    amp_dtype, use_scaler = pick_amp(device, logger)

    # ── fp16 主权重下 GradScaler 不可用（PyTorch 硬限制）──
    # GradScaler.unscale_() 只支持 fp32 梯度；--master_dtype fp16 时参数/梯度都是 fp16，
    # 不改的话第一步更新就抛 "Attempting to unscale FP16 gradients."。
    # 这里改成静态 loss scaling：前向把 loss 放大 loss_scale，反向拿到放大后的梯度，
    # step 前再除回来。没有 scaler 的自适应，但足以避免 lr=2e-5 的小梯度直接下溢成 0。
    fp16_master = (args.mode == "full" and device == "cuda"
                   and next(model.parameters()).dtype == torch.float16)
    loss_scale = 1.0
    if fp16_master and use_scaler:
        use_scaler = False
        loss_scale = 128.0
        logger.info(f"fp16 主权重 → 关闭 GradScaler（它只能 unscale fp32 梯度），"
                    f"改用静态 loss scaling ×{loss_scale:g}")
    logger.info(f"autocast={str(amp_dtype).split('.')[-1]} scaler={use_scaler} "
                f"| loss_scale={loss_scale:g}")

    # ── 数据 ──
    train_ds = LLMSFTDataset(train_rows, tok, max_length=args.max_length,
                             use_title=args.use_title, label_names=label_names)
    collate = make_pad_collate_fn(tok)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          collate_fn=collate, drop_last=False)

    # prompt 长度统计：决定 max_length 是否截得太多（EDA 的延续）。
    # 等距抽样——文件按类分组存放，掐头 300 条会全是「高」类，分布失真。
    with torch.no_grad():
        stat_rows = stratified_subset(train_rows, 300)
        sample_prompts = [render_chat_prompt(tok, r, args.use_title) for r in stat_rows]
        lens = [len(x) for x in tok(sample_prompts, add_special_tokens=False)["input_ids"]]
    lens.sort()
    stats = {"mean": round(sum(lens) / len(lens), 1), "p95": lens[int(len(lens) * 0.95)],
             "max": lens[-1], "truncation_pct": round(
                 sum(l > args.max_length for l in lens) / len(lens) * 100, 2)}
    logger.info(f"prompt token 长度: mean={stats['mean']} p95={stats['p95']} "
                f"max={stats['max']} | >max_length({args.max_length}) 占 {stats['truncation_pct']}%")

    # val/test 的打分 prompt 预渲染一次；label 首 token 打分表
    val_prompts = [render_chat_prompt(tok, r, args.use_title) for r in val_rows]
    label_ids = label_first_tokens(tok, label_names, logger)
    val_y = [name2id[r["label"]] for r in val_rows]

    opt, opt_used = build_optimizer(model, args.mode, args.lr, opt_name, logger)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    opt_params = [p for p in model.parameters() if p.requires_grad]

    def score(prompts, truth):
        preds, probs = logprob_classify(model, tok, prompts, label_ids, device,
                                        batch_size=max(8, args.batch_size * 2),
                                        amp_dtype=amp_dtype if device == "cuda" else None,
                                        logger=logger)
        return classification_report(truth, preds, probs=probs, id2name=id2name)

    # ── 训练循环 ──
    epochs_log, best_f1, best_epoch, t0 = [], -1.0, 0, time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running, n_b, step = 0.0, 0, 0
        ep_t = time.time()
        for i, batch in enumerate(train_dl):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=amp_dtype, enabled=(device == "cuda")):
                raw_loss = model(**batch).loss
            # 梯度累积后统一缩放：loss_scale 只影响反向数值，日志仍打印真实 loss
            loss = raw_loss / args.grad_accum * loss_scale
            scaler.scale(loss).backward()
            running += raw_loss.item()
            n_b += 1
            if (i + 1) % args.grad_accum == 0 or (i + 1) == len(train_dl):
                skipped = loss_scale != 1.0 and unscale_and_check(opt_params, loss_scale)
                if skipped:
                    logger.info("  !! 梯度出现 inf/nan，跳过本次更新")
                else:
                    scaler.step(opt)
                    scaler.update()
                opt.zero_grad(set_to_none=True)
                step += 1
            if i % 50 == 0:
                logger.info(f"  ep{epoch} {i}/{len(train_dl)} loss={raw_loss.item():.4f}")
            if args.max_steps and step >= args.max_steps:
                break

        rep = score(val_prompts, val_y)
        is_best = rep["macro_f1"] > best_f1
        if is_best:
            best_f1, best_epoch = rep["macro_f1"], epoch
        epochs_log.append({"epoch": epoch,
                           "train_loss": round(running / max(1, n_b), 4),
                           "steps": step, "seconds": round(time.time() - ep_t, 1),
                           "val": rep, "is_best": is_best})
        logger.info(f"epoch {epoch}: loss={epochs_log[-1]['train_loss']} "
                    f"| val {brief(rep, id2name)}"
                    + ("  ← best" if is_best else ""))

        # 最佳 epoch 才落盘（full 一份 3.4GB，不每次都写 Drive）
        if is_best and not args.no_save:
            ck = paths.CKPT_DIR / run_name
            ck.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(ck))
            # 全量 ckpt 自带 tokenizer：否则评测时从 ckpt 目录读到空 tokenizer，
            # 标签会被切成 0 个 token（见 predict_llm.py 的 full_weights 分支）。
            tok.save_pretrained(str(ck))
            (ck / "train_args.json").write_text(
                json.dumps({**vars(args), "model_dir": str(args.model), "run_name": run_name,
                            "train_file": str(train_path), "train_n": len(train_rows),
                            "train_sha256": data_sha256},
                           ensure_ascii=False, indent=2, default=str), encoding="utf-8")
            logger.info(f"  保存最佳 ckpt → {ck}")
        if args.max_steps:
            break

    # ── test：所有配置定死后只报一次 ──
    test_rep = None
    if not args.skip_test:
        test_rows = load_jsonl(test_path)
        test_prompts = [render_chat_prompt(tok, r, args.use_title) for r in test_rows]
        test_y = [name2id[r["label"]] for r in test_rows]
        test_rep = score(test_prompts, test_y)
        test_rep["file"] = test_path.name
        logger.info(f"TEST（{len(test_rows)} 条）: acc={test_rep['acc']} "
                    f"macro_f1={test_rep['macro_f1']}")
        logger.info("混淆矩阵（行为真实，列为预测）:\n" + format_confusion(test_rep, id2name))

    total_min = (time.time() - t0) / 60
    logger.info(f"完成：best_epoch={best_epoch} val_macro_f1={best_f1} | 总耗时 {total_min:.1f} 分钟")

    out = {"run_name": run_name, "model": args.model, "mode": args.mode,
           "optimizer_used": opt_used,
           "train_file": str(train_path), "train_n": len(train_rows),
           "train_sha256": data_sha256,
           "args": json.loads(json.dumps(vars(args), default=str)),
           "prompt_token_stats": stats,
           "epochs": epochs_log, "best_epoch": best_epoch,
           "best_val_macro_f1": best_f1, "test": test_rep,
           "minutes": round(total_min, 1)}
    (paths.LOG_DIR / f"{run_name}.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"日志 → {paths.LOG_DIR / (run_name + '.json')}")

    if test_rep:
        print("\n" + to_markdown_table({run_name: test_rep}, ece_header="ECE·LLM"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
