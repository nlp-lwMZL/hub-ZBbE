"""BiLSTM + CRF 事件抽取训练脚本。

只用训练集 + 验证集：训练集更新参数，验证集选最优模型并 early stopping。
测试集不参与本脚本（后续评估脚本再单独用）。

示例
----
# 默认（自由解码）
python -m extract.bilstm_crf.train

# 开启解码期 BIO 硬约束
python -m extract.bilstm_crf.train --constrained

# 快速调试：每集只取 500 条、2 个 epoch、短窗口
python -m extract.bilstm_crf.train --limit 500 --epochs 2 --max-len 256 --stride 256

输出目录：extract/output/bilstm/<每次运行子文件夹>/，内含日志、词表、最优模型与验证报告。
子文件夹默认用时间戳，也可用 --run-name 自定义（如 --run-name baseline_v1）。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path

import torch

# 允许以 `python extract/bilstm_crf/train.py` 直接运行
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.data import (  # noqa: E402
    EventBIODataset,
    build_dataloader,
    build_vocab,
    compute_label_counts,
    extract_spans,
    load_records,
    make_chunks,
)
from common.label import BIO_LABELS, NUM_BIO_LABELS  # noqa: E402
from common.logging_utils import setup_logger  # noqa: E402
from common.trainer import Trainer, TrainerConfig, evaluate, load_checkpoint  # noqa: E402

try:
    from .model import BiLSTMCRF, BiLSTMSoftmax  # noqa: E402  (python -m 方式)
except ImportError:  # 直接 python extract/bilstm_crf/train.py 运行
    from model import BiLSTMCRF, BiLSTMSoftmax  # type: ignore  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BiLSTM+CRF 事件抽取训练")
    # 数据
    p.add_argument("--train-file", default=str(PROJECT_ROOT / "data/dataset/split/train.jsonl"))
    p.add_argument("--val-file", default=str(PROJECT_ROOT / "data/dataset/split/val.jsonl"))
    p.add_argument("--max-len", type=int, default=512, help="滑窗最大字符数")
    p.add_argument("--stride", type=int, default=384, help="滑窗步长（<max-len 时重叠）")
    p.add_argument("--min-freq", type=int, default=2, help="词表最小字符频次")
    p.add_argument("--limit", type=int, default=0, help="每集只取前 N 条（调试）")

    # 模型
    p.add_argument("--embed-dim", type=int, default=128)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--num-layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--constrained", action="store_true",
                   help="开启解码期 BIO 硬约束（仅影响 Viterbi 解码）")
    p.add_argument("--no-crf", action="store_true",
                   help="消融用：去掉 CRF，改用逐 token 交叉熵 + argmax/约束解码")
    p.add_argument("--class-weight", default="none", choices=["none", "inverse"],
                   help="类别不平衡：inverse=按标签频率倒数初始化 emission 偏置")
    p.add_argument("--prior-scale", type=float, default=1.0,
                   help="inverse 先验的缩放系数（越大越偏向稀有标签）")

    # 训练
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument(
        "--monitor", default="f1",
        choices=["f1", "event_exact", "type_macro_f1"],
        help="选优/早停依据：f1=overall F1；event_exact=事件级完全匹配率；type_macro_f1=9 类事件宏平均 F1",
    )
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--lr-scheduler", default="none", choices=["none", "plateau", "cosine"],
                   help="学习率调度：plateau=按 val_F1 衰减，cosine=余弦退火")
    p.add_argument("--warmup-steps", type=int, default=0,
                   help="线性 warmup 步数：前 N 步 lr 从 ~0 升到 --lr（0 关闭）")
    p.add_argument("--lr-patience", type=int, default=2, help="plateau 调度的耐心轮数")
    p.add_argument("--min-lr", type=float, default=1e-6, help="调度器学习率下限")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])

    # 输出：每次运行在 output-root 下新建一个独立子文件夹
    p.add_argument("--output-root", default=str(PROJECT_ROOT / "extract/output/bilstm"),
                   help="输出根目录；每次运行在其下新建子文件夹")
    p.add_argument("--run-name", default="",
                   help="本次运行的子文件夹名，默认为时间戳")
    p.add_argument("--log-every", type=int, default=50, help="每 N 个 step 打印一次 batch loss")
    return p.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_run_name(args: argparse.Namespace, timestamp: str) -> str:
    """生成带关键配置摘要的 run 名，如 20261123_184236_cw-inv_plat_constr。"""
    if args.run_name:
        return args.run_name
    cw = "cw-inv" if args.class_weight == "inverse" else "base"
    ls = {"none": "nosched", "plateau": "plat", "cosine": "cos"}[args.lr_scheduler]
    con = "constr" if args.constrained else "plain"
    warm = f"_w{args.warmup_steps}" if args.warmup_steps > 0 else ""
    head = "nocrf" if args.no_crf else "crf"
    return f"{timestamp}_{cw}_{ls}_{head}_{con}{warm}"


EXPERIMENT_FIELDS = [
    "run_name", "start_time", "class_weight", "prior_scale", "lr",
    "lr_scheduler", "warmup_steps", "constrained",
    "hidden_dim", "num_layers", "dropout",
    "best_epoch", "best_val_f1", "event_exact",
]


def append_experiment_row(path: Path, row: dict) -> None:
    """向根目录 experiments.csv 追加一行汇总（不存在则先写表头）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=EXPERIMENT_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    start_time = time.strftime("%Y-%m-%d %H:%M:%S")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = make_run_name(args, timestamp)
    if args.monitor != "f1":
        run_name += {"event_exact": "_m-exact", "type_macro_f1": "_m-macro"}[args.monitor]
    run_dir = Path(args.output_root) / run_name
    if run_dir.exists() and args.run_name:  # 显式命名重复时不覆盖旧实验
        run_name = f"{args.run_name}_{timestamp}"
        run_dir = Path(args.output_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    log_file = run_dir / f"bilstm_crf_{run_name}.log"
    logger = setup_logger(log_file)

    # 完整参数快照
    (run_dir / "config.json").write_text(
        json.dumps(
            {**vars(args), "run_name": run_name, "start_time": start_time},
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )

    logger.info("=" * 70)
    logger.info("BiLSTM + CRF 事件抽取训练")
    logger.info("run 名称: %s", run_name)
    logger.info("输出目录: %s", run_dir)
    logger.info("参数: %s", json.dumps(vars(args), ensure_ascii=False))
    logger.info("设备: %s", device)
    logger.info("解码约束: %s", "开启（BIO 硬约束）" if args.constrained else "关闭")
    logger.info("模型类型: %s", "BiLSTM+Softmax（无 CRF）" if args.no_crf else "BiLSTM+CRF")

    # ---------- 数据 ----------
    logger.info("加载数据 ...")
    train_records = load_records(args.train_file, limit=args.limit)
    val_records = load_records(args.val_file, limit=args.limit)
    logger.info("训练集 %d 条，验证集 %d 条", len(train_records), len(val_records))

    train_chunks = []
    for rec in train_records:
        train_chunks.extend(
            make_chunks(rec.get("text") or "", extract_spans(rec),
                        max_len=args.max_len, stride=args.stride,
                        source=str(rec.get("source", "")))
        )
    vocab = build_vocab(train_chunks, min_freq=args.min_freq)
    logger.info("词表大小: %d（字符级）", len(vocab))

    train_ds = EventBIODataset(train_records, vocab, max_len=args.max_len, stride=args.stride)
    val_ds = EventBIODataset(val_records, vocab, max_len=args.max_len, stride=args.stride)
    logger.info("切块后：训练 %d 块，验证 %d 块", len(train_ds), len(val_ds))

    train_loader = build_dataloader(
        train_ds, batch_size=args.batch_size, shuffle=True, pad_id=vocab.pad_id, seed=args.seed
    )
    val_loader = build_dataloader(
        val_ds, batch_size=args.batch_size, shuffle=False, pad_id=vocab.pad_id
    )

    # ---------- 类别不平衡：标签先验偏置 ----------
    label_prior = None
    if args.class_weight == "inverse":
        counts = compute_label_counts(train_ds)
        total = max(sum(counts), 1)
        inv = [total / (c + 1) for c in counts]          # 频率越低权重越大
        mean_inv = sum(inv) / len(inv)
        # log 归一化并裁剪到 [-3, 3]，避免极稀有标签偏置过强
        label_prior = [
            max(-3.0, min(3.0, math.log(x / mean_inv))) * args.prior_scale
            for x in inv
        ]
        preview = sorted(
            zip(BIO_LABELS, label_prior), key=lambda kv: kv[1], reverse=True
        )[:5]
        logger.info(
            "类别不平衡: inverse 先验 scale=%.2f，偏置最高 5 个标签: %s",
            args.prior_scale,
            json.dumps([(k, round(v, 3)) for k, v in preview], ensure_ascii=False),
        )

    # ---------- 模型 ----------
    model_cls = BiLSTMSoftmax if args.no_crf else BiLSTMCRF
    model = model_cls(
        vocab_size=len(vocab),
        num_tags=NUM_BIO_LABELS,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        pad_id=vocab.pad_id,
        constrained=args.constrained,
        label_prior=label_prior,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("模型可训练参数: %d", n_params)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # ---------- 学习率调度 ----------
    scheduler = None
    if args.lr_scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5,
            patience=args.lr_patience, min_lr=args.min_lr,
        )
    elif args.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.min_lr,
        )
    logger.info(
        "学习率调度: %s（初始 lr=%.2e，warmup=%s）",
        args.lr_scheduler, args.lr,
        f"{args.warmup_steps} step" if args.warmup_steps > 0 else "关闭",
    )

    # ---------- 训练 ----------
    ckpt_dir = run_dir
    vocab.save(ckpt_dir / "vocab.json")
    logger.info("词表已保存: %s", ckpt_dir / "vocab.json")

    config = TrainerConfig(
        epochs=args.epochs,
        patience=args.patience,
        ckpt_dir=str(ckpt_dir),
        ckpt_prefix="bilstm_crf",
        monitor=args.monitor,
    )
    trainer = Trainer(
        model, optimizer, device, config, logger=logger,
        scheduler=scheduler, warmup_steps=args.warmup_steps,
        log_every=args.log_every,
    )
    result = trainer.fit(train_loader, val_loader)

    logger.info("=" * 70)
    logger.info("训练完成：最优 epoch=%d，%s=%.4f，val_F1=%.4f",
                result["best_epoch"], result["monitor"], result["best_score"], result["best_f1"])
    logger.info("最优模型: %s", result["best_ckpt"])

    # 用最优权重复评一次并打印完整指标
    load_checkpoint(model, result["best_ckpt"], device)
    final_loss, final_report = evaluate(model, val_loader, device)
    logger.info("最终验证: val_loss=%.4f | %s", final_loss, final_report.summary())
    logger.info("各槽位 F1: %s", json.dumps(final_report.as_dict()["per_type_f1"], ensure_ascii=False))
    et = final_report.as_dict().get("event_type")
    if et:
        logger.info("事件类型推断: %s", json.dumps(et, ensure_ascii=False))
    em = final_report.as_dict().get("event_match")
    if em:
        logger.info("事件级完全匹配: %s", json.dumps(em, ensure_ascii=False))

    report_path = ckpt_dir / "val_report.json"
    report_path.write_text(
        json.dumps(
            {
                "args": vars(args),
                "best_epoch": result["best_epoch"],
                "best_val_f1": result["best_f1"],
                "val_report": final_report.as_dict(),
                "history": [
                    {
                        "epoch": h["epoch"],
                        "train_loss": h["train_loss"],
                        "val_loss": h["val_loss"],
                        "val_f1": h["val_f1"],
                    }
                    for h in result["history"]
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("验证报告已保存: %s", report_path)

    # 汇总到根目录 experiments.csv，便于后期横向对比
    append_experiment_row(
        Path(args.output_root) / "experiments.csv",
        {
            "run_name": run_name,
            "start_time": start_time,
            "class_weight": args.class_weight,
            "prior_scale": args.prior_scale if args.class_weight == "inverse" else "",
            "lr": args.lr,
            "lr_scheduler": args.lr_scheduler,
            "warmup_steps": args.warmup_steps,
            "constrained": args.constrained,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "best_epoch": result["best_epoch"],
            "best_val_f1": round(result["best_f1"], 4),
            "event_exact": round(em["accuracy"], 4) if em else "",
        },
    )
    logger.info("实验汇总: %s", Path(args.output_root) / "experiments.csv")
    logger.info("日志文件: %s", log_file)


if __name__ == "__main__":
    main()
