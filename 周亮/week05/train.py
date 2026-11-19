"""train.py — SFT 训练脚本（所有超参走 argparse，附默认值）

两种起点（Q1/Q10 共识）：
    --init_from ../../week4/tf_mask.pt   默认。加载 week4 预训练权重（字表/位置编码按行前缀扩展）
    --init_from none                     从头训（对比实验，用同一份扩展字表）

学习率智能默认（Q7）：SFT 要防灾难性遗忘（课件 slide 31），继续训用小 lr——
    加载预训练权重 → 3e-4；从头训 → 1e-3（与 week4 一致）。显式传 --lr 则覆盖。

loss（Q4/Q5 共识）：交叉熵 ignore_index=-100，只有回答段（output + <|end|>）
参与训练与 PPL；train loss 与 val PPL 同口径（都按回答段 token 平均），可直接对照。

用法：
    python train.py                                          # 预训练起点 → best_sft.pt
    python train.py --init_from none --save best_scratch.pt  # 从头训对比

每次运行的完整配置与逐 epoch 日志自动写入 logs/<run_name>.json
（--run_name 可指定名字，默认按时间戳；每个 epoch 落盘一次，中途崩溃也保留已完成记录）。
"""
import argparse
import json
import math
import os
import random
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import DEFAULT_TRAIN, DEFAULT_VAL, SFTDataset, build_vocab
from model import WEEK4_CKPT, TransformerLM, load_pretrained

HERE = os.path.dirname(os.path.abspath(__file__))


def evaluate_ppl(model, loader, device, pad_id):
    """验证集 PPL：只统计 y != -100 的位置（回答段），reduction=sum 按token加权（Q5）"""
    model.eval()
    total, tokens = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pad_mask = x == pad_id
            logits = model(x, pad_mask)
            total += float(F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                           y.reshape(-1), ignore_index=-100,
                                           reduction="sum"))
            tokens += int((y != -100).sum())
    if tokens == 0:
        return float("inf"), float("inf")
    loss = total / tokens
    return loss, math.exp(loss)


def main():
    ap = argparse.ArgumentParser(description="week5 SFT 训练")
    ap.add_argument("--train_data", default=DEFAULT_TRAIN)
    ap.add_argument("--val_data", default=DEFAULT_VAL)
    ap.add_argument("--init_from", default=WEEK4_CKPT,
                    help="预训练 ckpt 路径；传 none 则从头训练")
    ap.add_argument("--save", default="best_sft.pt")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=None,
                    help="不传则按起点自动选：预训练 3e-4 / 从头 1e-3")
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--d_model", type=int, default=192)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--clip", type=float, default=1.0, help="梯度裁剪阈值")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log_dir", default="logs", help="训练日志目录（相对本脚本目录）")
    ap.add_argument("--run_name", default=None,
                    help="本次运行的日志文件名（不含扩展名）；默认按时间戳生成")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cpu")  # 本机 Intel CPU，MPS 不可用
    init_pretrained = args.init_from != "none"
    if args.lr is None:
        args.lr = 3e-4 if init_pretrained else 1e-3

    char2idx, idx2char, special_ids = build_vocab(args.train_data, args.val_data)
    V = len(char2idx)
    pad_id = special_ids["<pad>"]

    tr = SFTDataset(args.train_data, char2idx, special_ids, args.max_len)
    va = SFTDataset(args.val_data, char2idx, special_ids, args.max_len)
    # drop_last=False：SFT 数据只有 3000 条，末批 24 条也是有效监督，不丢
    train_loader = DataLoader(tr, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(va, batch_size=args.batch_size, shuffle=False)

    model = TransformerLM(V, args.d_model, args.heads, args.layers,
                          args.max_len, args.dropout).to(device)
    if init_pretrained:
        load_pretrained(model, args.init_from)
    n_par = sum(p.numel() for p in model.parameters())
    src = os.path.basename(args.init_from) if init_pretrained else "从头训练（随机初始化）"
    print(f"device={device}  起点={src}  lr={args.lr}  参数量 {n_par/1e6:.2f}M  "
          f"样本 train={len(tr)} val={len(va)}  epochs={args.epochs} batch={args.batch_size}")

    # ── 训练日志：完整配置 + 逐 epoch 记录，每个 epoch 落盘一次（崩溃也有已完成部分）──
    log_dir = args.log_dir if os.path.isabs(args.log_dir) else os.path.join(HERE, args.log_dir)
    os.makedirs(log_dir, exist_ok=True)
    run_name = args.run_name or time.strftime("train_%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"{run_name}.json")
    log = {
        "run_name": run_name,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "env": {"torch": torch.__version__, "device": str(device)},
        "config": {**vars(args), "lr_resolved": args.lr, "init_source": src,
                   "n_params": n_par, "vocab_size": V,
                   "n_train": len(tr), "n_val": len(va)},
        "sft_before_val_ppl": None,
        "epochs": [],
        "best_val_ppl": None,
        "checkpoint": args.save,
        "finished_at": None,
    }

    def flush():
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(log, f, ensure_ascii=False, indent=2)

    _, ppl0 = evaluate_ppl(model, val_loader, device, pad_id)
    log["sft_before_val_ppl"] = ppl0
    flush()
    print(f"日志: {log_path}")
    print(f"SFT 前验证 PPL（仅回答段）: {ppl0:.2f}")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    best = float("inf")
    print(f"\n{'Epoch':>6} {'TrainLoss':>10} {'ValPPL':>10} {'耗时':>8}")
    print("-" * 40)
    for ep in range(1, args.epochs + 1):
        model.train()
        st, seen, tot = time.time(), 0, 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            pad_mask = x == pad_id
            logits = model(x, pad_mask)
            # ignore_index=-100：问题段 + padding 不产生梯度（Q4）
            loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1),
                                   ignore_index=-100)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            n_tok = int((y != -100).sum())
            tot += float(loss.detach()) * n_tok   # 按回答段 token 加权，与 val PPL 同口径
            seen += n_tok
        _, v_ppl = evaluate_ppl(model, val_loader, device, pad_id)
        mark = ""
        if v_ppl < best:
            best = v_ppl
            mark = " *"
            torch.save({"model_state": model.state_dict(),
                        "char2idx": char2idx, "idx2char": idx2char,
                        "special_ids": special_ids, "args": vars(args)}, args.save)
        print(f"{ep:>6} {tot/seen:>10.4f} {v_ppl:>10.2f} {time.time()-st:>7.1f}s{mark}")
        log["epochs"].append({"epoch": ep, "train_loss": tot / seen, "val_ppl": v_ppl,
                              "seconds": round(time.time() - st, 1), "is_best": bool(mark)})
        flush()

    log["best_val_ppl"] = best
    log["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    flush()
    print(f"\n最佳验证 PPL: {best:.2f}  已保存 {args.save}")
    print(f"日志已写入: {log_path}")


if __name__ == "__main__":
    main()
