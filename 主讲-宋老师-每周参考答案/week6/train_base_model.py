# -*- coding: utf-8 -*-
"""
换基座实验：BERT vs 中文 RoBERTa（同一个分类任务，同一套超参，只换预训练底座）

作业要回答的问题：
  「换一个更强的预训练模型，效果能提升多少？值不值得为它多花这些成本？」

为什么要专门做这个实验：
  1. 前面课上的对比都是「换方法」（规则 → TF-IDF → BERT → LLM），
     这个实验是**控制变量**：方法不变、数据不变、超参不变，只换 BERT 本体。
  2. 换基座是工程上最省事的提分手段之一（代码几乎不动、标注不用重做），
     但它的收益上限由**预训练语料与训练目标**决定，不是无限的。
  3. 「差 0.02 的 Macro-F1 到底算不算提升」是必须量化的问题——
     compare_base_models.py 里用配对 bootstrap 给出置信区间，别只看小数点后第三位。

两个基座的区别（这才是教学重点）：
  bert-base-chinese          12 层 / 768 隐层 / 21128 词表 / 102M 参数，中文维基百科，随机整词掩码
  chinese-roberta-wwm-ext    12 层 / 768 隐层 / 21128 词表 / 102M 参数 —— **结构完全一致**，但：
                             · 预训练语料更大（wiki + 新闻 + 百科等扩展语料）
                             · RoBERTa 的训练配方：动态掩码 + 去掉 NSP + 更大 batch
                             · 全词掩码（whole word masking，一个词整个被遮掉）
  → 注意：hfl 这个 checkpoint 的 config.json 里 model_type 仍然写的是 "bert"
    （它是「用 RoBERTa 的训练配方去训 BERT 结构」的产物），所以加载路径、代码、参数量全都一样，
    差异**只**来自「预训练见过什么、怎么学」——这正好是「基座本身值多少钱」的干净实验。

用法（把本文件放到 code/bert/ 下运行）：
  python bert/train_base_model.py                                    # 默认跑 BERT 基线
  python bert/train_base_model.py --model chinese-roberta-wwm-ext     # 换成中文 RoBERTa
  python bert/train_base_model.py --model chinese-roberta-wwm-ext --tag roberta
  python bert/train_base_model.py --max_steps 20 --max_eval 200       # 冒烟：几十秒看通不通

产物（默认不保存训练后的模型权重）：
  results/<tag>_log.json   逐 epoch 的 loss / acc / macro-F1 / 耗时
  results/<tag>_pred.json  验证集逐条预测（给 compare_base_models.py 做配对检验用）
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset

# ── 找到 code/ 根目录（本文件放在 code/bert/ 下时，common/ 就在上一级）──────────
# 这样做是为了让脚本不依赖「从哪个目录启动」：从项目根、从 bert/、从别处启动都能 import common。
def find_code_root() -> Path:
    for p in Path(__file__).resolve().parents:
        if (p / "common" / "data.py").is_file():
            return p
    raise SystemExit("找不到 common/ 目录：请把本文件放到 code/bert/ 下运行")


CODE_ROOT = find_code_root()
sys.path.insert(0, str(CODE_ROOT))

from common.data import clean_val, load_splits           # noqa: E402
from common.labels import LABELS                        # noqa: E402
from common.model_paths import PRETRAIN, resolve         # noqa: E402

# 产物目录：跟课件其它实验一致，放 code/outputs/ 下（可用 --out_dir 覆盖）
RESULTS_DIR = CODE_ROOT / "outputs" / "base_models"


def set_seed(seed: int):
    """
    固定所有随机源。为什么要单独写一个函数：
      DataLoader(shuffle=True) 用的是 torch 全局 RNG，模型初始化也用同一个，
      不显式设种子的话，两次跑出来的差距可能比「换基座」还大。
    注意：同 seed ≠ 逐位可复现（cuDNN 卷积算法、AMP 有非确定性），
    所以结论要看「多次的均值/置信区间」，而不是某一次的第三个小数位。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class TnewsDataset(Dataset):
    """把 {"sentence","label"} 变成 BERT/RoBERTa 的输入张量（两个基座的 tokenizer 接口一致）"""

    def __init__(self, data, tok, max_length=64):
        self.data, self.tok, self.max_length = data, tok, max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        d = self.data[i]
        enc = self.tok(d["sentence"], truncation=True, max_length=self.max_length,
                       padding="max_length", return_tensors="pt")
        return {"input_ids": enc["input_ids"][0],
                "attention_mask": enc["attention_mask"][0],
                "labels": torch.tensor(d["label"])}


class BaseClassifier(nn.Module):
    """
    任意 BERT 系基座 + 池化 + 线性分类头。

    为什么用 AutoModel 而不是 BertModel：
      AutoModel 按 config.json 里的 model_type 自动选 RobertaModel / BertModel，
      换基座时**一行代码都不用改**——这就是「Auto 类」存在的意义。
    为什么两个基座能共用同一个池化代码：
      RoBERTa 的 <s> 和 BERT 的 [CLS] 都固定在序列第 0 位，CLS 池化天然可复用。
    """

    def __init__(self, model_dir, num_labels=15, pool="cls", dropout=0.1):
        super().__init__()
        from transformers import AutoModel
        # transformers 5.x 里 from_pretrained 不接受含 ".." 的相对路径（会被当成 repo id），这里统一传绝对路径
        self.bert = AutoModel.from_pretrained(str(Path(model_dir).resolve()))
        self.pool = pool
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask):
        # return_dict=True：transformers 5.x 里不传会拿到 tuple，容易下标取错
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        hs = out.last_hidden_state                      # [B, L, hidden]
        if self.pool == "cls":
            vec = hs[:, 0]
        else:
            m = attention_mask.unsqueeze(-1).float()
            if self.pool == "mean":
                vec = (hs * m).sum(1) / m.sum(1).clamp(min=1e-9)
            else:
                vec = hs.masked_fill(m == 0, -1e9).max(1).values
        return self.classifier(self.dropout(vec))


@torch.no_grad()
def evaluate(model, loader, device):
    """返回 (acc, macro_f1, 逐条预测, 逐条标签)"""
    model.eval()
    ys, ps = [], []
    for b in loader:
        logits = model(b["input_ids"].to(device), b["attention_mask"].to(device))
        ps.extend(logits.argmax(-1).cpu().tolist())
        ys.extend(b["labels"].tolist())
    return accuracy_score(ys, ps), f1_score(ys, ps, average="macro"), ps, ys


def per_class_f1(ys, ps):
    """15 类各自的 F1：换基座不一定全面提升，看「提升集中在哪几类」比看总分更有信息量"""
    f1s = f1_score(ys, ps, average=None, labels=list(range(len(LABELS))), zero_division=0)
    return {LABELS[i]: round(float(v), 4) for i, v in enumerate(f1s)}


def main():
    ap = argparse.ArgumentParser()
    # ↓ 唯一需要改的「实验变量」
    ap.add_argument("--model", default="bert-base-chinese",
                    help="基座名（pretrain_models 下的目录名）或绝对路径")
    ap.add_argument("--tag", default="", help="产物文件名前缀，默认由 --model 推导")
    # ↓ 与课件 BERT 基线完全一致的训练配置（不要动，动了就不是「同等实验」）
    ap.add_argument("--pool", default="cls", choices=["cls", "mean", "max"])
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-5, help="基座学习率")
    ap.add_argument("--head_lr_mult", type=float, default=5.0, help="分类头 lr = 基座 lr × 该倍数")
    ap.add_argument("--max_length", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    # ↓ 冒烟/调试
    ap.add_argument("--max_steps", type=int, default=0, help="每个 epoch 最多跑多少步，0=全跑")
    ap.add_argument("--max_eval", type=int, default=0, help="验证集最多评多少条，0=全评")
    ap.add_argument("--save", action="store_true", help="保存训练后的权重（默认不保存）")
    ap.add_argument("--out_dir", default=str(RESULTS_DIR), help="产物目录，默认 code/outputs/base_models")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # --model 既可以是目录名（自动在 pretrain_models 下查），也可以是路径
    model_dir = Path(args.model)
    if not model_dir.is_dir():
        try:
            model_dir = resolve(args.model)
        except FileNotFoundError as e:
            raise SystemExit(f"{e}\n\n先把基座下载到 pretrain_models/ 下（见 README「怎么下载基座」），"
                             f"或者直接把 --model 传成模型目录的绝对路径")
    tag = args.tag or model_dir.name

    set_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    train, val = load_splits()
    # 关键口径：评测集必须是「去掉与训练集重复样本」的干净验证集，
    # 否则靠记忆词面的模型会虚高（实测全量 val 比 clean val 高 1.7 个点）
    clean, n_dup = clean_val(train, val)
    if args.max_eval:
        clean = clean[: args.max_eval]

    train_dl = DataLoader(TnewsDataset(train, tok, args.max_length),
                          batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(TnewsDataset(clean, tok, args.max_length), batch_size=args.batch_size)

    model = BaseClassifier(model_dir, num_labels=len(LABELS), pool=args.pool).to(device)
    n_param = sum(p.numel() for p in model.parameters()) / 1e6

    # 分层学习率：分类头随机初始化要学得快，基座要大步？不——基座必须小步，否则冲掉预训练知识
    head_params = list(model.classifier.parameters()) + list(model.dropout.parameters())
    base_params = [p for n, p in model.named_parameters()
                   if not n.startswith("classifier") and not n.startswith("dropout")]
    opt = torch.optim.AdamW([
        {"params": base_params, "lr": args.lr},
        {"params": head_params, "lr": args.lr * args.head_lr_mult},
    ], weight_decay=0.01)

    # 类别加权：15 类最大/最小差 23 倍，不加权尾部类 recall 会趋近 0
    cw = compute_class_weight("balanced", classes=np.arange(len(LABELS)),
                              y=[d["label"] for d in train])
    weight = torch.tensor(cw, dtype=torch.float).to(device)
    crit = nn.CrossEntropyLoss(weight=weight)

    steps_per_epoch = min(len(train_dl), args.max_steps) if args.max_steps else len(train_dl)
    total_steps = steps_per_epoch * args.epochs
    warmup = int(total_steps * 0.1)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / max(1, warmup) if s < warmup
        else max(0.0, (total_steps - s) / max(1, total_steps - warmup)))
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    print(f"[cfg] base={model_dir.name} params={n_param:.1f}M pool={args.pool} "
          f"epochs={args.epochs} bs={args.batch_size} lr={args.lr} seed={args.seed} "
          f"train={len(train)} clean_val={len(clean)}(去重 {n_dup}) steps/epoch={steps_per_epoch} "
          f"save={'on' if args.save else 'off'}", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best = {"macro_f1": -1.0}
    log = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, running, n_step = time.time(), 0.0, 0
        for i, batch in enumerate(train_dl):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            y = batch["labels"].to(device)

            opt.zero_grad(set_to_none=True)
            # AMP 混合精度：显存约降 40%、速度更快；分类任务用 fp16 完全够（不需要 bf16）
            with torch.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
                loss = crit(model(ids, mask), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()

            running += loss.item()
            n_step += 1
            if i % 200 == 0:
                print(f"  epoch {epoch} step {i}/{steps_per_epoch} loss={loss.item():.4f}", flush=True)
            if args.max_steps and n_step >= args.max_steps:
                break

        acc, mf1, ps, ys = evaluate(model, val_dl, device)
        elapsed = round(time.time() - t0, 1)
        log.append({"epoch": epoch, "train_loss": round(running / max(1, n_step), 4),
                    "val_acc": round(float(acc), 4), "val_macro_f1": round(float(mf1), 4),
                    "elapsed_s": elapsed})
        print(f"epoch {epoch}: val_acc={acc:.4f} val_macro_f1={mf1:.4f} ({elapsed}s)", flush=True)

        # 按 Macro-F1 选最优 epoch（不取最后一轮：3 epoch 时第 2/3 轮经常交叉）
        if mf1 > best["macro_f1"]:
            best = {"macro_f1": round(float(mf1), 4), "acc": round(float(acc), 4),
                    "epoch": epoch, "pred": ps, "y": ys, "elapsed_s": elapsed}

    (out_dir / f"{tag}_log.json").write_text(
        json.dumps({"meta": {"base": model_dir.name, "n_train": len(train), "n_clean_val": len(clean),
                             "params_M": round(n_param, 1), "epochs": args.epochs,
                             "batch_size": args.batch_size, "lr": args.lr,
                             "head_lr_mult": args.head_lr_mult, "max_length": args.max_length,
                             "pool": args.pool, "seed": args.seed,
                             "device": device, "model_dir": str(model_dir)},
                    "best_epoch": best["epoch"], "best_acc": best["acc"],
                    "best_macro_f1": best["macro_f1"], "per_class_f1": per_class_f1(best["y"], best["pred"]),
                    "epochs_log": log},
                   ensure_ascii=False, indent=2), encoding="utf-8")

    # 逐条预测单独存：compare_base_models.py 要用它做「配对」比较（同一批样本、同一顺序）
    (out_dir / f"{tag}_pred.json").write_text(
        json.dumps({"base": model_dir.name, "epoch": best["epoch"],
                    "y": best["y"], "pred": best["pred"]}, ensure_ascii=False), encoding="utf-8")

    if args.save:
        torch.save({"state_dict": model.state_dict(), "base": model_dir.name, "pool": args.pool},
                   out_dir / f"{tag}_ckpt.pt")
        print(f"权重已保存 → {out_dir / f'{tag}_ckpt.pt'}", flush=True)

    print(f"\n[{model_dir.name}] 最优 epoch={best['epoch']} "
          f"acc={best['acc']:.4f} macro_f1={best['macro_f1']:.4f}", flush=True)
    print(f"产物 → {out_dir / (tag + '_log.json')}", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # Windows 上 torch/numpy 各自带一份 OpenMP 会报 #15
    main()
