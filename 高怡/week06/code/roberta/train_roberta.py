"""
基金公告六分类 —— RoBERTa (6 层) 微调实验

核心训练流程：
  - 分层学习率：BERT 层 2e-5，分类头 = BERT × head_lr_mult（默认 5）
  - Warmup 10% + 线性衰减
  - 类别加权 CE（可关）
  - AMP 混合精度（CUDA: fp16 + GradScaler；CPU: bfloat16）
  - 每个 epoch 在 val 上算 Macro-F1，按 Macro-F1 存最优 ckpt
  - 可选：标签平滑 / Focal Loss / FGM 对抗训练

数据：data/dataset/split/cls_{train,val,test}.jsonl
输出：output/roberta/
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from common.label import CLS_ID2LABEL, CLS_NUM_LABELS  # noqa: E402

SPLIT_DIR = BASE_DIR / "data" / "dataset" / "split"
PRETRAIN_DIR = BASE_DIR / "pretrain_models" / "chinese-roberta-wwm-ext"
OUT_DIR = BASE_DIR / "output" / "roberta"


# ---------------------------------------------------------------- 数据集


class AnnDataset(Dataset):
    def __init__(self, records: list[dict], tok, max_length: int = 128):
        self.records = records
        self.tok = tok
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i):
        text = self.records[i]["text"]
        label = int(self.records[i]["cls_label"])
        enc = self.tok(text, truncation=True, max_length=self.max_length,
                       padding="max_length", return_tensors="pt")
        return {"input_ids": enc["input_ids"][0],
                "attention_mask": enc["attention_mask"][0],
                "labels": torch.tensor(label, dtype=torch.long)}


def load_split(name: str) -> list[dict]:
    path = SPLIT_DIR / f"cls_{name}.jsonl"
    return [json.loads(l) for l in path.open(encoding="utf-8")]


# ---------------------------------------------------------------- 模型


class BertClassifier(nn.Module):
    """BERT + 池化 + 线性分类头。"""

    def __init__(self, model_dir: Path, num_labels: int, pool: str = "cls",
                 dropout: float = 0.1, n_layers: int | None = None):
        super().__init__()
        from transformers import AutoConfig, AutoModel
        if n_layers is None:
            self.bert = AutoModel.from_pretrained(str(model_dir))
        else:
            # 指定层数时按 config 覆盖构建（加载旧 ckpt 时用，不改动磁盘上的配置）
            cfg = AutoConfig.from_pretrained(str(model_dir))
            cfg.num_hidden_layers = n_layers
            self.bert = AutoModel.from_config(cfg)
        self.pool = pool
        hidden = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_labels)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        hs = out.last_hidden_state
        if self.pool == "cls":
            vec = hs[:, 0]
        else:
            m = attention_mask.unsqueeze(-1).float()
            if self.pool == "mean":
                vec = (hs * m).sum(1) / m.sum(1).clamp(min=1e-9)
            else:  # max
                vec = hs.masked_fill(m == 0, -1e9).max(1).values
        return self.classifier(self.dropout(vec))


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


def focal_loss(logits, targets, weight=None, gamma: float = 2.0):
    logp = F.log_softmax(logits, dim=-1)
    logp_t = logp.gather(1, targets.unsqueeze(1)).squeeze(1)
    p_t = logp_t.exp()
    alpha = weight[targets] if weight is not None else 1.0
    return (-alpha * ((1.0 - p_t) ** gamma) * logp_t).mean()


# ---------------------------------------------------------------- 评估


@torch.no_grad()
def evaluate(model, loader, device):
    from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
    model.eval()
    ys, ps = [], []
    for batch in loader:
        logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        ps.extend(logits.argmax(-1).cpu().tolist())
        ys.extend(batch["labels"].tolist())
    return {
        "acc": float(accuracy_score(ys, ps)),
        "macro_f1": float(f1_score(ys, ps, average="macro")),
        "weighted_f1": float(f1_score(ys, ps, average="weighted")),
        "report": classification_report(
            ys, ps, labels=list(range(CLS_NUM_LABELS)),
            target_names=[CLS_ID2LABEL[i] for i in range(CLS_NUM_LABELS)],
            output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(ys, ps, labels=list(range(CLS_NUM_LABELS))).tolist(),
        "y_true": ys, "y_pred": ps,
    }


def plot_cm(cm: list, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    cm = np.asarray(cm)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)
    labels = [CLS_ID2LABEL[i] for i in range(CLS_NUM_LABELS)]
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(xticks=range(CLS_NUM_LABELS), yticks=range(CLS_NUM_LABELS),
           xticklabels=labels, yticklabels=labels, xlabel="预测", ylabel="真实")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")
    thresh = cm_norm.max() / 2.0 if cm_norm.max() > 0 else 0.5
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center",
                    color="white" if cm_norm[i, j] > thresh else "black", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- 主流程


def set_seed(seed: int) -> None:
    """固定 Python / NumPy / PyTorch 三个 RNG，保证可复现。

    DataLoader(shuffle=True, num_workers=0)、模型初始化、dropout 都走全局
    RNG，在这里统一固定后即可复现整条训练流程。
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="cls", choices=["cls", "mean", "max"])
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5, help="BERT 层学习率")
    ap.add_argument("--head_lr_mult", type=float, default=5.0)
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--no_class_weight", action="store_true")
    ap.add_argument("--label_smoothing", type=float, default=0.0)
    ap.add_argument("--focal_gamma", type=float, default=0.0)
    ap.add_argument("--fgm", action="store_true")
    ap.add_argument("--fgm_eps", type=float, default=1.0)
    ap.add_argument("--max_steps", type=int, default=0, help="每个 epoch 最多跑多少步，0=全跑")
    ap.add_argument("--max_eval", type=int, default=0, help="验证集最多评多少条，0=全评")
    ap.add_argument("--no_save", action="store_true")
    ap.add_argument("--eval_test", action="store_true", help="训练结束后在测试集上做最终评估")
    ap.add_argument("--eval_train", action="store_true",
                    help="每个 epoch 额外在训练集上评估，用于观察 train/val 差距（过拟合诊断）")
    ap.add_argument("--tag", default="run", help="实验标识，用于区分多次实验的输出文件名")
    ap.add_argument("--seed", type=int, default=42, help="随机种子，固定后可复现")
    args = ap.parse_args()
    set_seed(args.seed)

    from transformers import AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # CPU 上用 bfloat16 做 autocast；GradScaler 只在 CUDA 上启用
    amp_dtype = torch.float16 if device == "cuda" else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    tok = AutoTokenizer.from_pretrained(str(PRETRAIN_DIR))
    train = load_split("train")
    val = load_split("val")
    test = load_split("test") if args.eval_test else []
    if args.max_eval:
        val = val[: args.max_eval]

    train_dl = DataLoader(AnnDataset(train, tok, args.max_length),
                          batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_dl = DataLoader(AnnDataset(val, tok, args.max_length),
                        batch_size=args.batch_size, num_workers=0)
    test_dl = DataLoader(AnnDataset(test, tok, args.max_length),
                         batch_size=args.batch_size, num_workers=0) if test else None
    # 训练集评估用独立 loader：不打乱，保证多次评估的数据顺序一致
    train_eval_dl = DataLoader(AnnDataset(train, tok, args.max_length),
                               batch_size=args.batch_size, num_workers=0) if args.eval_train else None

    model = BertClassifier(PRETRAIN_DIR, num_labels=CLS_NUM_LABELS,
                           pool=args.pool, dropout=args.dropout).to(device)

    # 分层学习率
    head_params = list(model.classifier.parameters()) + list(model.dropout.parameters())
    bert_params = [p for n, p in model.named_parameters()
                   if not n.startswith("classifier") and not n.startswith("dropout")]
    opt = torch.optim.AdamW([
        {"params": bert_params, "lr": args.lr},
        {"params": head_params, "lr": args.lr * args.head_lr_mult},
    ], weight_decay=0.01)

    # 类别加权
    if args.no_class_weight:
        weight = None
    else:
        from sklearn.utils.class_weight import compute_class_weight
        cw = compute_class_weight("balanced", classes=np.arange(CLS_NUM_LABELS),
                                  y=[d["cls_label"] for d in train])
        weight = torch.tensor(cw, dtype=torch.float).to(device)

    # 损失函数
    if args.focal_gamma > 0:
        crit = lambda logits, y: focal_loss(logits, y, weight, args.focal_gamma)  # noqa: E731
        crit_desc = f"Focal(gamma={args.focal_gamma})"
    else:
        crit = nn.CrossEntropyLoss(weight=weight, label_smoothing=args.label_smoothing)
        crit_desc = "CE" + (f"+LS({args.label_smoothing})" if args.label_smoothing > 0 else "")

    steps_per_epoch = min(len(train_dl), args.max_steps) if args.max_steps else len(train_dl)
    total_steps = steps_per_epoch * args.epochs
    warmup = int(total_steps * 0.1)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda step: step / max(1, warmup) if step < warmup
        else max(0.0, (total_steps - step) / max(1, total_steps - warmup)))

    fgm = FGM(model, eps=args.fgm_eps) if args.fgm else None

    print(f"[cfg] device={device} pool={args.pool} loss={crit_desc} "
          f"class_weight={'off' if weight is None else 'on'} "
          f"fgm={'on(eps=%s)' % args.fgm_eps if fgm else 'off'} "
          f"epochs={args.epochs} bs={args.batch_size} max_len={args.max_length} "
          f"steps/epoch={steps_per_epoch} val_n={len(val)} save={'off' if args.no_save else 'on'}",
          flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = args.tag
    best_f1, log = 0.0, []
    ckpt_path = OUT_DIR / f"roberta_best_{tag}.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, running, n_step = time.time(), 0.0, 0
        for i, batch in enumerate(train_dl):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            y = batch["labels"].to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device, dtype=amp_dtype, enabled=True):
                loss = crit(model(ids, mask), y)
            scaler.scale(loss).backward()
            if fgm is not None:
                fgm.attack()
                with torch.amp.autocast(device_type=device, dtype=amp_dtype, enabled=True):
                    loss_adv = crit(model(ids, mask), y)
                scaler.scale(loss_adv).backward()
                fgm.restore()
            scaler.step(opt)
            scaler.update()
            sched.step()
            running += loss.item()
            n_step += 1
            if i % 200 == 0:
                print(f"  epoch {epoch} step {i}/{steps_per_epoch} loss={loss.item():.4f}",
                      flush=True)
            if args.max_steps and n_step >= args.max_steps:
                break

        res = evaluate(model, val_dl, device)
        entry = {"epoch": epoch,
                 "train_loss": round(running / max(1, n_step), 4),
                 "val_acc": round(res["acc"], 4),
                 "val_macro_f1": round(res["macro_f1"], 4),
                 "val_weighted_f1": round(res["weighted_f1"], 4),
                 "elapsed_s": round(time.time() - t0, 1)}
        if train_eval_dl is not None:
            tres = evaluate(model, train_eval_dl, device)
            entry.update({
                "train_acc": round(tres["acc"], 4),
                "train_macro_f1": round(tres["macro_f1"], 4),
                "gap_macro_f1": round(tres["macro_f1"] - res["macro_f1"], 4),
                "gap_acc": round(tres["acc"] - res["acc"], 4),
            })
        log.append(entry)
        msg = (f"epoch {epoch}: val_acc={res['acc']:.4f} "
               f"val_macro_f1={res['macro_f1']:.4f} "
               f"val_weighted_f1={res['weighted_f1']:.4f}")
        if "train_macro_f1" in entry:
            msg += (f" | train_acc={entry['train_acc']:.4f} "
                    f"train_macro_f1={entry['train_macro_f1']:.4f} "
                    f"gap={entry['gap_macro_f1']:+.4f}")
        print(msg, flush=True)

        if res["macro_f1"] > best_f1 and not args.no_save:
            best_f1 = res["macro_f1"]
            torch.save({"state_dict": model.state_dict(),
                        "pool": args.pool,
                        "num_labels": CLS_NUM_LABELS,
                        "val_macro_f1": res["macro_f1"],
                        "val_acc": res["acc"],
                        "args": vars(args)}, ckpt_path)
            print(f"  >> 新最优 val_macro_f1={best_f1:.4f}，已保存 {ckpt_path.name}", flush=True)

    if not args.no_save:
        (OUT_DIR / f"roberta_train_log_{tag}.json").write_text(
            json.dumps({"config": vars(args), "epochs": log}, ensure_ascii=False, indent=2),
            encoding="utf-8")
        print(f"最佳 val_macro_f1={best_f1:.4f}")

    # 最终测试集评估
    if args.eval_test and test_dl is not None and ckpt_path.exists() and not args.no_save:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        tres = evaluate(model, test_dl, device)
        (OUT_DIR / f"test_metrics_{tag}.json").write_text(
            json.dumps({
                "test_acc": tres["acc"],
                "test_macro_f1": tres["macro_f1"],
                "test_weighted_f1": tres["weighted_f1"],
                "per_class": tres["report"],
                "confusion_matrix": tres["confusion_matrix"],
            }, ensure_ascii=False, indent=2), encoding="utf-8")
        plot_cm(tres["confusion_matrix"], OUT_DIR / f"test_confusion_matrix_{tag}.png")
        print(f"[test] acc={tres['acc']:.4f} macro_f1={tres['macro_f1']:.4f} "
              f"weighted_f1={tres['weighted_f1']:.4f}")


if __name__ == "__main__":
    main()
