"""
week6 分类任务 · 医疗问句 → 科室分诊（BERT 微调，单文件自包含版）

作业说明：
  老师的原始代码是分模块的（common/ labels、data、metrics + bert/ 训练、推理），
  交作业只需看主实验这一条链路，所以把 5 个文件的必要部分合并成本文件，
  不再依赖任何项目内模块——只依赖 torch / transformers / scikit-learn。

做了什么：
  1. 标签口径：15 个分诊科室（顺序 = label id，与 data/train.json、val.json 一致）
  2. 数据：读 train.json / val.json → 去掉验证集中与训练集完全重复的句子（防泄漏）
     → 固定 seed 抽样，保证和规则/TF-IDF/LLM 在同一批样本上评测
  3. 模型：bert-base-chinese + 池化（cls/mean/max）+ 线性分类头，只改这两处，12 层本体不动
  4. 训练：分层学习率、warmup + 线性衰减、AMP、类别加权（可关）、FGM 对抗训练（可选）
  5. 评测：准确率、Macro-F1、per-class recall、ECE（置信度校准误差）、逐条预测

实测结果（v1 数据：训练 53,360 / 验证 10,000 / 15 类 / 不均衡 8.7×，
         干净验证集 500 条，同一批样本，seed=42）：
    准确率 0.814   Macro-F1 0.7902   ECE 0.0812   264 秒（A10 24G，max_len 64）
  消融（全量干净集 9,926 条）：
    加权 on        0.8120 / 0.7817 / ECE 0.0675
    加权 off       0.8365 / 0.7975 / ECE 0.0443   ← 8.7× 未到 100× 阈值，关掉加权更好
    加权 on + FGM  0.8124 / 0.7840 / ECE 0.0563   ← Macro-F1 +0.23，ECE −17%

用法：
  python week6_med_bert.py                                  # 默认 cls 池化，3 epoch，全量训练
  python week6_med_bert.py --no_class_weight                # 关掉类别加权做对照
  python week6_med_bert.py --fgm                            # FGM 对抗训练（训练约 ×1.5）
  python week6_med_bert.py --pool max --max_length 32       # 换池化 + 短序列提速
  python week6_med_bert.py --epochs 1 --max_steps 5 --max_eval 50 --no_save   # CPU 冒烟
"""

import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------- 1. 标签口径
# 15 个分诊科室，顺序即 label id，必须与 data/*.json 里的 label 一致
LABELS = [
    "呼吸内科", "消化内科", "心血管内科", "神经内科", "内分泌科", "肝病科",
    "普通外科", "神经外科", "泌尿外科", "肛肠科",
    "妇科", "产科",
    "儿科", "男科", "肿瘤科",
]
NUM_LABELS = len(LABELS)

# ---------------------------------------------------------------- 2. 数据与评测


def find_bert_dir(model_dir=None):
    """从脚本所在目录向上找 pretrain_models/bert-base-chinese，找不到就报错（不联网下载）"""
    if model_dir:
        return Path(model_dir)
    for root in [HERE, *HERE.parents]:
        c = root / "pretrain_models" / "bert-base-chinese"
        if (c / "pytorch_model.bin").exists() or any(c.glob("*.safetensors")):
            return c
    raise FileNotFoundError(
        "未找到 pretrain_models/bert-base-chinese，请用 --model_dir 指定本地模型目录")


def load_splits(data_dir):
    """读 train.json / val.json，每条是 {"idx", "sentence", "label"}"""
    data_dir = Path(data_dir)
    with open(data_dir / "train.json", encoding="utf-8") as f:
        train = json.load(f)
    with open(data_dir / "val.json", encoding="utf-8") as f:
        val = json.load(f)
    return train, val


def clean_val(train, val):
    """
    去掉与训练集句子完全重复的验证样本。
    这份数据有 74/10000（0.74%）完全重复，不去掉的话靠词面匹配的模型会「背答案」。
    """
    seen = {d["sentence"] for d in train}
    return [d for d in val if d["sentence"] not in seen]


def sample_subset(items, n, seed=42):
    """固定 seed 抽样：规则 / TF-IDF / BERT / LLM 必须用同一批样本才可比"""
    if n <= 0 or n >= len(items):
        return list(items)
    return random.Random(seed).sample(items, n)


def classification_report(y_true, y_pred, num_classes=NUM_LABELS):
    """
    统一口径的指标：准确率 + Macro-F1 + 每类 recall + 混淆矩阵。
    多分类必须显式写 average="macro"——sklearn 默认 binary，会报错或给出误导结果。
    """
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

    report = {
        "n": len(y_true),
        "acc": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro")), 4),
        "weighted_f1": round(float(f1_score(y_true, y_pred, average="weighted")), 4),
    }
    labels = list(range(num_classes))
    per_class = {}
    for c in labels:
        mask = [t == c for t in y_true]
        if any(mask):
            hit = sum(1 for t, p, m in zip(y_true, y_pred, mask) if m and p == c)
            per_class[c] = round(hit / sum(mask), 4)
    report["per_class_recall"] = per_class
    report["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    return report


def expected_calibration_error(y_true, probs, n_bins=10):
    """
    ECE：把置信度分桶，比较「平均置信度」与「实际准确率」的差。越小越可信。
    >0.1 时置信度不能用来做「低置信转人工」的阈值。
    """
    import numpy as np

    probs = np.asarray(probs)
    conf, pred = probs.max(axis=1), probs.argmax(axis=1)
    correct = (pred == np.asarray(y_true)).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.sum():
            ece += abs(correct[mask].mean() - conf[mask].mean()) * mask.sum() / len(conf)
    return round(float(ece), 4)


# ---------------------------------------------------------------- 3. 模型

class MedDataset(Dataset):
    """{"sentence", "label"} → BERT 输入张量"""

    def __init__(self, data, tok, max_length=32):
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


class BertClassifier(nn.Module):
    """
    BERT + 池化 + 线性头。架构只改两处：池化方式、输出维度。
    池化前必须把 [PAD] 位置 mask 掉，否则 mean/max 会把补位统计进去（常见隐形 bug）。
    """

    def __init__(self, model_dir, num_labels=NUM_LABELS, pool="cls", dropout=0.1):
        super().__init__()
        from transformers import AutoModel
        self.bert = AutoModel.from_pretrained(str(model_dir))
        self.pool = pool
        hidden = self.bert.config.hidden_size          # bert-base = 768
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_labels)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        hs = out.last_hidden_state                     # [B, L, 768]
        if self.pool == "cls":
            vec = hs[:, 0]
        else:
            m = attention_mask.unsqueeze(-1).float()   # [B, L, 1]
            if self.pool == "mean":
                vec = (hs * m).sum(1) / m.sum(1).clamp(min=1e-9)
            else:                                      # max：PAD 位置压到极小值再取 max
                vec = hs.masked_fill(m == 0, -1e9).max(1).values
        return self.classifier(self.dropout(vec))


class FGM:
    """
    FGM 对抗训练：反传拿到梯度后，在 word_embeddings 上沿梯度方向加扰动 r = eps*g/||g||，
    再用扰动后的权重重算一次前向+反传，两次梯度一起更新，最后还原权重。
    只扰 embedding 是因为文本任务的「输入」就是 embedding；必须除 ||g|| 否则 eps 强度漂移。
    """

    def __init__(self, model, emb_name="word_embeddings", eps=1.0):
        self.model, self.emb_name, self.eps, self.backup = model, emb_name, eps, {}

    def attack(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name and param.grad is not None:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    param.data.add_(self.eps * param.grad / norm)

    def restore(self):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


def focal_loss(logits, targets, weight=None, gamma=2.0):
    """Focal Loss：L = -alpha_t * (1 - p_t)^gamma * log(p_t)，alpha 复用类别权重"""
    logp = F.log_softmax(logits, dim=-1)
    logp_t = logp.gather(1, targets.unsqueeze(1)).squeeze(1)
    p_t = logp_t.exp()
    alpha = weight[targets] if weight is not None else 1.0
    return (-alpha * ((1.0 - p_t) ** gamma) * logp_t).mean()


# ---------------------------------------------------------------- 4. 训练与评测

@torch.no_grad()
def evaluate(model, loader, device):
    from sklearn.metrics import accuracy_score, f1_score

    model.eval()
    ys, ps, logits_all = [], [], []
    for batch in loader:
        logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        ps.extend(logits.argmax(-1).cpu().tolist())
        ys.extend(batch["labels"].tolist())
        logits_all.append(logits.cpu())
    return {"acc": round(float(accuracy_score(ys, ps)), 4),
            "macro_f1": round(float(f1_score(ys, ps, average="macro")), 4),
            "logits": torch.cat(logits_all)}


@torch.no_grad()
def predict(model, tok, texts, device, max_length=32, batch_size=32):
    """返回 (预测 id, 每类概率, 每条毫秒数)"""
    preds, probs = [], []
    t0 = time.time()
    for i in range(0, len(texts), batch_size):
        enc = tok(texts[i:i + batch_size], truncation=True, max_length=max_length,
                  padding="max_length", return_tensors="pt")
        logits = model(enc["input_ids"].to(device), enc["attention_mask"].to(device))
        p = torch.softmax(logits.float(), dim=-1)
        preds.extend(p.argmax(-1).cpu().tolist())
        probs.extend(p.cpu().tolist())
    return preds, probs, round((time.time() - t0) * 1000 / max(1, len(texts)), 1)


def main():
    ap = argparse.ArgumentParser(description="医疗问句科室分诊 · BERT 微调（单文件版）")
    ap.add_argument("--data_dir", default=str(HERE / "data"), help="train.json / val.json 所在目录")
    ap.add_argument("--model_dir", default=None, help="本地 bert-base-chinese 目录，默认自动向上查找")
    ap.add_argument("--out_dir", default=str(HERE / "outputs"))
    ap.add_argument("--pool", default="cls", choices=["cls", "mean", "max"])
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-5, help="BERT 层学习率")
    ap.add_argument("--head_lr_mult", type=float, default=5.0, help="分类头 lr = BERT lr × 该倍数")
    ap.add_argument("--max_length", type=int, default=32,
                    help="句长 p99=22 字，32 零信息损失，比默认 64 快约 40%%")
    ap.add_argument("--no_class_weight", action="store_true", help="关掉类别加权（做对照）")
    ap.add_argument("--label_smoothing", type=float, default=0.0)
    ap.add_argument("--focal_gamma", type=float, default=0.0, help=">0 启用 Focal Loss")
    ap.add_argument("--fgm", action="store_true", help="打开 FGM 对抗训练")
    ap.add_argument("--fgm_eps", type=float, default=1.0)
    ap.add_argument("--n", type=int, default=500, help="评测抽样条数，0=全部干净样本")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_steps", type=int, default=0, help="每 epoch 最多步数，0=全跑（冒烟用）")
    ap.add_argument("--max_eval", type=int, default=0, help="验证集最多评多少条，0=全评（冒烟用）")
    ap.add_argument("--no_save", action="store_true", help="不落盘 checkpoint（冒烟用）")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bert_dir = find_bert_dir(args.model_dir)
    tok = AutoTokenizer.from_pretrained(str(bert_dir))

    train, val = load_splits(args.data_dir)
    clean = clean_val(train, val)
    if args.max_eval:
        val_eval = val[: args.max_eval]
    else:
        val_eval = val
    print(f"[data] train={len(train)} val={len(val)} 去掉重复={len(val) - len(clean)} "
          f"干净集={len(clean)} 训练时验证={len(val_eval)}", flush=True)

    train_dl = DataLoader(MedDataset(train, tok, args.max_length),
                          batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(MedDataset(val_eval, tok, args.max_length), batch_size=args.batch_size)

    model = BertClassifier(bert_dir, NUM_LABELS, pool=args.pool).to(device)

    # 分层学习率：分类头随机初始化要学得快，BERT 要小步更新，避免冲掉预训练知识
    head = list(model.classifier.parameters()) + list(model.dropout.parameters())
    body = [p for n, p in model.named_parameters()
            if not n.startswith("classifier") and not n.startswith("dropout")]
    opt = torch.optim.AdamW([{"params": body, "lr": args.lr},
                             {"params": head, "lr": args.lr * args.head_lr_mult}],
                            weight_decay=0.01)

    # 类别加权：balanced = 按 1/类频 反比，用于缓解长尾（本数据 8.7× 时实测是负收益）
    if args.no_class_weight:
        weight = None
    else:
        import numpy as np
        from sklearn.utils.class_weight import compute_class_weight
        cw = compute_class_weight("balanced", classes=np.arange(NUM_LABELS),
                                  y=[d["label"] for d in train])
        weight = torch.tensor(cw, dtype=torch.float).to(device)

    if args.focal_gamma > 0:
        crit = lambda logits, y: focal_loss(logits, y, weight, args.focal_gamma)   # noqa: E731
        crit_desc = f"Focal(gamma={args.focal_gamma})"
    else:
        crit = nn.CrossEntropyLoss(weight=weight, label_smoothing=args.label_smoothing)
        crit_desc = "CE" + (f"+LS({args.label_smoothing})" if args.label_smoothing else "")

    steps_per_epoch = min(len(train_dl), args.max_steps) if args.max_steps else len(train_dl)
    total_steps = steps_per_epoch * args.epochs
    warmup = int(total_steps * 0.1)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / max(1, warmup) if s < warmup
        else max(0.0, (total_steps - s) / max(1, total_steps - warmup)))
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    fgm = FGM(model, eps=args.fgm_eps) if args.fgm else None

    print(f"[cfg] device={device} pool={args.pool} loss={crit_desc} "
          f"class_weight={'off' if weight is None else 'on'} "
          f"fgm={'on(eps=%s)' % args.fgm_eps if fgm else 'off'} epochs={args.epochs} "
          f"steps/epoch={steps_per_epoch} max_length={args.max_length}", flush=True)

    best, log = 0.0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, running, n_step = time.time(), 0.0, 0
        for i, batch in enumerate(train_dl):
            ids, mask = batch["input_ids"].to(device), batch["attention_mask"].to(device)
            y = batch["labels"].to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
                loss = crit(model(ids, mask), y)
            scaler.scale(loss).backward()
            if fgm is not None:
                fgm.attack()
                with torch.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
                    loss_adv = crit(model(ids, mask), y)
                scaler.scale(loss_adv).backward()
                fgm.restore()
            scaler.step(opt)
            scaler.update()
            sched.step()
            running, n_step = running + loss.item(), n_step + 1
            if i % 200 == 0:
                print(f"  epoch {epoch} step {i}/{steps_per_epoch} loss={loss.item():.4f}", flush=True)
            if args.max_steps and n_step >= args.max_steps:
                break

        res = evaluate(model, val_dl, device)
        log.append({"epoch": epoch, "train_loss": round(running / max(1, n_step), 4),
                    "val_acc": res["acc"], "val_macro_f1": res["macro_f1"],
                    "elapsed_s": round(time.time() - t0, 1)})
        print(f"epoch {epoch}: val_acc={res['acc']} val_f1={res['macro_f1']}", flush=True)
        if res["macro_f1"] > best and not args.no_save:
            best = res["macro_f1"]
            torch.save({"state_dict": model.state_dict(), "pool": args.pool,
                        "val_acc": res["acc"], "val_macro_f1": res["macro_f1"],
                        "args": vars(args)}, out_dir / "bert_ckpt.pt")

    if not args.no_save:
        (out_dir / "bert_train_log.json").write_text(
            json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"最佳 val Macro-F1 = {best} → {out_dir / 'bert_ckpt.pt'}")

    # ---- 干净集评测：与规则 / TF-IDF / LLM 使用同一批样本（seed 固定）
    samples = sample_subset(clean, args.n, args.seed)
    texts = [d["sentence"] for d in samples]
    y_true = [d["label"] for d in samples]
    preds, probs, ms = predict(model, tok, texts, device,
                               max_length=args.max_length, batch_size=32)
    report = classification_report(y_true, preds)
    report.update({"method": "bert_finetune", "pool": args.pool,
                   "ms_per_sample": ms,
                   "ece": expected_calibration_error(y_true, probs)})
    (out_dir / "bert_pred.json").write_text(
        json.dumps({"meta": {"n_eval": len(samples), "seed": args.seed},
                    "report": report,
                    "preds": [{"text": t, "true": y, "pred": p}
                              for t, y, p in zip(texts, y_true, preds)]},
                   ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n评测 {len(samples)} 条（干净集，seed={args.seed}）| {ms} ms/条 | ECE={report['ece']}")
    print("| 方法 | 准确率 | Macro-F1 | 解析失败 |\n|---|---|---|---|")
    print(f"| BERT 微调 | {report['acc']} | {report['macro_f1']} | 0 |")
    print(f"→ 结果写入 {out_dir / 'bert_pred.json'}")


if __name__ == "__main__":
    main()
