"""在 BERT 基线上比较类别加权、Focal Loss、标签平滑和 FGM。

运行 python l2_bert_tricks.py [weighted|focal|smooth|fgm]，默认 weighted。
前三种方法替换损失函数；FGM 在普通交叉熵训练中增加一次对抗步。
"""
import csv
import time
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

import l2_bert
from l2_bert import (TnewsDataset, BertClassifier, read_labels, LABEL2IDX)
from common import DATA_DIR, report

TRICK = "weighted"
USE_FGM = (TRICK == "fgm")

BATCH_SIZE = 32
EPOCHS = 3
LR = 2e-5
MAX_STEPS = 200  # 所有 epoch 合计最多更新 200 次
TRAIN_SECONDS = 120  # 单个方法训练预算；在当前 batch 完成后检查时间


def class_weights(train_tsv_path):
    """统计训练集各类样本数，返回 shape=[15] 的权重 tensor：样本越少权重越大"""
    counts = [0] * 15
    rows = list(csv.reader(open(train_tsv_path, encoding="utf-8"), delimiter="\t"))[1:]
    for label, text in rows:
        counts[LABEL2IDX[label]] += 1

    weights = []
    for c in counts:
        weights.append(1.0 / c if c else 0.0)
    mean = sum(weights) / len(weights)  # 权重归一化到均值 1
    for i in range(len(weights)):
        weights[i] = weights[i] / mean
    return torch.tensor(weights, dtype=torch.float32)


class WeightedCE(torch.nn.Module):
    """按训练集类别频次加权的交叉熵。"""
    def __init__(self, weight):
        super().__init__()
        # buffer 随模块移动设备，但不是需要训练的参数。
        self.register_buffer("weight", weight)

    def forward(self, logits, labels):
        return F.cross_entropy(logits, labels, weight=self.weight)


class FocalLoss(torch.nn.Module):
    """FL(p_t) = -(1 - p_t)^gamma * log(p_t)，容易样本损失被压低，难样本主导梯度"""
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, labels):
        # dim=-1 沿类别维计算概率的对数；后面取出每个样本真实类别对应的值。
        log_p = F.log_softmax(logits, dim=-1)
        log_pt = []
        for i in range(len(labels)):
            log_pt.append(log_p[i][labels[i]])
        log_pt = torch.stack(log_pt)
        p = log_pt.exp()
        loss = -((1 - p) ** self.gamma) * log_pt
        return loss.mean()


class LabelSmoothing(torch.nn.Module):
    """将 one-hot 标签与均匀分布混合，正确类概率为 1-eps+eps/类别数。"""
    def __init__(self, eps=0.1):
        super().__init__()
        self.eps = eps

    def forward(self, logits, labels):
        n = logits.shape[0]
        n_cls = logits.shape[1]
        target = torch.full((n, n_cls), self.eps / n_cls, device=logits.device)
        for i in range(n):
            target[i][labels[i]] = 1 - self.eps + self.eps / n_cls
        target = target.to(logits.device)
        log_p = F.log_softmax(logits, dim=-1)
        loss = -(target * log_p).sum(dim=-1)
        return loss.mean()


class FGM:
    """在 embedding 上加梯度方向的小扰动，让模型对扰动后的输入也预测正确（一种正则化）"""
    def __init__(self, model, epsilon=1.0):
        self.model = model
        self.epsilon = epsilon
        self.backup = {}

    def attack(self):
        for name, param in self.model.named_parameters():
            if "word_embeddings" not in name:
                continue
            if param.grad is None:
                continue
            # 单独备份权重，扰动结束后恢复。
            # clone 独立复制权重，避免备份随原权重一起改变。
            self.backup[name] = param.data.clone()
            norm = torch.norm(param.grad)
            if norm != 0:
                r_at = self.epsilon * param.grad / norm
                param.data.add_(r_at)

    def restore(self):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}


def make_criterion():
    if TRICK == "weighted":
        return WeightedCE(class_weights(DATA_DIR / "train.tsv"))
    if TRICK == "focal":
        return FocalLoss()
    if TRICK == "smooth":
        return LabelSmoothing()
    return None


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("device:", device, "  trick:", TRICK)
    tokenizer = AutoTokenizer.from_pretrained(l2_bert.MODEL_NAME)
    l2_bert.tokenizer = tokenizer

    train_ds = TnewsDataset(DATA_DIR / "train.tsv", tokenizer)
    test_ds = TnewsDataset(DATA_DIR / "test.tsv", tokenizer)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=l2_bert.collate_fn)
    test_loader = DataLoader(test_ds, batch_size=256, collate_fn=l2_bert.collate_fn)

    model = BertClassifier().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = make_criterion()
    if criterion is not None:
        criterion = criterion.to(device)
    fgm = FGM(model) if USE_FGM else None

    started = time.monotonic()
    steps = 0
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        n = 0
        for batch in tqdm(train_loader, desc=f"epoch {epoch + 1}", total=min(len(train_loader), MAX_STEPS - steps)):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids, attention_mask)
            if criterion is not None:
                loss = criterion(logits, labels)
            else:
                loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()

            # 对抗步累加梯度，恢复原权重后再更新参数。
            if fgm is not None:
                fgm.attack()
                loss_adv = F.cross_entropy(model(input_ids, attention_mask), labels)
                loss_adv.backward()
                fgm.restore()

            optimizer.step()

            total_loss += loss.item()
            n += 1
            steps += 1
            if steps >= MAX_STEPS or time.monotonic() - started >= TRAIN_SECONDS:
                break
        print(f"epoch {epoch + 1}  平均loss={total_loss / n:.4f}")

        if steps >= MAX_STEPS or time.monotonic() - started >= TRAIN_SECONDS:
            break
    print(f"实际训练 {steps} 步，耗时 {time.monotonic() - started:.1f} 秒")

    y_test = read_labels(DATA_DIR / "test.tsv")
    y_pred = l2_bert.predict(model, test_loader, device)
    print(f"\ntrick={TRICK}  train_samples={len(train_ds)}")
    result = report(y_test, y_pred)
    result["train_steps"] = steps
    return result


if __name__ == "__main__":
    from common import prepare_quick_data
    prepare_quick_data()
    if len(sys.argv) > 1:
        TRICK = sys.argv[1]
        if TRICK not in ("weighted", "focal", "smooth", "fgm"):
            raise SystemExit("技巧只能是 weighted / focal / smooth / fgm")
        USE_FGM = TRICK == "fgm"
    main()
