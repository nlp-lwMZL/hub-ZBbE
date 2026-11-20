"""L2 · BERT 微调：预训练模型 + 池化 + 分类头。运行: python l2_bert.py

流程：Dataset -> 模型(backbone+池化+分类头) -> 训练循环 -> 评测
训练技巧（加权CE/Focal/FGM等）见 l2_bert_tricks.py，是在本文件基础上的改动。
"""
import csv
import time
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
from common import DATA_DIR, report, macro_f1

MODEL_NAME = "hfl/chinese-roberta-wwm-ext"
POOLING = "cls"  # 对比实验："cls" / "mean" / "max"
MAX_LEN = 64
BATCH_SIZE = 32
EPOCHS = 3
LR = 2e-5
MAX_STEPS = 200  # 所有 epoch 合计最多更新 200 次
TRAIN_SECONDS = 120  # 单个方法训练预算；在当前 batch 完成后检查时间

LABELS = ["100","101","102","103","104","106","107","108","109","110","112","113","114","115","116"]
# 原始标签先映射到 0~14，预测后再转回原编号。
LABEL2IDX = {}
for i, l in enumerate(LABELS):
    LABEL2IDX[l] = i
IDX2LABEL = {}
for l, i in LABEL2IDX.items():
    IDX2LABEL[i] = l


# Dataset 约定实现 __len__ 和 __getitem__；DataLoader 会调用它们取一批样本。
class TnewsDataset(Dataset):
    def __init__(self, tsv_path, tokenizer):
        self.tokenizer = tokenizer
        self.samples = []
        rows = list(csv.reader(open(tsv_path, encoding="utf-8"), delimiter="\t"))[1:]
        for label, text in rows:
            self.samples.append((LABEL2IDX[label], text))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        label_idx, text = self.samples[i]
        # tokenizer 将文字转成 token 编号；truncation 截断超长输入。
        # input_ids 是编号列表，attention_mask 中 1 是有效位置，0 是补齐位置。
        enc = self.tokenizer(text, truncation=True, max_length=MAX_LEN)
        item = {}
        item["input_ids"] = enc["input_ids"]
        item["attention_mask"] = enc["attention_mask"]
        item["label"] = label_idx
        return item


def collate_fn(batch):
    # pad 将本批序列补到相同长度；return_tensors="pt" 返回 PyTorch 张量。
    # 张量可以理解成多维数组，此处形状是 [批大小, 序列长度]。
    features = tokenizer.pad(
        [{"input_ids": b["input_ids"], "attention_mask": b["attention_mask"]} for b in batch],
        return_tensors="pt",
    )
    labels = []
    for b in batch:
        labels.append(b["label"])
    features["labels"] = torch.tensor(labels)
    return features


class BertClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # from_pretrained 读取现成权重，优先使用本地缓存；AutoModel 不带本任务的分类头。
        self.backbone = AutoModel.from_pretrained(MODEL_NAME)
        hidden = self.backbone.config.hidden_size
        # Dropout 在训练时随机置零 10% 的元素；Linear 把标题向量变成 15 个类别分数。
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(hidden, 15)

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # 每个 token 对应一个 hidden 维向量，池化将整条标题变成一个向量。
        h = out.last_hidden_state  # [batch, seq_len, hidden]
        if POOLING == "cls":
            pooled = h[:, 0]
        elif POOLING == "mean":
            # 平均池化时排除 padding。
            mask = attention_mask.unsqueeze(-1).float()  # [batch, seq_len, 1]
            summed = (h * mask).sum(dim=1)  # [batch, hidden]
            counts = mask.sum(dim=1).clamp(min=1e-9)
            pooled = summed / counts
        elif POOLING == "max":
            # 排除 padding，避免它影响最大池化。
            mask = attention_mask.unsqueeze(-1).float()
            h = h.masked_fill(mask == 0, float("-inf"))
            pooled = h.max(dim=1).values
        else:
            raise ValueError("POOLING 只能是 cls / mean / max")
        return self.classifier(self.dropout(pooled))


def train_one_epoch(model, loader, optimizer, device, remaining_steps, started):
    model.train()
    total_loss = 0
    n = 0
    for batch in tqdm(loader, desc="训练", total=min(len(loader), remaining_steps)):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids, attention_mask)
        # cross_entropy 接收原始分数 logits 和正确类别下标，不需要先算 softmax。
        loss = torch.nn.functional.cross_entropy(logits, labels)

        # zero_grad 清空旧梯度，backward 计算梯度，step 按梯度更新参数。
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n += 1
        if n >= remaining_steps or time.monotonic() - started >= TRAIN_SECONDS:
            break
    return total_loss / n, n


# no_grad 关闭梯度记录以节省内存；eval 另行关闭 Dropout，两个操作作用不同。
@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    y_pred = []
    for batch in tqdm(loader, desc="预测"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        logits = model(input_ids, attention_mask)
        # argmax(-1) 找每行最高分的下标，tolist 转回 Python 列表。
        idx_list = logits.argmax(-1).tolist()
        for idx in idx_list:
            y_pred.append(IDX2LABEL[idx])
    return y_pred


def read_labels(tsv_path):
    rows = list(csv.reader(open(tsv_path, encoding="utf-8"), delimiter="\t"))[1:]
    labels = []
    for label, text in rows:
        labels.append(label)
    return labels


def main():
    global tokenizer
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("device:", device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_ds = TnewsDataset(DATA_DIR / "train.tsv", tokenizer)
    dev_ds = TnewsDataset(DATA_DIR / "dev.tsv", tokenizer)
    test_ds = TnewsDataset(DATA_DIR / "test.tsv", tokenizer)

    # batch_size 指一批多少条；shuffle 每轮打乱；collate_fn 负责补齐和拼成张量。
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_ds, batch_size=256, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=256, collate_fn=collate_fn)

    model = BertClassifier().to(device)
    # AdamW 更新 model.parameters() 中的参数，lr 是每次更新的学习率。
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    y_dev = read_labels(DATA_DIR / "dev.tsv")
    started = time.monotonic()
    steps = 0
    for epoch in range(EPOCHS):
        loss, n = train_one_epoch(model, train_loader, optimizer, device, MAX_STEPS - steps, started)
        steps += n
        print(f"\nepoch {epoch + 1}  平均loss={loss:.4f}")
        y_pred = predict(model, dev_loader, device)
        print(f"dev macro_f1 = {macro_f1(y_dev, y_pred):.4f}")

        if steps >= MAX_STEPS or time.monotonic() - started >= TRAIN_SECONDS:
            break
    print(f"实际训练 {steps} 步，耗时 {time.monotonic() - started:.1f} 秒")

    y_test = read_labels(DATA_DIR / "test.tsv")
    y_pred = predict(model, test_loader, device)
    print(f"\npooling={POOLING}")
    result = report(y_test, y_pred)
    result["train_steps"] = steps
    return result


if __name__ == "__main__":
    from common import prepare_quick_data
    prepare_quick_data()
    main()
