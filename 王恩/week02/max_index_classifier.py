# coding:utf-8
"""
任务目标：学习五维向量中最大值所在的位置。
运行：python max_index_classifier.py
依赖：torch。运行后在本文件所在目录保存权重和训练日志。
内部标签为 0～4，对外展示为第 1～5 类。
生成标签时，若最大值并列，约定取最先出现的位置。
神经网络是近似模型，对几乎相等或并列的最大值不保证判断正确。
"""

import json
from pathlib import Path

import torch
from torch import nn


INPUT_SIZE = 5
OUTPUT_DIR = Path(__file__).resolve().parent
MODEL_PATH = OUTPUT_DIR / "max_index_model.pt"


class TorchModel(nn.Module):
    def __init__(self, input_size=INPUT_SIZE):
        super().__init__()
        # 此任务的分类边界是 x_i = x_j，使用一个线性层即可学习。
        self.linear = nn.Linear(input_size, input_size)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        # 每个向量单独归一化，保留元素大小顺序，适应负数和不同尺度。
        # 先缩放再中心化，避免较大数值在相减时溢出。
        scale = x.abs().amax(dim=-1, keepdim=True)
        x = x / scale.clamp_min(torch.finfo(x.dtype).tiny)
        x = x - x.mean(dim=-1, keepdim=True)
        logits = self.linear(x)
        if y is not None:
            # 交叉熵直接接收原始分数，无需事先执行 softmax。
            return self.loss(logits, y)
        return logits


def build_dataset(total_sample_num, generator=None):
    """随机生成五维向量，并以最大值的下标作为真实类别。"""
    x = torch.rand(total_sample_num, INPUT_SIZE, generator=generator) * 20 - 10
    # CrossEntropyLoss 要求标签为形状 (样本数,) 的 long 整数张量。
    y = x.argmax(dim=1)
    return x, y


def evaluate(model, x, y):
    model.eval()
    with torch.no_grad():
        logits = model(x)
        loss = model.loss(logits, y).item()
        accuracy = (logits.argmax(dim=1) == y).float().mean().item()
    return loss, accuracy


def train(epoch_num=40, batch_size=128, train_sample_num=10000,
          validation_sample_num=2000, learning_rate=0.03):
    torch.manual_seed(42)
    # 小矩阵使用单线程，避免线程调度开销。
    torch.set_num_threads(1)
    model = TorchModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # 验证集独立生成，使用独立随机数发生器，且不参与梯度更新。
    validation_generator = torch.Generator().manual_seed(2026)
    validation_x, validation_y = build_dataset(
        validation_sample_num, validation_generator)
    history = []

    initial_loss, initial_accuracy = evaluate(model, validation_x, validation_y)
    print(f"训练前：验证 loss={initial_loss:.4f}，准确率={initial_accuracy:.2%}")
    for epoch in range(epoch_num):
        model.train()
        # 每轮重新随机采样，增加模型见到的向量数量。
        train_x, train_y = build_dataset(train_sample_num)
        total_loss = 0.0
        for start in range(0, train_sample_num, batch_size):
            x = train_x[start:start + batch_size]
            y = train_y[start:start + batch_size]
            optimizer.zero_grad()
            loss = model(x, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(x)

        validation_loss, accuracy = evaluate(model, validation_x, validation_y)
        train_loss = total_loss / train_sample_num
        history.append({"epoch": epoch + 1, "train_loss": train_loss,
                        "validation_loss": validation_loss,
                        "validation_accuracy": accuracy})
        print(f"第 {epoch + 1:02d} 轮：训练 loss={train_loss:.4f}，"
              f"验证 loss={validation_loss:.4f}，准确率={accuracy:.2%}")

    # 测试集与训练集、验证集独立，训练完成后仅评估一次。
    test_generator = torch.Generator().manual_seed(2027)
    test_x, test_y = build_dataset(10000, test_generator)
    test_loss, test_accuracy = evaluate(model, test_x, test_y)
    print(f"独立测试集（10000 条）：loss={test_loss:.4f}，准确率={test_accuracy:.2%}")
    torch.save(model.state_dict(), MODEL_PATH)
    log = {"seed": 42, "epochs": epoch_num, "batch_size": batch_size,
           "learning_rate": learning_rate, "train_samples_per_epoch": train_sample_num,
           "validation_samples": validation_sample_num, "test_samples": len(test_x),
           "initial_validation_accuracy": initial_accuracy,
           "history": history, "test_loss": test_loss, "test_accuracy": test_accuracy}
    (OUTPUT_DIR / "training_log.json").write_text(
        json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"模型已保存：{MODEL_PATH}")
    return model


def predict(model_path, input_vec):
    """加载训练权重，返回类别（1～5）和五个类别的概率。"""
    # float64 转换、检查和缩放后再转 float32，减少大数值溢出的风险。
    x = torch.as_tensor(input_vec, dtype=torch.float64)
    if x.ndim == 1:
        x = x.unsqueeze(0)
    if x.ndim != 2 or x.shape[1] != INPUT_SIZE or len(x) == 0:
        raise ValueError("输入必须是一个五维向量，或非空的五维向量列表")
    if not torch.isfinite(x).all():
        raise ValueError("向量中的数字必须是有限实数")
    scale = x.abs().amax(dim=1, keepdim=True)
    x = (x / scale.clamp_min(torch.finfo(x.dtype).tiny)).float()
    model = TorchModel()
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()
    with torch.no_grad():
        probabilities = torch.softmax(model(x), dim=1)
        classes = probabilities.argmax(dim=1) + 1
    return classes.tolist(), probabilities.tolist()


def main():
    train()
    test_vec = [[9, 2, 3, 4, 5], [1, 9, 3, 4, 5], [1, 2, 9, 4, 5],
                [1, 2, 3, 9, 5], [1, 2, 3, 4, 9], [-8, -2, -5, -6, -7],
                [1000, 2000, 9000, 4000, 5000]]

    classes, probabilities = predict(MODEL_PATH, test_vec)
    for vec, category, probability in zip(test_vec, classes, probabilities):
        expected = vec.index(max(vec)) + 1
        print(f"输入：{vec}，预测：第 {category} 类，真实：第 {expected} 类，"
              f"预测置信度：{probability[category - 1]:.2%}")


if __name__ == "__main__":
    main()
