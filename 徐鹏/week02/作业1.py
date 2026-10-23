import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

"""
    对于任意5维向量，它最大数字出现在第几维就属于第几类
"""


class TorchModel(nn.Module):
    def __init__(self, input_size, class_num):
        super(TorchModel, self).__init__()
        # 线性层 5输入 -》5输出(5个类别的logits)
        self.linear = nn.Linear(input_size, class_num)
        self.loss = nn.functional.cross_entropy

    def forward(self, x, y=None):
        logits = self.linear(x)  # (batch, 5)
        if y is not None:
            return self.loss(logits, y)
        else:
            return logits


# 生成单个样本
def build_sample():
    x = np.random.random(5)
    # 最大值下标就是类别
    label = np.argmax(x)
    return x, label


# 批量生成数据集
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    # X:[N, 5] Y:[N] 标签数字0-4
    # 标签必须是 long 类型，cross_entropy 只接受整型标签
    return torch.from_numpy(np.array(X, dtype=np.float32)), torch.from_numpy(np.array(Y, dtype=np.int64))


# 评估函数
def evaluate(model):
    model.eval()
    test_sample_num = 200
    x, y = build_dataset(test_sample_num)
    correct = 0
    with torch.no_grad():
        logits = model(x)
        pred = torch.argmax(logits, dim=1)
        correct = torch.sum(pred == y).item()
    acc = correct / test_sample_num
    print(f"测试集正确: {correct}, 总数: {test_sample_num}, 准确率: {acc:.4f}")
    return acc


def main():
    # 参数
    epoch_num = 20
    batch_size = 20
    train_sample = 1000
    input_size = 5
    class_num = 5
    lr = 0.1

    # 固定随机种子，保证每次运行结果可复现
    np.random.seed(42)
    torch.manual_seed(42)

    model = TorchModel(input_size, class_num)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    log = []

    train_x, train_y = build_dataset(train_sample)

    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        # 按batch切片
        for batch_index in range(train_sample // batch_size):
            start = batch_index * batch_size
            end = start + batch_size
            x = train_x[start:end]
            y = train_y[start:end]

            loss = model(x, y)
            loss.backward()
            optim.step()
            optim.zero_grad()
            watch_loss.append(loss.item())

        mean_loss = np.mean(watch_loss)
        print(f"====第{epoch + 1}轮 avg loss:{mean_loss:.4f}")
        acc = evaluate(model)
        log.append([acc, mean_loss])

    # 保存模型
    torch.save(model.state_dict(), "multi_class_model.bin")

    # 用训练好的模型做预测演示
    test_vec = [list(np.random.random(5)) for _ in range(5)]
    predict("multi_class_model.bin", test_vec)

    # 画图
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")
    plt.legend()
    plt.show()


# 预测函数
def predict(model_path, input_vec):
    input_size = 5
    class_num = 5
    model = TorchModel(input_size, class_num)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    with torch.no_grad():
        logits = model(torch.FloatTensor(input_vec))
        prob = torch.softmax(logits, dim=1)
        pred = torch.argmax(logits, dim=1)
    for vec, p, pr in zip(input_vec, pred, prob):
        print(f"输入: {vec}, 预测类别: {int(p)}, 各类概率:{pr.numpy()}")


if __name__ == "__main__":
    main()
