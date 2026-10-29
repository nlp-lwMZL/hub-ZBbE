# coding:utf8
# 解决 OpenMP 库冲突问题
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import numpy as np
import random

"""
基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：x是一个5维向量，取最大值首次出现的位置作为类别，共5类
输入经过最小-最大标准化，保留大小关系与并列关系
"""

class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.layer1 = nn.Linear(input_size, 32)  # 第一层全连接
        self.activation = nn.ReLU()  # 激活函数
        self.layer2 = nn.Linear(32, 5)  # 第二层全连接，输出五个类别的logits
        self.loss = nn.CrossEntropyLoss()  # 多分类交叉熵损失

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.layer1(x)  # (batch_size, input_size) -> (batch_size, 32)
        x = self.activation(x)  # (batch_size, 32) -> (batch_size, 32)
        y_pred = self.layer2(x)  # (batch_size, 32) -> (batch_size, 5)
        if y is not None:
            return self.loss(y_pred, y)  # 预测值和真实值计算损失
        else:
            return y_pred  # 输出五个类别的预测分数


# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
# 随机生成一个5维向量，最大值首次出现的位置即为类别
# 主动加入部分并列最大值样本，避免训练数据全是无并列的连续随机数
def build_sample():
    x = np.random.uniform(-10, 10, 5)
    tie_random = np.random.random()
    if tie_random < 0.02:
        x[:] = 0.5  # 五个数完全相同，按规则归入第0类
    elif tie_random < 0.22:
        # 随机选择一个位置放最大值，再在其后选一个位置并列
        first = np.random.randint(0, 4)
        second = np.random.randint(first + 1, 5)
        x[first] = 10.0
        x[second] = 10.0
    y = int(np.argmax(x))  # 并列时np.argmax取最先出现的位置
    return x, y


# 输入标准化：逐个向量做最小-最大标准化，不改变大小关系与并列关系
def normalize_vector(x):
    x = np.asarray(x, dtype=np.float64)
    min_value = np.min(x)
    max_value = np.max(x)
    if max_value == min_value:  # 五个数全部相等
        return np.zeros(5, dtype=np.float32)
    x = (x - min_value) / (max_value - min_value)
    return x.astype(np.float32)


# 随机生成一批样本
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        x = normalize_vector(x)  # 输入标准化，标签仍为原始向量的最大值位置
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(np.array(X)), torch.LongTensor(Y)  # 输入[样本数,5]，标签[样本数]


# 评估代码
# 用来在指定数据集上测试模型的准确率
def evaluate(model, x, y, set_name="验证集"):
    model.eval()
    total = len(y)
    class_counts = np.bincount(y.numpy(), minlength=5)
    print("[%s] 各类别样本数量：%s" % (set_name, class_counts.tolist()))
    with torch.no_grad():
        y_pred = model(x)  # 模型预测 model.forward(x)
        predictions = torch.argmax(y_pred, dim=1)  # 取分数最高的类别
        correct = (predictions == y).sum().item()
        wrong = total - correct
    print("[%s] 正确预测个数：%d, 错误预测个数：%d, 正确率：%f" % (set_name, correct, wrong, correct / total))
    return correct / total


def main():
    # 固定随机种子，保证结果可复现
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    # 配置参数
    epoch_num = 30  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 2000  # 每轮训练总共训练的样本总数
    val_sample = 500  # 验证集样本总数
    test_sample = 1000  # 测试集样本总数
    input_size = 5  # 输入向量维度
    learning_rate = 0.01  # 学习率
    # 建立模型
    model = TorchModel(input_size)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # 创建训练集、验证集、测试集（三份数据相互独立，避免数据泄露）
    train_x, train_y = build_dataset(train_sample)
    val_x, val_y = build_dataset(val_sample)
    test_x, test_y = build_dataset(test_sample)
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            # 取出一个batch数据作为输入
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
            loss = model(x, y)  # 计算loss  model.forward(x,y)
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        evaluate(model, val_x, val_y, set_name="验证集")  # 在验证集上评估本轮模型结果
    # 训练结束后，在测试集上做最终评估
    print("=========\n训练结束，在测试集上进行最终评估：")
    test_acc = evaluate(model, test_x, test_y, set_name="测试集")
    # 保存模型
    torch.save(model.state_dict(), "argmax_model.bin")
    print("模型已保存：argmax_model.bin，测试集准确率：%f" % test_acc)
    return


# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))  # 加载训练好的权重

    model.eval()  # 测试模式
    raw = np.asarray(input_vec, dtype=np.float64)
    if raw.ndim == 1:  # 支持传入单个五维向量
        raw = raw.reshape(1, -1)
    if raw.ndim != 2 or raw.shape[1] != 5:
        raise ValueError("每个输入样本必须包含五个数值。")
    if not np.all(np.isfinite(raw)):
        raise ValueError("输入不能包含NaN或无穷值。")
    x = np.array([normalize_vector(vec) for vec in raw])  # 与训练阶段相同的预处理

    with torch.no_grad():  # 不计算梯度
        logits = model(torch.FloatTensor(x))  # 模型预测
        probabilities = torch.softmax(logits, dim=1)  # 转为类别概率，仅用于展示
        predictions = torch.argmax(logits, dim=1)

    for vec, pred, prob in zip(raw, predictions, probabilities):
        true_label = int(np.argmax(vec))  # 规则计算结果
        print("输入：%s, 预测类别：第%d维最大, 规则类别：第%d维最大, 概率分布：%s"
              % (vec.tolist(), int(pred) + 1, true_label + 1, prob.numpy().round(4).tolist()))


if __name__ == "__main__":
    # 先训练模型
    main()

    # 再加载模型测试新样本
    test_vec = [
        [0.80815, 0.80811, 0.80813, 0.80814, 0.80815],
        [0.111, 0.108, 0.113, 0.104, 0.125],
        [0.871, 0.872, 0.869, 0.874, 0.868],
        [0.211, 0.222, 0.213, 0.219, 0.225],
        [0.1001, 0.1002, 0.1003, 0.1004, 0.10009],
        [0.9, 0.2, 0.9, 0.4, 0.5],
        [72.9, 71.9, 70.99, 71.9, 72.39]
    ]
    predict("argmax_model.bin", test_vec)
