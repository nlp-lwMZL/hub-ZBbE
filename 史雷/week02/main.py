# coding:utf8

# 解决 OpenMP 库冲突问题
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""

基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：x是一个5维向量，最大值在第几维，就属于第几类（共5类）

标签说明：
    内部标签使用 0~4，分别对应第 1~5 类
    打印结果时统一 +1，显示为"第几类"

"""


class TorchModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, output_size)  # 线性层，5维输入 -> 5类输出
        self.loss = nn.functional.cross_entropy  # 多分类使用交叉熵损失

    # 当输入真实标签，返回loss值；无真实标签，返回预测概率
    def forward(self, x, y=None):
        logits = self.linear(x)  # (batch_size, input_size) -> (batch_size, 5)
        if y is not None:
            # 交叉熵函数内部自带softmax，所以这里直接传logits
            return self.loss(logits, y)  # 预测值和真实值计算损失
        else:
            return torch.softmax(logits, dim=1)  # 输出每一类的概率


# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
# 随机生成一个5维向量，最大值在第几维，就属于第几类
def build_sample():
    x = np.random.random(5)
    y = int(np.argmax(x))  # 最大值所在维度下标，0~4
    return x, y


# 随机生成一批样本
# 由于5维向量随机均匀生成，每一维成为最大值的概率相同，所以各类样本大致均匀
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    # print(X)
    # print(Y)
    return torch.FloatTensor(X), torch.LongTensor(Y)


# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    counts = np.bincount(y.numpy(), minlength=5)
    print("本次预测集中第1~5类样本数分别为：%s" % counts.tolist())
    correct = 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测 model.forward(x)，输出各类概率
        pred = torch.argmax(y_pred, dim=1)  # 取概率最大的类别
        correct = int((pred == y).sum())  # 与真实标签进行对比
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / test_sample_num))
    return correct / test_sample_num


def main():
    # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    output_size = 5  # 类别数
    learning_rate = 0.01  # 学习率
    # 建立模型
    model = TorchModel(input_size, output_size)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 创建训练集，正常任务是读取训练集
    train_x, train_y = build_dataset(train_sample)
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
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    torch.save(model.state_dict(), "model.bin")
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return


# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5
    output_size = 5
    model = TorchModel(input_size, output_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测，输出各类概率
    for vec, res in zip(input_vec, result):
        res = res.numpy()
        true_cls = int(np.argmax(vec)) + 1  # 真实类别：最大值所在维度+1
        pred_cls = int(np.argmax(res)) + 1  # 预测类别：概率最大的类别+1
        print("输入：%s, 最大值在第%d维(真实第%d类), 预测类别：第%d类, 各类概率：%s"
              % (vec, true_cls, true_cls, pred_cls, np.round(res, 4)))


if __name__ == "__main__":
    main()
    test_vec = [[0.81889086, 0.15229675, 0.31082123, 0.03504317, 0.88920843],
                [0.94963533, 0.5524256, 0.95758807, 0.95520434, 0.84890681],
                [0.90797868, 0.67482528, 0.13625847, 0.34675372, 0.19871392],
                [0.39349776, 0.59416669, 0.92579291, 0.41567412, 0.8358894]]
    predict("model.bin", test_vec)