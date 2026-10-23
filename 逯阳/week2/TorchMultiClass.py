# coding:utf8

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

"""
numpy实现一个多分类任务
多分类任务---损失函数采用交叉熵，激活函数使用softmax
"""

# 搭建一个2层神经网络，每层都是线性层
class TorchMultiClass(nn.Module):
    def __init__(self, input_size, class_num):
        super().__init__()
        self.linear1 = nn.Linear(input_size, 10)  # 线性层
        self.linear2 = nn.Linear(10, class_num)  # 输出类别值
        self.loss = nn.CrossEntropyLoss()  # 多分类交叉熵损失

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.linear1(x)  # 先过第1层
        x = torch.relu(x)  # 中间加激活
        y_pred = self.linear2(x)
        if y is not None:
            return self.loss(y_pred, y)   # 预测值和真实值计算损失
        else:
            return y_pred

# 生成一个样本：1x5数组，最大值位置为类别编号
def build_sample():
    x = np.random.random(5)                    # 生成5维随机数组
    c = int(np.argmax(x))                       # 最大值所在位置 = 类别
    return x, c

# 生成一个样本：3个高斯簇，二维输入（已注释）
# def build_sample():
#     centers = np.array([[0.0, 0.0],
#                         [2.0, 2.0],
#                         [2.0, -2.0]])
#     c = np.random.randint(0, 3)               # 随机选一个类别
#     x = centers[c] + np.random.randn(2) * 0.5  # 该类别附近采样
#     return x, c

# 随机生成一批样本
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(np.array(X)), torch.LongTensor(np.array(Y))

# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    # 标记模型是评估
    model.eval()
    # 测试100条数据
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    # 统计各类样本数量
    for c in range(5):
        print("第%d类样本：%d个" % (c, (y == c).sum().item()))
    correct, wrong = 0, 0
    # 关闭梯度
    with torch.no_grad():
        y_pred = model(x)  # 模型预测 model.forward(x)
        for x_i, y_p, y_t in zip(x, y_pred, y):  # 与真实标签进行对比
            if int(torch.argmax(y_p)) == y_t:
                correct += 1
            else:
                print(f"预测结果是{torch.argmax(y_p)}, 实际结果是{y_t}, {x_i}, {y_p}")
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


if __name__ == '__main__':
    input_size = 5  # 输入向量维度（5维数组）
    class_num = 5 # 类别数量
    # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    learning_rate = 0.01  # 学习率

    # 构建模型 5 -> 10
    model = TorchMultiClass(input_size, class_num)
    # 优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 生成训练集数据
    x, y = build_dataset(train_sample)
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for i in range(train_sample // batch_size):
            # 取出一个batch数据作为输入 0:20
            xi = x[i * batch_size:(i + 1) * batch_size]
            yi = y[i * batch_size:(i + 1) * batch_size]
            # 计算损失
            loss = model(xi, yi)
            # 计算梯度
            loss.backward()
            # 更新权重
            optim.step()
            # 梯度归零
            optim.zero_grad()
            # 添加损失值
            watch_loss.append(loss.item())
        # 打印损失值
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        # 本轮预测结果
        acc = evaluate(model)
        # 放到
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    torch.save(model.state_dict(), "model.bin")
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()