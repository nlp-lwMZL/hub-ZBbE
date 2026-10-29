# 解决 OpenMP 库冲突问题
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt


# 模型定义
class TorchModule(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TorchModule, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.activation = nn.Sigmoid()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y = None):
        x = self.linear1(x)
        x = self.activation(x)
        y_pred = self.linear2(x)
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred

# 测试数据
# 输入5维向量，哪一维最大，即为对应分类
def build_sample():
    x = np.random.random(5)
    # y = np.zeros(5)
    # y[np.argmax(x)] = 1
    y = np.argmax(x)
    return x, y

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

# 测试函数
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    # y = torch.argmax(y, dim=1)

    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)
        pred = torch.argmax(y_pred,dim=1)
        for y_p, y_t in zip(pred, y):
            if y_p.item() == y_t.item():
                correct += 1
            else:
                wrong += 1
    acc = correct / (correct + wrong)
    print("正确预测个数：%d, 正确率：%f" % (correct, acc))
    return acc

def main():
    # 配置参数
    # 训练轮数  每次训练样本个数  总样本数
    epoch_num = 20
    batch_size = 32
    train_sample = 5000

    # 输入、隐藏、输出层
    input_size = 5
    hidden_size = 10
    output_size = 5

    # 学习率
    learning_rate = 0.01

    # 模型
    model = TorchModule(input_size, hidden_size, output_size)
    # 优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 训练集
    train_x, train_y = build_dataset(train_sample)

    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
            # 计算损失
            loss = model(x, y)
            loss.backward() # 计算梯度
            optim.step() # 更新权重
            optim.zero_grad() # 梯度归零
            watch_loss.append(loss.item())
        print(f'第{epoch + 1}轮平均loss：{np.mean(watch_loss)}')
        acc = evaluate(model)
        log.append([acc, float(np.mean(watch_loss))])

    # 保存模型
    torch.save(model.state_dict(), './model/model.pth')
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return

def predict(model_path, input_vec):
    input_size = 5
    hidden_size = 10
    output_size = 5
    model = TorchModule(input_size, hidden_size, output_size)
    # 加载参数
    model.load_state_dict(torch.load(model_path))

    model.eval()
    with torch.no_grad():
        result = model.forward(torch.FloatTensor(input_vec))
    for vec, res in zip(input_vec, result):
        print(f'输入{vec},输出各类概率为{torch.softmax(res, dim=0)},类别为{torch.argmax(res).item()}')

if __name__ == '__main__':
    # main()

    test_vec = [[0.81889086, 0.15229675, 0.31082123, 0.03504317, 0.88920843],
                [0.94963533, 0.5524256, 0.95758807, 0.95520434, 0.84890681],
                [0.90797868, 0.67482528, 0.13625847, 0.34675372, 0.19871392],
                [0.39349776, 0.59416669, 0.92579291, 0.41567412, 0.8358894]]
    predict("./model/model.pth", test_vec)


















