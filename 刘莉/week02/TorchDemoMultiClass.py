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
改造为多分类任务（4分类），使用交叉熵损失CrossEntropyLoss
规律：x是一个5维向量，取前4个维度，哪个维度数值最大，则类别为该维度下标(0,1,2,3)
"""

class TorchModel(nn.Module):
    def __init__(self, input_size, class_num):
        super(TorchModel, self).__init__()
        # 线性层：输入input_size，输出类别数目，CrossEntropyLoss输出logits，不加softmax
        self.linear = nn.Linear(input_size, class_num)
        # CrossEntropyLoss自带softmax，网络层不写激活
        self.loss = nn.CrossEntropyLoss()

    # 当输入真实标签，返回loss值；无真实标签，返回预测logits
    def forward(self, x, y=None):
        x = self.linear(x)  # (batch_size, input_size) -> (batch_size, class_num)
        if y is not None:
            return self.loss(x, y)  # y是类别索引，不是onehot
        else:
            return x  # 返回logits，外部做softmax得到概率

# 生成一个样本
# 随机生成5维向量，前4维最大值下标作为分类标签0/1/2/3
def build_sample():
    x = np.random.random(5)
    # 取前4个值
    first_four = x[:4]
    label = int(np.argmax(first_four))
    return x, label

# 随机生成一批样本
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)  # 标签直接存类别数字，不要套[]变成二维
    # X：float张量；Y：CrossEntropyLoss需要long类型
    return torch.FloatTensor(X), torch.LongTensor(Y)

# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    # 统计各类样本数量
    class_counts = np.bincount(y.numpy(), minlength=4)
    print(f"本次预测集各类样本数量：0:{class_counts[0]},1:{class_counts[1]},2:{class_counts[2]},3:{class_counts[3]}")
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred_logits = model(x)  # (batch,4) logits
        y_pred = torch.argmax(y_pred_logits, dim=1) # 取最大概率下标作为预测类别
        for y_p, y_t in zip(y_pred, y):
            if int(y_p) == int(y_t):
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)

def main():
    # 配置参数
    epoch_num = 30  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 1000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    class_num = 4  # 分类数目
    learning_rate = 0.005  # 学习率
    # 建立模型
    model = TorchModel(input_size, class_num)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 创建训练集
    train_x, train_y = build_dataset(train_sample)
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
            loss = model(x, y)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    torch.save(model.state_dict(), "multi_model.bin")
    # 画图
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return

# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5
    class_num = 4
    model = TorchModel(input_size, class_num)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        logits = model.forward(torch.FloatTensor(input_vec))  # (N,4)
        probs = torch.softmax(logits, dim=1) # 转成概率
        pred_classes = torch.argmax(probs, dim=1)
    for vec, pred_cls, prob in zip(input_vec, pred_classes, probs):
        print(f"输入：{vec}, 预测类别：{int(pred_cls)}, 各类概率：{prob.numpy()}")

if __name__ == "__main__":
    main()
    test_vec = [[0.81889086,0.15229675,0.31082123,0.03504317,0.88920843],
                [0.14963533,0.9524256,0.05758807,0.05520434,0.84890681],
                [0.10797868,0.17482528,0.93625847,0.34675372,0.19871392],
                [0.09349776,0.09416669,0.12579291,0.91567412,0.8358894]]
    predict("multi_model.bin", test_vec)
