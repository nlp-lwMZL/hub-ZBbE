# coding:utf8

# 解决 OpenMP 库冲突问题
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import numpy as np
import time
import shutil
import matplotlib.pyplot as plt

"""

基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：x是一个5维向量，哪个维度上的数值最大，就属于哪一类（共5类）
      标签 0~4 依次对应第1~5维最大，即多分类任务

"""


class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, input_size)  # 线性层，输出维度=类别数
        self.loss = nn.functional.cross_entropy  # 多分类采用交叉熵损失（内部自带softmax）

    # 当输入真实标签，返回loss值；无真实标签，返回各类别得分(logits)
    def forward(self, x, y=None):
        y_pred = self.linear(x)  # (page_size, input_size) -> (page_size, 类别数)
        if y is not None:
            return self.loss(y_pred, y)  # logits与类别标签计算损失
        else:
            return y_pred  # 输出各类别得分


# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
# 随机生成一个5维向量，最大值的下标即为该样本的类别（0~4）
def build_sample():
    x = np.random.random(5)
    y = int(np.argmax(x))  # 最大值所在维度下标作为类别
    return x, y


# 随机生成一批样本
def build_dataset(total_sample_num):
    X = []
    Y = []
    for _ in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)  # 多分类标签为整数，形状 (N,)
    return torch.FloatTensor(X), torch.LongTensor(Y)

# 测试代码，用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    class_count = [int((y == i).sum()) for i in range(5)]  # 统计各类别样本数量
    print("本次预测集中各类别样本数(第1~5维最大)：%s" % class_count)
    correct = 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测 model.forward(x)
        y_pred = y_pred.argmax(dim=1)  # 取得分最大的类别作为预测类别
        for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比
            if int(y_p) == int(y_t):
                correct += 1  # 类别判断正确
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / test_sample_num))
    return correct / test_sample_num


def main():
    # 配置参数
    epoch_num = 20  # 训练轮数
    page_size = 20  # 每次训练样本个数
    train_sample = 500  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    learning_rate = 0.01  # 学习率
    # 建立模型
    model = TorchModel(input_size)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 创建训练集，正常任务是读取训练集
    train_x, train_y = build_dataset(train_sample)
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for page_index in range(train_sample // page_size): 
            #取出一个batch数据作为输入   train_x[0:20]  train_y[0:20] train_x[20:40]  train_y[20:40]
            page_start = page_index * page_size
            page_end = (page_index + 1) * page_size
            x = train_x[page_start:page_end]
            y = train_y[page_start:page_end]
            loss = model(x, y)  # 计算loss  model.forward(x,y)
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型：先备份旧权重再覆盖，避免新存档异常时无法回滚
    if os.path.exists("model.bin"):
        shutil.copyfile("model.bin", "model_backup_%s.bin" % time.strftime("%Y%m%d_%H%M%S"))
    torch.save(model.state_dict(), "model.bin")
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.savefig("train_curve.png")  # 保存曲线图片，避免plt.show()阻塞
    plt.close()


# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测，得到各类别得分
    result = torch.softmax(result, dim=1)  # 得分转概率
    for vec, res in zip(input_vec, result):
        pred = int(res.argmax())  # 概率最大的类别
        print("输入：%s, 预测类别：%d(第%d维最大), 概率值：%f" % (np.round(vec, 2), pred, pred + 1, float(res[pred])))  # 打印结果


if __name__ == "__main__":
    main()
    # 测试样本，每类各一条：最大值分别落在第1~5维，便于核对预测结果
    test_vec = [[0.91370000,0.12480000,0.36250000,0.04190000,0.58060000],  # 第1维最大 -> 类别0
                [0.21360000,0.88420000,0.34170000,0.49730000,0.27050000],  # 第2维最大 -> 类别1
                [0.11740000,0.45280000,0.96030000,0.33910000,0.22870000],  # 第3维最大 -> 类别2
                [0.28650000,0.13420000,0.40190000,0.89570000,0.52380000],  # 第4维最大 -> 类别3
                [0.43820000,0.26170000,0.63140000,0.31050000,0.93160000]]  # 第5维最大 -> 类别4
    predict("model.bin", test_vec)
