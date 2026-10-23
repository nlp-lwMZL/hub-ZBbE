# 解决 OpenMP 库冲突问题
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

"""

基于pytorch框架编写模型训练（多分类版本）
实现一个自行构造的找规律(机器学习)任务
规律：x是一个5维向量，最大值所在的维度下标即为类别（共5类：0~4）

与二分类版本(TorchDemo.py)的核心区别：
1. 标签从 0/1 变为 0~4 的类别索引，使用 LongTensor
2. 输出层从 1 维变为 num_classes(5) 维，每个维度对应一个类别的得分
3. sigmoid + mse_loss 变为 cross_entropy（内部已含softmax，forward计算loss时不再加激活函数）
4. 评估与预测时用 argmax 取得分最高的类别作为预测结果

"""


class TorchModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.linear = nn.Linear(input_size, num_classes)  # 线性层，输出维度=类别数
        self.loss = nn.functional.cross_entropy  # 多分类采用交叉熵损失，内部包含log_softmax

    # 当输入真实标签，返回loss值；无真实标签，返回预测值（logits）
    def forward(self, x, y=None):
        logits = self.linear(x)  # (batch_size, input_size) -> (batch_size, num_classes)
        if y is not None:
            return self.loss(logits, y)  # 预测值和真实值计算损失
        return logits  # 输出原始得分，需要概率时外部再套softmax


# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
# 随机生成一个5维向量，最大值所在的下标即为类别
def build_sample():
    x = np.random.random(5)
    return x, int(np.argmax(x))


# 随机生成一批样本
# 各类别样本数基本均匀（随机向量取最大值下标，各类概率相同）
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    X = np.array(X)
    Y = np.array(Y)
    out_X = torch.FloatTensor(X).to(device)
    out_Y = torch.LongTensor(Y).to(device)  # 交叉熵要求标签为long类型的类别索引，shape=(batch_size,)
    print(f"out_X.shape = {out_X.shape}, out_Y.shape = {out_Y.shape}")
    return out_X, out_Y


# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    label_count = [int(torch.sum(y == i)) for i in range(5)]
    print("本次预测集中各类样本个数：", label_count)
    correct, wrong = 0, 0
    with torch.no_grad():
        logits = model(x)  # 模型预测 model.forward(x) -> (100, 5)
        pred = torch.argmax(logits, dim=-1)  # 取得分最高的类别作为预测结果
        for y_p, y_t in zip(pred, y):  # 与真实标签进行对比
            if int(y_p) == int(y_t):
                correct += 1  # 类别判断正确
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    num_classes = 5  # 类别数
    learning_rate = 0.01  # 学习率
    # 建立模型
    model = TorchModel(input_size, num_classes).to(device)
    print("模型设备：", next(model.parameters()).device)
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
            # 取出一个batch数据作为输入   train_x[0:20]  train_y[0:20] train_x[20:40]  train_y[20:40]
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
    torch.save(model.state_dict(), "model_multi_class.bin")
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
    num_classes = 5
    model = TorchModel(input_size, num_classes).to(device)
    print("模型设备：", next(model.parameters()).device)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        tensor_input = torch.FloatTensor(input_vec).to(device)
        logits = model(tensor_input)  # 模型输出logits
        probs = torch.softmax(logits, dim=-1)  # 转换为概率分布 (batch_size, 5)
        pred = torch.argmax(probs, dim=-1)  # 取概率最大的类别
    for vec, label, prob in zip(input_vec, pred, probs):
        print("输入：%s, 预测类别：%d, 概率值：%f" % (vec, int(label), float(prob[label])))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    # main()
    test_vec = [
        [0.9, 0.1, 0.2, 0.3, 0.4],  # 最大值在第1个位置，应预测类别0
        [0.1, 0.9, 0.2, 0.3, 0.4],  # 最大值在第2个位置，应预测类别1
        [0.1, 0.2, 0.3, 0.9, 0.4],  # 最大值在第4个位置，应预测类别3
        [0.1, 0.2, 0.3, 0.4, 0.9],  # 最大值在第5个位置，应预测类别4
    ]
    predict("model_multi_class.bin", test_vec)
