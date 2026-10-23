# coding:utf8

# 解决 OpenMP 库冲突问题
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import matplotlib
matplotlib.use("Agg")  # 不弹窗，直接把训练曲线存成图片
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np

"""

基于pytorch框架编写模型训练
# 设计多分类任务模型，对于任意五维向量，它最大数字在第几维就属于第几类

任务：输入 5 维向量 x = [x0, x1, x2, x3, x4]，如果 xk 最大，则样本属于第 k 类（k = 0..4）。
     所以是 5 分类问题：模型的输出是 5 个类别的分数（logits），用交叉熵损失训练。

和 TorchDemo.py 的对比：
     二分类：Linear(5, 1) + sigmoid  -> 1 个概率值 + 均方差损失
     多分类：Linear(5, 5) + softmax  -> 5 个类别得分 + 交叉熵损失（softmax 已包含在 cross_entropy 里）

运行环境：/opt/miniconda3/envs/py312/bin/python TorchClassfication.py

"""


class TorchModel(nn.Module):
    def __init__(self, input_size, class_num):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, class_num)  # 线性层：5 维输入 -> 5 个类别得分
        self.loss = nn.functional.cross_entropy  # 多分类用交叉熵损失（内部自带 softmax）

    # 当输入真实标签，返回loss值；无真实标签，返回预测类别
    def forward(self, x, y=None):
        logits = self.linear(x)  # (batch_size, 5) -> (batch_size, 5)，每个类别一个得分
        if y is not None:
            return self.loss(logits, y)  # 预测得分和真实类别计算损失
        else:
            return torch.argmax(logits, dim=-1)  # 得分最大的那一维就是预测类别


# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
# 随机生成一个5维向量，样本类型为最大数字的索引
def build_sample():
    x = np.random.random(5)
    return x, np.argmax(x)


# 随机生成一批样本
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.LongTensor(Y)  # 标签是类别下标，必须是整型


# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model, verbose=False):
    model.eval()
    test_sample_num = 1000
    x, y = build_dataset(test_sample_num)
    if verbose:
        print("测试集各类别样本数（第0~4类）：", np.bincount(y.numpy(), minlength=5))
    with torch.no_grad():
        y_pred = model(x)  # 模型预测，返回每个样本的类别下标
        correct = (y_pred == y).sum().item()  # 与真实标签对比
    acc = correct / test_sample_num
    if verbose:
        print("正确预测个数：%d, 正确率：%f" % (correct, acc))
        # 每个类别各自的准确率（召回率）
        for c in range(5):
            mask = (y == c)
            print("  第%d类：样本 %d 个，预测正确 %d 个，召回率 %.3f"
                  % (c, mask.sum().item(), (y_pred[mask] == c).sum().item(),
                     (y_pred[mask] == c).float().mean().item()))
    return acc


def main():
    # 配置参数
    epoch_num = 200  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    class_num = 5  # 类别数（5 维中哪一维最大就是哪一类）
    learning_rate = 0.1  # 学习率
    # 建立模型
    model = TorchModel(input_size, class_num)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # 学习率逐渐衰减，后半程权重能长得更大，分类边界更准
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, epoch_num)
    log = []
    # 创建训练集，正常任务是读取训练集
    train_x, train_y = build_dataset(train_sample)
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            #取出一个batch数据作为输入   train_x[0:20]  train_y[0:20] train_x[20:40]  train_y[20:40]
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
            loss = model(x, y)  # 计算loss  model.forward(x,y)
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())
        scheduler.step()
        avg_loss = float(np.mean(watch_loss))
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, avg_loss])
        # 每 10 轮打印一次，避免刷屏
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print("第%3d轮  平均loss:%f  准确率:%f" % (epoch + 1, avg_loss, acc))

    print("========= 训练结束，最终测试：")
    final_acc = evaluate(model, verbose=True)
    print("（随机乱猜的准确率是 0.2，直接用 argmax 的准确率是 1.0）")

    # 保存模型
    # 注意：这里另存为 model_classify.bin，避免覆盖 TorchDemo.py 二分类任务用的 model.bin
    torch.save(model.state_dict(), "model_classify.bin")
    print("模型已保存到 model_classify.bin，最终准确率 %.4f" % final_acc)

    # 画图
    plt.figure(figsize=(7, 4.5))
    plt.plot(range(1, len(log) + 1), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(1, len(log) + 1), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.xlabel("epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig("classify_acc_loss.png", dpi=120)  # 存在脚本同目录下
    plt.close()
    print("训练曲线已保存到 classify_acc_loss.png")
    return


# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5
    class_num = 5
    model = TorchModel(input_size, class_num)
    model.load_state_dict(torch.load(model_path, weights_only=True))  # 加载训练好的权重
    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        x = torch.FloatTensor(input_vec)
        result = model(x)  # 预测类别
        probs = torch.softmax(model.linear(x), dim=-1)  # 顺便看下各类别概率
    print("========= 预测结果：")
    for vec, res, prob in zip(input_vec, result, probs):
        res = int(res)  # 真实答案：数值最大的那一维
        true_class = int(np.argmax(vec))
        print("输入：%s" % np.round(vec, 4))
        print("      各类别概率：%s" % np.round(prob.numpy(), 3))
        print("      预测类别：%d（第%d维最大，概率 %.3f）, 真实类别：%d, %s"
              % (res, res + 1, prob[res], true_class, "正确" if res == true_class else "错误"))
    return


if __name__ == "__main__":
    main()
    test_vec = [[0.81889086,0.15229675,0.31082123,0.03504317,0.88920843],
                [0.94963533,0.5524256,0.95758807,0.95520434,0.84890681],
                [0.90797868,0.67482528,0.13625847,0.34675372,0.19871392],
                [0.39349776,0.59416669,0.92579291,0.41567412,0.8358894]]
    predict("model_classify.bin", test_vec)
