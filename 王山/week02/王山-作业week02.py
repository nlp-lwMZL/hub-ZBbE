# coding:utf8
# 解决 OpenMP 库冲突问题
import os
import torch.sparse
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt
"""规则（自己生成数据，不用外部数据集）：
输入是6 维随机向量 x = [x0,x1,x2,x3,x4,x5]，每个元素 0~1 之间。
如果 x0 是 6 个数字里面最大值 → 类别 0
如果 x1 是 6 个数字里面最大值 → 类别 1
如果 x2 是 6 个数字里面最大值 → 类别 2
如果 x3 是 6 个数字里面最大值 → 类别 3
如果 x4 是 6 个数字里面最大值 → 类别 4
如果 x5 是 6 个数字里面最大值 → 类别 5
比如：x=[0.8,0.3,0.2,0.1,0.05,0.4]，最大值是 x0=0.8 → 标签 0
"""
class TestModel(nn.Module):
    def __init__(self, input_size=6, h1=64, h2=32, out_size=6):
        super(TestModel, self).__init__()
        self.layer1 = nn.Linear(input_size, h1)
        self.layer2 = nn.Linear(h1, h2)
        self.layer3 = nn.Linear(h2, out_size)
        self.activation = nn.ReLU(inplace=True)
        self.loss = nn.CrossEntropyLoss()
    def forward(self, x, y=None):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        x = self.activation(x)
        y_pred = self.layer3(x)
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred
# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
# 随机生成一个6维向量，只比较前 3 维（x3~x5 是干扰特征，不参与判定）
def build_sample():
    x = np.random.random(6)
    label = int(np.argmax(x))  # 并列最大值时取第一个下标
    if label == 0:
        return x, 0
    elif label == 1:
        return x, 1
    elif label == 2:
        return x, 2
    elif label == 3:
        return x, 3
    elif label == 4:
        return x, 4
    elif label == 5:
        return x, 5

# 随机生成一批样本
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    # 修复1：X 由 float64 的 ndarray 组成，直接 torch.tensor 会得到 Double，
    # 与 Linear 层默认的 Float 权重不匹配，会报
    # "mat1 and mat2 must have the same dtype, but got Double and Float"
    # 先用 np.array 聚合成一个数组（同时避免慢速构造的警告），再显式指定 dtype。
    X_tensor = torch.tensor(np.array(X), dtype=torch.float32)
    # 修复2：CrossEntropyLoss 的目标必须是 long（int64），给 Y 显式指定。
    Y_tensor = torch.tensor(Y, dtype=torch.long)
    return X_tensor, Y_tensor
def evaluate(model):
    model.eval()
    total_sample_num = 150
    x, y = build_dataset(total_sample_num)
    correct1, correct2, correct3, correct4,correct5,correct6,wrong = 0, 0, 0, 0, 0, 0, 0
    with torch.no_grad():
        y_pred = model(x)
        y_pred_3 = torch.argmax(y_pred, dim=1)
        for y_p, y_t in zip(y_pred_3, y):
            y_p = int(y_p)
            y_t = int(y_t)
            if y_p == 0 and y_t == 0:
                correct1 += 1
            elif y_p == 1 and y_t == 1:
                correct2 += 1
            elif y_p == 2 and y_t == 2:
                correct3 += 1
            elif y_p == 3 and y_t == 3:
                correct4 += 1
            elif y_p == 4 and y_t == 4:
                correct5 += 1
            elif y_p == 5 and y_t == 5:
                correct6 += 1
            else:
                wrong += 1
    # 修复3：计数是整数，用 %d 打印（原来用 %f 会印出 "12.000000个"）
    print("本次预测共%d个样本，其中0类正确%d个，1类正确%d个，2类正确%d个，3类正确%d个,4类正确%d个,5类正确%d个,错误%d个"
          % (total_sample_num, correct1, correct2, correct3,correct4,correct5,correct6, wrong))
    total_correct = correct1 + correct2 + correct3 + correct4 +correct5 + correct6
    print(f"准确率为{(total_correct / total_sample_num):.4f}")
    return total_correct / total_sample_num
def main():
    # 配置参数
    epoch_num = 50  # 训练轮数
    batch_size = 50  # 单批次样本数量
    train_sample = 50000  # 总训练样本数量
    input_size = 6  # 输入向量维度
    learning_rate = 0.005  # 学习率
    # 建立模型
    model = TestModel(input_size=input_size)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 创建训练集，正常任务是读取训练集
    train_x, train_y = build_dataset(train_sample)
    # 修复4：模型保存到脚本所在目录，避免换个工作目录就找不到 / 到处散落
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_test.bin")
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size: (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size: (batch_index + 1) * batch_size]
            loss = model(x, y)  # 计算loss  model.forward(x,y)
            loss.backward()
            optim.step()
            optim.zero_grad()
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    torch.save(model.state_dict(), save_path)
    print("模型已保存到:", save_path)
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return save_path
def predict(model_path, input_vec):
    model = TestModel(input_size=6)
    # 修复5：PyTorch 2.6 起 torch.load 默认 weights_only=True，
    # 显式写出来更稳；map_location 保证在只有 CPU 的机器上也能加载。
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()  # 测试模式
    with torch.no_grad():
        # 修复6：显式 float32，和训练时保持一致
        result = model.forward(torch.tensor(input_vec, dtype=torch.float32))
    for vec, res in zip(input_vec, result):
        pre_cls = int(torch.argmax(res).item())
        gl = torch.softmax(res, dim=0).tolist()
        # 修复7：原来打印的是整个 input_vec，应打印当前这条 vec
        print(f"输入：{list(vec)}, 预测类别：{pre_cls}, 概率值：{[round(p, 4) for p in gl]}")

if __name__ == "__main__":
    model_path = main()
    test_vec = [[0.81889086, 0.15229675, 0.31082123, 0.03504317, 0.88920843, 0.23404353],
                [0.94963533, 0.5524256, 0.95758807, 0.95520434, 0.84890681, 0.47385832],
                [0.90797868, 0.67482528, 0.13625847, 0.34675372, 0.19871392, 0.37485928],
                [0.39349776, 0.59416669, 0.92579291, 0.41567412, 0.8358894, 0.33345674]]
    predict(model_path, test_vec)
