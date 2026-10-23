# 用numpy做一个多分类模型
# x一个5维向量，最大值在第几维，就属于第几类，一共5类

import numpy as np
import matplotlib.pyplot as plt

# 参考课件，把用到的几个函数加进来
# 定义一个sigmoid激活函数 - 把任意实数正的、负的、很大的、很小的）压缩到0 到 1 之间，输出可以理解为一个 "概率值"。
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 定义一个函数softmax - 把任意一组实数转化成总和为 1 的概率分布，每个类别的输出值在 0~1 之间，表示 "属于该类别的概率"。
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

# # 定义One-Hot标签矩阵函数
# def to_one_hot(target, shape):
#     one_hot_target = np.zeros(shape)
#     for i, t in enumerate(target):
#         one_hot_target[i][t] = 1
#     return one_hot_target

# 定义交叉熵损失函数 - 预测越准，损失值越小；预测越偏，损失值越大。训练让损失值尽可能小。
def corss_entropy(pred, target):
    batch_size, clase_num = pred.shape
    pred = softmax(pred)
    target = to_one_hot(target, pred.shape)
    entropy = - np.sum(target * np.log(pred), axis=1)
    return sum(entropy) / batch_size

# 输入层： 5维向量就是5； 输出层：5个类别也是5； 隐藏层用输入维度的2-3倍？；  学习率0.01/0.1； 批次batchsize 32
# 定义一个模型

class DiyModel:
    def __init__(self, input_size, hidden_size, class_num):
        # 权重随机，简单数据，按照课件先用0.1
        self.w1 = np.random.randn(hidden_size, input_size) * 0.1
        self.b1 = np.zeros(hidden_size)
        self.w2 = np.random.randn(class_num, hidden_size) * 0.1
        self.b2 = np.zeros(class_num)

    def forward(self, x):
        hidden = np.dot(x, self.w1.T) + self.b1
        # hidden = sigmoid(hidden)          # 用sigmoid激活
        hidden = np.maximum(0, hidden)      # 用ReLU

        y_pred = np.dot(hidden, self.w2.T) + self.b2
        return hidden, y_pred

    def backward(self, x, hidden, y_pred, y_true):
        batch_size = len(y_true)
        p = softmax(y_pred)

        dz2 = p.copy()                      # copy预测概率
        for i in range(batch_size):
            dz2[i][y_true[i]] -= 1          # 正确一列减1
        dz2 = dz2 / batch_size              # 取平均

        dw2 = np.dot(dz2.T, hidden)         # 误差
        db2 = np.sum(dz2, axis=0)           # 按类别求和

        dhidden = np.dot(dz2, self.w2)
        # dz1 = dhidden * hidden * (1 - hidden) # 用sigmoid激活
        dz1 = dhidden * (hidden > 0)        # 用ReLU
        dw1 = np.dot(dz1.T, x)
        db1 = np.sum(dz1, axis=0)
        return dw1, db1, dw2, db2

# 定义一个随机生成样本的函数
def creat_sample():
    x = np.random.random(5)
    y = np.argmax(x)            # y是最大值在第几维就是第几类
    return x, y

# 定义一个随机生成批量样本的函数
def creat_datasanples(sample_num):
    X = []
    Y = []
    for i in range(sample_num):
        x, y = creat_sample()
        X.append(x)
        Y.append(y)
    # print(X[:5], Y[:5])
    return np.array(X), np.array(Y)

# 测试每轮的准确度,假设默认样本为500，可以修改
def evaluate(model, test_sample_num= 500):
    x, y = creat_datasanples(test_sample_num)
    correct, wrong = 0, 0
    hidden,y_pred = model.forward(x)
    for y_p, y_t in zip(y_pred, y):
        # softmax 不改变大小顺序，不用过
        if np.argmax(y_p) == y_t:
            correct += 1
        else:
            wrong += 1
    print("正确预测个数: %d, 正确率: %f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)
    
# 定义一个输入int格式的输入函数
def inputint(prompt):
    while True:
        n = input(prompt)
        try:
            int(n)
            return int(n)
        except (ValueError, TypeError, EOFError):
            continue

# 主流程
def main():
    # 参数设置
    input_size = 5              # 向量维度，要求是5
    class_num = 5               # 输出类别，要求是5
    hidden_size = 16            # 隐藏层神经元个数，输入的2～3倍，再加一点点，设置成16 设再大一点 感觉也差别不大
    train_samples = 5000        # 训练样本总数， 可以调整，看结果，第一次设500 准确率很低
    batchsize = 30              # 每批次训练个数，训练一次需要执行批次是 train_samples / batchsize
    epoch_num = inputint("请输入一个整数，代表训练的轮数")  # 可以多测试
    lrt = 0.05                  #，可以多次调整，看一下loss降速 0.01 有点慢了

    # 定义一个模型的实例
    model = DiyModel(input_size, hidden_size, class_num)
    # 根据样本总数的参数值来批量建样本
    train_x, train_y = creat_datasanples(train_samples)
    log = []

    for epoch in range(epoch_num):
        watch_loss = []
        # 整取一共要执行的批次数进行循环
        for batch_index in range(train_samples // batchsize):
            x = train_x[batch_index * batchsize : (batch_index + 1) * batchsize]
            y = train_y[batch_index * batchsize : (batch_index + 1) * batchsize]

            hidden, y_pred = model.forward(x)              # 向前传播
            loss = corss_entropy(y_pred, y)                # loss的值
            dw1, db1, dw2, db2 = model.backward(x, hidden, y_pred, y)           # 反向传播 计算梯度

            model.w1 = model.w1 - lrt * dw1
            model.b1 = model.b1 - lrt * db1
            model.w2 = model.w2 - lrt * dw2
            model.b2 = model.b2 - lrt * db2
            watch_loss.append(loss)

        print("=" * 50)
        print("第%d轮平均loss:%F" % (epoch +1, np.mean(watch_loss)))
        accuracy = evaluate(model)
        log.append([accuracy, float(np.mean(watch_loss))])

    # 画图
    plt.plot(range(len(log)), [l[0] for l in log], label= "accuracy")
    plt.plot(range(len(log)), [l[1] for l in log], label= "loss" )
    plt.legend()
    plt.show()
    return
    
if __name__ == "__main__":
    main()
