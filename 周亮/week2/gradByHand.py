# 假设面试中要把深度学习的原理用代码实现

# 核心：循环执行：向前传播->计算损失值->反向传播，梯度下降->更新权重


from math import  log
import random

from matplotlib import pyplot as plt

random.seed(10)
def build_data(nums):
    X = []
    Y =[]
    for i in range(1,nums):
        x =  0.9 * i / nums
        y = 10 ** x + x ** 4 + log(2)/log(x) + 7     # 三种常见函数的导数
        X.append(x)
        Y.append(y)

    return X,Y

# 定义模型，就是一个带有多个参数的数学公式
def model(x):
    y= W1 ** x + x ** W2 + log(W3) / log(x) + B
    return y

# 定义损失函数，公式要能计算出x与y之间的有最小值
def loss(y_pred,y_true):
    return (y_pred - y_true) ** 2

# 随机初始化权重
W1 = random.random()
W2 = random.random()
W3 = random.random()
B = random.random()
print(f"初始权重：W1:{W1:.4f},W2:{W2:.4f},W3:{W3:.4f},B:{B:.4f}")

# 定义超参数
lr = 0.001 #学习率，影响权重调整的大小

# 训练
epoch = 40 # 训练轮数，为了让模型充分学习，loss更加收敛
data_nums = 1000 # 训练数据条数
batch_size = 20 #   批次大小, 累计batch_size条数据后，更新一次权重，应该是课上讲的SGD
batch_nums = data_nums // batch_size

# 创建训练集
X,Y = build_data(data_nums)

for i in range(epoch):
    epoch_loss_sum = 0
    grad_w1 = 0
    grad_w2 = 0
    grad_w3 = 0
    grad_b = 0

    # 在每轮训练之前打乱训练集
    train_list = list(zip(X,Y))
    random.shuffle(train_list)

    for j in range(batch_nums):
        batch_data = train_list[j*batch_size:(j+1)*batch_size]
        for x,y_true in batch_data:
            pred = model(x)   # 向前计算
            epoch_loss_sum += loss(pred,y_true) # 计算损失
            # 计算梯度，链式法则
            grad_w1 += 2*(pred - y_true) * x * W1 ** (x-1)
            grad_w2 += 2*(pred - y_true) * x**W2 * log(x)
            grad_w3 += 2*(pred - y_true) * 1 / (W3 * log(x))
            grad_b += 2*(pred - y_true)
        # 更新权重
        W1 = W1 - lr * (grad_w1/batch_size)
        W2 = W2 - lr * (grad_w2/batch_size)
        W3 = W3 - lr * (grad_w3/batch_size)
        B = B - lr * (grad_b/batch_size)
        # 梯度清零
        grad_w1 = 0
        grad_w2 = 0
        grad_w3 = 0
        grad_b = 0
    epoch_loss = epoch_loss_sum / len(train_list) # 计算这一轮的平均损失,MSE
    print(f"第{i}轮：损失率：{epoch_loss}")

    # 用损失率判断是否收敛，收敛就停止训练
    if epoch_loss < 0.0001:
        print("损失率收敛，停止训练")
        break

print(f"训练后权重：W1:{W1:.4f},W2:{W2:.4f},W3:{W3:.4f},B:{B:.4f}")

# 把预测值和真实值分别画出来.
pred_Y = [model(x) for x in X]
plt.plot(X,Y,label="true_y")
plt.plot(X,pred_Y,label="pred_y")
plt.legend()
plt.show()


