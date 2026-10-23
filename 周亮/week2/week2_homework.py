# 多分类任务作业：随机生成5个数，分类值是数值最大的数的索引

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 构造测试数据集
def build_dataset(total_sample_num):
    x = np.random.rand(total_sample_num, 5)  # 随机生成5个数
    y = np.argmax(x, axis=1)  # 分类值是数值最大的数的索引
    return torch.FloatTensor(x), torch.LongTensor(y)

# 定义模型
class MultiModel(nn.Module):
    def __init__(self, hidden_size, lr) -> None:
        super().__init__()
        self.linear1 = nn.Linear(5, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.classifier = nn.Linear(hidden_size, 5)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x, y=None):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        x = torch.relu(x)
        y_pred = self.classifier(x)
        if y is not None:
            return self.criterion(y_pred, y)  # 预测值和真实值计算损失
        else:
            return y_pred  # 输出预测结果

def train():
    # 定义超参数
    epoch_num = 50
    batch_size = 50
    hidden_size = 32
    lr = 0.01
    data_num = 2000
    batch_num = data_num // batch_size
    train_x, train_y = build_dataset(data_num)
    model = MultiModel(hidden_size, lr)
    log = []
    for i in range(epoch_num):
        watch_loss = []
        for j in range(batch_num):
            x = train_x[j*batch_size:(j+1)*batch_size]
            y = train_y[j*batch_size:(j+1)*batch_size]
            loss = model(x, y)
            model.optimizer.zero_grad() # 梯度清零
            loss.backward()             # 反向传播计算梯度
            model.optimizer.step()      # 更新权重
            watch_loss.append(loss.item())  # 记录每个batch的loss
        # 输出loss和准确率
        print(f"epoch: {i}, loss: {np.mean(watch_loss)}, acc: {evaluate(model)}")
        log.append((np.mean(watch_loss), evaluate(model)))
    # 保存模型
    torch.save(model.state_dict(), "multi_model.pth")
    # 画出loss和acc曲线
    plt.plot(range(epoch_num), [x[0] for x in log], label="loss")
    plt.plot(range(epoch_num), [x[1] for x in log], label="acc")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.show()

def evaluate(model):
    model.eval()
    test_x, test_y = build_dataset(100)
    with torch.no_grad():
        y_pred = model(test_x)
        y_pred = torch.argmax(y_pred, dim=1)
        acc = (y_pred == test_y).sum().item() / len(test_y)
    return acc

if __name__ == "__main__":
    train()
    # 训练完成后，加载模型进行预测
    model = MultiModel(hidden_size=32, lr=0.01)
    model.load_state_dict(torch.load("multi_model.pth"))
    model.eval()
    test_x, test_y = build_dataset(10)
    with torch.no_grad():
        y_pred = model(test_x)
        y_pred = torch.argmax(y_pred, dim=1)
        print("预测结果:", y_pred.numpy())
        print("真实结果:", test_y.numpy())
