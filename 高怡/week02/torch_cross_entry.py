import torch
from torch import nn, optim
import numpy as np
from math import ceil
import matplotlib.pyplot as plt

class TorchModel(nn.Module):
    def __init__(self, hidden_sizes):
        super().__init__()
        self.levels = nn.ModuleList()
        for i in range(1,len(hidden_sizes)):
            self.levels.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
        self.loss = nn.functional.cross_entropy
    
    def forward(self, x, y=None):
        y_pred = x
        for level in self.levels:
            y_pred = level(y_pred)
        if y == None:
            return y_pred
        else:
            return self.loss(y_pred, y)

def build_data_set(data_set_size):
    x = []
    y = []
    for i in range(data_set_size):
        sample = np.random.rand(5)
        x.append(sample)
        y.append(np.argmax(sample))
    return torch.FloatTensor(x), torch.LongTensor(y)

def evaluate(module, x_validate, y_validate):
    module.eval()
    correct = 0
    wrong = 0
    with torch.no_grad():
        logits = module.forward(x_validate)
        y_validate_pred = torch.argmax(logits, dim=1)
        for y_pred, y in zip(y_validate_pred, y_validate):
            if y_pred == y:
                correct += 1
            else:
                wrong += 1
    return correct / (correct + wrong)

def test_module(module_path):
    module = TorchModel([5,5])
    module.load_state_dict(torch.load(module_path))
    print(f"模型权重：{module.state_dict()}\n")
    x_test, y_test = build_data_set(5)
    module.eval()
    correct = 0
    wrong = 0
    with torch.no_grad():
        logits = module.forward(x_test)
        probs = torch.softmax(logits, dim=1)
        y_pred_test = torch.argmax(logits, dim=1)
        for x, y_pred_logits, y_pred, y in zip(x_test, probs, y_pred_test, y_test):
            if y_pred == y:
                correct += 1
            else:
                wrong += 1
            print(f"测试数据：{x}, 预测结果(softmax)：{y_pred_logits}, 预测结果：{y_pred}, 实际结果：{y}")
    print(f"测试数据准确率：{correct / (correct + wrong)}")

def main():
    # 设置参数
    train_size = 200
    validate_size = int(train_size * 0.01)
    epoch_size = 200
    batch_size = 10
    hidden_sizes = [5,5]
    # 创建训练集和验证集
    x_train, y_train = build_data_set(train_size)
    x_validate, y_validate = build_data_set(validate_size)
    # 创建模型和优化器
    module = TorchModel(hidden_sizes)
    optimizer = optim.Adam(module.parameters(), lr=0.15)
    log = []
    # 模型训练
    for epoch in range(epoch_size):
        # 每轮训练分多个批次训练
        module.train()
        epoch_loss = None
        for index in range(ceil(epoch_size/batch_size)):
            # 切分每批次训练集
            x_batch = x_train[index * batch_size:(index + 1) * batch_size]
            y_batch = y_train[index * batch_size:(index + 1) * batch_size]
            # 每批次训练
            loss = module.forward(x_batch, y_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss = loss.item()
        # 每轮训练完毕使用验证集计算模型预测准确度
        accuracy = evaluate(module, x_validate, y_validate)
        print(f"第{epoch + 1}轮训练的损失：{epoch_loss}, 准确度：{accuracy}")
        log.append([accuracy, float(np.mean(epoch_loss))])
    torch.save(module.state_dict(), "model.bin")
    # 绘制损失函数图
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    
if __name__ == "__main__":
    main()
    test_module("model.bin")