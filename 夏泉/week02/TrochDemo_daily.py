import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

"""

基于pytorch框架编写模型训练
实现一个自行构造的找规律分类任务
规律：x是一个5维向量，如果第n个数最大，则为第n个类型的样本

"""


class TorchModel(nn.Module):
    def __init__(self):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(5, 5)  # 线性层
        self.loss = nn.CrossEntropyLoss()  # 多分类交叉熵

    def forward(self, x, y=None):
        logits = self.linear(x)
        if y is not None:
            return self.loss(logits, y)
        else:
            return torch.argmax(logits, dim=-1)


class DataGenerator:
    def __init__(self):
        pass

    def build_sample(self):
        x = np.random.random(5)
        y = int(np.argmax(x))
        return x, y

    def build_dataset(self, total_sample_num):
        X = []
        Y = []
        for i in range(total_sample_num):
            x, y = self.build_sample()
            X.append(x)
            Y.append(y)  # 注意：多分类标签是 int，不用包成 [y]
        # return torch.FloatTensor(X), torch.IntTensor(Y)
        return torch.FloatTensor(X), torch.LongTensor(
            Y
        )  # 多分类标签使用 LongTensor（int64）


def evaluate(model):
    model.eval()
    x, y = DataGenerator().build_dataset(100)
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测
        for p, t in zip(y_pred, y):
            if int(p) == int(t):
                correct += 1
            else:
                wrong += 1
    acc = correct / (correct + wrong)
    print("正确预测个数：%d, 正确率：%f" % (correct, acc))
    return acc


def main():
    epoch_num = 20  # 训练轮数
    batch_size = 20  # 每批训练样本数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    learning_rate = 0.01  # 学习率
    # 建立模型
    model = TorchModel()
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 开始训练
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        x, y = DataGenerator().build_dataset(train_sample)
        for i in range(0, train_sample, batch_size):
            x_batch = x[i : i + batch_size]
            y_batch = y[i : i + batch_size]
            loss = model(x_batch, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            watch_loss.append(loss.item())
        print("epoch %d, loss: %f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    torch.save(model.state_dict(), "model2.bin")
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return


# 使用模型做预测
def predict(model_path, test_vec):
    model = TorchModel()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    x = torch.FloatTensor(test_vec)
    with torch.no_grad():
        y_pred_result = model.forward(x)
        for vec, y_p in zip(test_vec, y_pred_result):
            print("输入向量：", vec, "预测结果：", int(y_p))


if __name__ == "__main__":
    # main()

    predict(
        "model2.bin",
        [
            [0.1, 0.2, 0.3, 0.4, 0.5],
            [0.9, 0.1, 0.2, 0.3, 0.4],
            [0.1, 0.9, 0.2, 0.3, 0.4],
        ],
    )
