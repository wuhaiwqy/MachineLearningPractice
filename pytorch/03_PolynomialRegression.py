import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

# 真实函数
w_target = torch.FloatTensor([0.5, 3, 2.4]).unsqueeze(1)
b_target = torch.FloatTensor([0.19])

def f(x):
    x = x.unsqueeze(1)
    x = torch.cat([x ** i for i in range(1, 4)], dim=1)
    return x.mm(w_target) + b_target


# 训练数据
x_train = torch.randn(100)
y_train = f(x_train)


# 多项式回归类
class PolyRegression(nn.Module):
    # n：多项式次数
    def __init__(self, n):
        super(PolyRegression, self).__init__()
        self.n = n
        self.linear = nn.Linear(n + 1, 1)

    def forward(self, x):
        x_ = self.adjust_features(x)
        return self.linear(x_)

    def adjust_features(self, x):
        x = x.unsqueeze(1)
        return torch.cat([x ** i for i in range(0, self.n + 1)], dim=1)


use_gpu = torch.cuda.is_available()
n = 3
if use_gpu:
    model = PolyRegression(n).cuda()
else:
    model = PolyRegression(n)

criterion = nn.MSELoss()
optimizer = optim.SGD(params=model.parameters(), lr=1e-3)


if __name__ == '__main__':
    # 训练模型
    epoch_num = 1000
    for epoch in range(epoch_num):
        if use_gpu:
            inputs = Variable(x_train).cuda()
            targets = Variable(y_train).cuda()
        else:
            inputs = Variable(x_train)
            targets = Variable(y_train)
        # 前向传播
        out = model(inputs)
        loss = criterion(out, targets)
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch + 1 % 20 == 0:
            print('Epoch[{}/{}]: {:.6f}'.format(epoch + 1, epoch_num, loss.data))

    # 模型预测
    model.eval()
    x_predict = torch.from_numpy(np.sort(x_train.numpy()))
    y_predict = model(Variable(x_predict))

    # 画图
    ax = plt.gca()
    ax.plot(x_train.numpy(), y_train.numpy(), 'bo')
    ax.plot(x_predict.numpy(), y_predict.data.numpy(), 'r')
    plt.show()


# 参考资料：
# 《深度学习之PyTorch》（廖星宇 编著）
