# coding=utf-8
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
import matplotlib.pyplot as plt


# 数据准备
x_train = np.array([[3.3 ], [4.4 ], [5.5], [6.71], [6.93], [4.168],
                    [9.779], [6.182], [7.59], [2.167], [7.042],
                    [10.791], [5.313] , [7.997] , [3.1]], dtype=np.float32)
y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694 ], [1.573],
                     [3.366], [2.596], [2.53], [1.221], [2.827],
                     [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)
x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train)


class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        out = self.linear(x)
        return out


use_gpu = torch.cuda.is_available()
if use_gpu:
    model = LinearRegression().cuda()
else:
    model = LinearRegression()

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=1e-3)


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
        # 每隔一定轮次打印结果
        if epoch % 20 == 0:
            print('Epoch[{}/{}]: {:.6f}'.format(epoch+1, epoch_num, loss.data))

    # 模型保存
    torch.save(model, 'model/02_LinearRegression.pth')

    # 模型加载
    load_model = torch.load('model/02_LinearRegression.pth')

    # 模型预测
    load_model.eval()
    predict = load_model(Variable(x_train))
    ax = plt.gca()
    ax.plot(x_train.numpy(), y_train.numpy(), 'ro', label='样本')
    ax.plot(x_train.numpy(), predict.data.numpy(), label='拟合曲线')
    plt.show()

# 参考资料：
# 《深度学习之PyTorch》（廖星宇 编著）