# coding=utf-8

from __future__ import print_function
import torch
from torch.autograd import Variable
import numpy as np

print('判断计算机是否支持GPU加速')
print(torch.cuda.is_available())
print()

print('张量的使用')
a = torch.zeros((3, 2))  # 全0
a = torch.randn((3, 2))  # 正态分布随机初始值
a = torch.Tensor([[1, 2], [3, 4], [5, 6]])
print(a)
print(a.size())
print(a[1, 0])  # 索引
print()

print('numpy数组和张量之间的转换')
npa = a.numpy()
print(npa)
npa = np.array([[0, 1], [10, 11], [20, 21]])
a = torch.from_numpy(npa)
print(a)
print()

print('变量的使用')
# Variable的成员变量
# data：变量中的张量数值
# grad：反向传播的梯度
# grad_fn：得到该变量的操作
x = Variable(torch.Tensor([5]), requires_grad=True)
w = Variable(torch.Tensor([2]), requires_grad=True)
b = Variable(torch.Tensor([3]), requires_grad=True)
y = w * x + b
y.backward()  # 自动求导
print('x.data=', x.data)
print('x.grad=', x.grad)
print('w.data=', w.data)
print('w.grad=', w.grad)
print('b.data=', b.data)
print('b.grad=', b.grad)
print('y.data=', y.data)
print('y.grad=', y.grad)
print()

print('矩阵的求导')
x = torch.randn(3)
x = Variable(x, requires_grad=True)
print(x)
print(x.grad)  # None
y = 2 * x
print(y)
y.backward(torch.Tensor([1, 0.1, 0.01]))
print(x.grad)  # [2, 0.2, 0.02]

# 参考资料：
# 《深度学习之PyTorch》（廖星宇 编著）
