# coding=utf-8
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 是否使用GPU
use_gpu = torch.cuda.is_available()

# 超参数
batch_size = 64
learning_rate = 1e-2
num_epoches = 20

# MNIST数据准备
data_tf = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize([0.5], [0.5])]
)
train_dataset = datasets.MNIST(root="./data", train=True, transform=data_tf, download=True)
test_dataset = datasets.MNIST(root="./data", train=False, transform=data_tf)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

'''
    nn.Conv2d():
        in_channels: 输入数据体的深度
        out_channels: 输出数据体的深度
        kernal_size: 卷积核的大小
        stride: 滑动的步长
        padding: 四周填充几圈像素点
    经过卷积或池化，图片的大小：
        width = (width - kernal_size + 2 * padding) / stride + 1
        height = (height - kernal_size + 2 * padding) / stride + 1
'''


# LeNet模型，1998年由LeCun提出
class LeNet(nn.Module):
    # shape：图片的形状，(宽, 高, 深度)，表示图片的宽、高，对于RGB图片，深度为3，灰度图片深度为1
    # num_classes：模型最终的分类数
    def __init__(self, shape, num_classes):
        super(LeNet, self).__init__()
        width = shape[0]
        height = shape[1]
        deep = shape[2]

        self.layer1 = nn.Sequential()
        self.layer1.add_module("conv1", nn.Conv2d(in_channels=deep, out_channels=6, kernel_size=3, stride=1))   # (width-2)*(height-2)*6
        self.layer1.add_module("pool1", nn.MaxPool2d(2, 2))
        # [(width-2)/2] * [(height-2)/2] * 6

        self.layer2 = nn.Sequential()
        self.layer2.add_module("conv2", nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1))  # ([(width-2)/2] - 4) * ([(height-2)/2] - 4)  * 16
        self.layer2.add_module("pool2", nn.MaxPool2d(2, 2))
        # [([(width-2)/2] - 4) / 2] * [([(height-2)/2] - 4) / 2] * 16

        self.layer3 = nn.Sequential()
        in_features = (int)(((width - 2) / 2 - 4) / 2) * (int)(((height - 2)/2 - 4) / 2) * 16
        self.layer3.add_module("fc1", nn.Linear(in_features=in_features, out_features=120))
        self.layer3.add_module("fc2", nn.Linear(in_features=120, out_features=84))
        self.layer3.add_module("fc3", nn.Linear(in_features=84, out_features=num_classes))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        x = self.layer3(x)
        return x


# 此处改变选择的模型
model = LeNet(shape=(28, 28, 1), num_classes=10)
if use_gpu:
    model = model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)


if __name__ == "__main__":
    # 模型训练
    for epoch in range(num_epoches):
        average_loss = 0
        total = 0
        success = 0

        model.train()
        for i, data in enumerate(train_loader, 0):
            train_inputs, train_labels = data
            if use_gpu:
                train_inputs = Variable(train_inputs).cuda()
                train_labes = Variable(train_labels).cuda()
            else:
                train_inputs = Variable(train_inputs)
                train_labes = Variable(train_labels)
            train_outputs = model(train_inputs)
            train_loss = criterion(train_outputs, train_labels)
            optimizer.zero_grad()
            train_loss.backward()

            optimizer.step()

            _, train_predicted = torch.max(train_outputs.data, 1)
            total += train_labels.size(0)
            success += (train_predicted == train_labels.data).sum()

        train_accuracy = 100.0 * success / total

        # 模型测试
        total = 0
        success = 0
        model.eval()
        for test_data in test_loader:
            test_inputs, test_labels = test_data
            if use_gpu:
                test_inputs = Variable(test_inputs).cuda()
                test_labels = Variable(test_labels).cuda()
            else:
                test_inputs = Variable(test_inputs)
                test_labels = Variable(test_labels)
            test_outputs = model(test_inputs)
            test_loss = criterion(test_outputs, test_labels)
            _, test_predicted = torch.max(test_outputs, 1)
            total += test_labels.size(0)
            success += (test_predicted == test_labels).sum()

        test_accuracy = 100.0 * success / total
        print(u"Epoch {}, train loss {}, test loss {}, train accuracy {}%, test accuracy {}%".format(
            epoch, train_loss.data, test_loss.data, train_accuracy,
            test_accuracy))

        # 每轮测试完成即保存模型
        torch.save(model, "model/04_CNN_LeNet.pth")
