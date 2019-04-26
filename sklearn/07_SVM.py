import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.datasets import make_moons
from sklearn.datasets import make_blobs
from sklearn.datasets import make_circles
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


# demo01：使用鸢尾花数据集进行分类，得到测试集上的正确率
# 鸢尾花数据集最后一维数据转换
def iris_type_converter(name):
    name = name.decode()
    types = {'setosa': 0, 'versicolor': 1, 'virginica': 2}
    return types[name]


# 计算模型正确率
def get_model_accuracy_rate(model, data_x, data_y):
    pred = model.predict(data_x)
    # 正确结果数量
    right_count = sum(pred == data_y)
    return right_count / data_x.shape[0]


def demo01():
    # 数据准备
    data = np.loadtxt(fname='data/iris.txt', dtype=float, delimiter=' ', converters={5: iris_type_converter})
    data_x, data_label = np.split(ary=data[:, 1:], indices_or_sections=(4,), axis=1)
    x_train, x_test, y_train, y_test = train_test_split(data_x, data_label.T[0], random_state=1, train_size=0.6,
                                                        test_size=0.4)
    # C：惩罚系数
    # kernel：核函数，linear：线性核函数, poly：多项式核函数, rbf：高斯核函数, sigmoid：sigmoid核函数
    # degree：若kernel使用多项式核函数poly，该参数对应K(x,z)=（γx∙z+r)^d中的d
    # gamma：若kernel使用多项式核函数poly、高斯核函数rbf或sigmoid核函数，需调整该参数，默认为auto_deprecated，即1/特征维度
    # coef0：若kernel使用了多项式核函数poly或sigmoid核函数，需调整此参数
    # class_weight：指定样本各类别的的权重，以防止训练集某些类别的样本过多，导致训练的决策过于偏向这些类别。这里可以自己指定各个样本的权重，或者用“balanced”，如果使用“balanced”，则算法会自己计算权重，样本量少的类别所对应的样本权重会高。如果样本类别分布没有明显的偏倚，则可以不管这个参数，选择默认的"None"
    # decision_function_shape：进行多分类时使用，ovr：one-vs-rest，选择其中一类为正样本，其余为负样本，ovo：one-vs-one：每两类做分类
    # cache_size：缓存大小在大样本的时候，缓存大小会影响训练速度，因此如果机器内存大，推荐用500MB甚至1000MB。默认是200，即200MB
    model = SVC(C=0,
                kernel='linear',
                degree=3,
                gamma='auto_deprecated',
                coef0=0,
                class_weight=None,
                decision_function_shape='ovr',
                cache_size=500)
    model.fit(x_train, y_train)
    accuracy_rate = get_model_accuracy_rate(model, x_test, y_test)
    print('正确率：', accuracy_rate)


# demo02：自己生成二维数据，并将分类结果作图展示
# 描绘数据集
def draw_data(data_x, data_label, ax=None):
    if ax is None:
        ax = plt.gca()
    ax.plot(data_x[:, 0][data_label == 0], data_x[:, 1][data_label == 0], "bo")
    ax.plot(data_x[:, 0][data_label == 1], data_x[:, 1][data_label == 1], "ro")


# 描绘分界面
def draw_model(model, ax=None):
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    x_, y_ = np.meshgrid(x, y)
    grid_data_x = np.c_[x_.ravel(), y_.ravel()]
    grid_pred = model.predict(grid_data_x).reshape(x_.shape)
    ax.contour(x_, y_, grid_pred, cmap=ListedColormap('black'), alpha=0.8)
    grid_decision = model.decision_function(grid_data_x).reshape(x_.shape)
    ax.contour(x_, y_, grid_decision, cmap=plt.cm.winter, alpha=0.2)


# 根据给定的数据，训练模型，给出正确率并画图
# 入参中的model需要外部指定参数并传入
# ax：plot的子图
def train_and_draw_svc(data_x, data_label, model, ax):
    draw_data(data_x, data_label)
    x_train, x_test, y_train, y_test = train_test_split(data_x, data_label, random_state=1, train_size=0.6,
                                                        test_size=0.4)
    model.fit(x_train, y_train)
    draw_model(model)


def demo02():
    fig = plt.figure()
    # 数据集1
    data_x1, data_label1 = make_moons(n_samples=100, noise=0.15, random_state=1)
    model1 = SVC(C=5, kernel='rbf', gamma=0.5)
    ax1 = fig.add_subplot(1, 3, 1)
    train_and_draw_svc(data_x1, data_label1, model1, ax1)
    # 数据集2
    data_x2, data_label2 = make_blobs(n_samples=100, n_features=2, centers=2, cluster_std=5.0, center_box=(-10.0, 10.0),
                                      shuffle=True, random_state=1)
    ax2 = fig.add_subplot(1, 3, 2)
    model2 = SVC(C=1.0, kernel='linear')
    model2.fit(data_x2, data_label2)
    train_and_draw_svc(data_x2, data_label2, model2, ax2)
    # 数据集3
    data_x3, data_label3 = make_circles(n_samples=100, shuffle=True, noise=0.1, random_state=1, factor=0.6)
    model3 = SVC(C=1.0, kernel='rbf', gamma=0.5)
    ax3 = fig.add_subplot(1, 3, 3)
    train_and_draw_svc(data_x3, data_label3, model3, ax3)

    plt.show()


# demo03：SVM用于回归



def train_and_draw_svr(data_x, data_label, model, ax):
    if ax is None:
        ax = plt.gca()
    # 画出数据
    ax.plot(data_x, data_label, 'bo')
    # 训练
    model.fit(data_x, data_label)

    xlim = ax.get_xlim()
    pred_data_x = np.linspace(xlim[0], xlim[1], 30).reshape(-1, 1)
    pred_data_label = model.predict(pred_data_x)
    ax.plot(pred_data_x, pred_data_label, "r-", linewidth=2, label=r"$\hat{y}$")
    ax.plot(pred_data_x, pred_data_label - model.epsilon, "k--")
    ax.plot(pred_data_x, pred_data_label + model.epsilon, "k--")


def demo03():
    np.random.seed(42)
    n_samples = 100
    data_x = 2 * np.random.rand(n_samples, 1)
    data_label = (4 + 3 * data_x + np.random.randn(n_samples, 1)).ravel()

    model = SVR(C=100, kernel='poly', degree=2, gamma=0.5, epsilon=1)
    train_and_draw_svr(data_x, data_label, model, plt.gca())
    plt.show()


if __name__ == '__main__':
    demo01()
    #demo02()
    #demo03()


# 参考文献：
# https://www.cnblogs.com/pinard/p/6117515.html
# https://blog.csdn.net/woaixuexihhh/article/details/84702233
# https://www.jianshu.com/p/84015743be01
# https://blog.csdn.net/u012526003/article/details/79088214