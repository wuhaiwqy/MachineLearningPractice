import numpy as np

'''
    逻辑回归模型类
'''
class LogisticRegression:
    def __init__(self, theta = None):
        self.theta = theta

    '''
        功能：模型训练
        输入：
            data_x：二维数组，每一行是一条训练数据
            data_label：训练数据对应的值
            learning_rate：学习率
    '''
    def train(self, data_x, data_label, learning_rate = 0.001):
        if data_x.shape[0] <= 0:
            raise Exception('缺少训练数据！')
        if data_x.shape[0] != data_label.shape[0]:
            raise Exception('训练数据与标签不一致')
        # 训练数据添加x0=1项
        ones = np.ones(shape=(data_x.shape[0], 1))
        x = np.hstack((ones, data_x))
        # 初始化参数
        self.theta = np.ones(shape=(x.shape[1]))
        # 迭代，批梯度下降
        new_theta = np.zeros(shape=(x.shape[1]))
        while(self.__distance(self.theta, new_theta) > 0.0001):
            self.theta = new_theta.copy()
            for j in range(self.theta.shape[0]):
                new_theta[j] = self.theta[j] - learning_rate * np.sum(np.dot(self.predict(data_x) - data_label, x[:,j]))

    '''
        功能：预测
        输入：
            data_x：二维数组，每一行是一条新样本
        输出：
            预测结果：一维数组，对应data_x的每一条样本的预测值
    '''
    def predict(self, data_x):
        ones = np.ones(shape=(data_x.shape[0], 1))
        x = np.hstack((ones, data_x))
        result = []
        for item in x:
            result.append(self.__predict(item))
        return np.array(result)

    def __predict(self, data_item):
        if np.dot(data_item, self.theta) < 0:
            return 0
        else:
            return 1

    '''
        功能：计算两点之间的几何距离
        输入：
            a、b：两个向量
    '''
    def __distance(self, a, b):
        return np.sqrt(np.sum((b - a) * (b - a)))


if __name__ == "__main__":
    data_x = np.array([[-1, -2], [-3, -2], [-1, -3], [3, 2], [10, 11], [4, 1], [3, 3]])
    data_label = np.array([0, 0, 0, 1, 1, 1, 1])
    model = LogisticRegression()
    model.train(data_x, data_label)
    print(model.theta)
    print(model.predict(data_x))
