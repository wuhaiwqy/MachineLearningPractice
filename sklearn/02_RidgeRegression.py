import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
import matplotlib.pyplot as plt

data = pd.read_csv('./data/CCPP.csv')
data_x = data[['AT', 'V', 'AP', 'RH']]
data_y = data[['PE']]

x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, random_state=1)


if __name__ == '__main__':
    # 岭回归
    model = Ridge(alpha=1)
    model.fit(x_train, y_train)
    print('theta0: ', model.intercept_)
    print('coefficients: ', model.coef_)

    # 超参数调优
    print()
    ridgecv = RidgeCV(alphas=[0.01, 0.1, 0.5, 1, 3, 5, 10, 30, 50, 100])
    ridgecv.fit(x_train, y_train)
    print(ridgecv.alpha_)

    # 超参数alpha与模型参数关系
    X = 1. / (np.arange(1, 11) + np.arange(0, 10)[:, np.newaxis])
    y = np.ones(10)
    alphas = np.logspace(-10, -2, 200)
    coefs = []
    for alpha in alphas:
        model = Ridge(alpha=alpha, fit_intercept=False)
        model.fit(X, y)
        coefs.append(model.coef_)
    ax = plt.gca()
    ax.plot(alphas, coefs)
    ax.set_xscale('log')
    ax.set_xlim(ax.get_xlim()[::-1])
    plt.xlabel('alpha')
    plt.ylabel('weight')
    plt.axis('tight')
    plt.show()
