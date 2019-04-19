import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import matplotlib.pyplot as plt

data = pd.read_csv('./data/CCPP.csv')
data_y = data[['PE']]
model = LinearRegression()

if __name__ == '__main__':
    # 只使用3个特征
    print("Features: AT V AP")
    data_x = data[['AT', 'V', 'AP']]
    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, random_state=1)
    model.fit(x_train, y_train)
    print('theta0: ', model.intercept_)
    print('coefficients: ', model.coef_)
    y_pred = model.predict(x_test)
    mse = metrics.mean_squared_error(y_pred, y_test)
    print('MSE: ', mse)

    # 使用4个特征
    print()
    print("Features: AT V AP RH")
    data_x = data[['AT', 'V', 'AP', 'RH']]
    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, random_state=1)
    model.fit(x_train, y_train)
    print('theta0: ', model.intercept_)
    print('coefficients: ', model.coef_)
    y_pred = model.predict(x_test)
    mse = metrics.mean_squared_error(y_pred, y_test)
    print('MSE: ', mse)

    # 交叉验证
    print()
    print('Cross Validation')
    y_pred_cross = cross_val_predict(model, data_x, data_y, cv=10)
    mse = metrics.mean_squared_error(y_pred_cross, data_y)
    print('MSE: ', mse)

    # 画图，越接近y=x说明预测越准确
    fig, ax = plt.subplots()
    ax.scatter(data_y, y_pred_cross)
    ax.plot([data_y.min(), data_y.max()], [data_y.min(), data_y.max()], 'k--', lw=4)
    ax.set_xlabel('实际值')
    ax.set_ylabel('预测值')
    plt.show()