import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge

data = pd.read_csv('./data/CCPP.csv')
data_x = data[['AT', 'V', 'AP', 'RH']]
data_y = data[['PE']]

x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, random_state=1)


if __name__ == '__main__':
    model = Ridge(alpha=1, fit_intercept=False)
    model.fit(x_train, y_train)
    print('theta0: ', model.intercept_)
    print('coefficients: ', model.coef_)