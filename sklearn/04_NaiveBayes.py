import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB

data_x = np.array([[1, 9], [3, 5], [2, 4], [3, 3], [1, 5], [2, 8], [4, 1]])
data_label = np.array([-1, 1, 1, 1, 1, -1, -1])
data_test = np.array([[1, 4]])


# 高斯朴素贝叶斯：先验分布是正态分布
def GaussianNBDemo():
    clf = GaussianNB()
    clf.fit(data_x, data_label)
    print('==== without prior probabilities ====')
    print('predict: ', clf.predict(data_test))
    print('probabilities: ', clf.predict_proba(data_test))
    print('log probabilities: ', clf.predict_log_proba(data_test))

    # 加入先验概率
    print('\n==== with prior probabilities ====')
    clf = GaussianNB(priors=(0.1, 0.9))
    clf.fit(data_x, data_label)
    print('predict: ', clf.predict(data_test))
    print('probabilities: ', clf.predict_proba(data_test))
    print('log probabilities: ', clf.predict_log_proba(data_test))

    # 部分拟合
    print('\n==== partial fit ====')
    clf = GaussianNB(priors=(0.1, 0.9))
    # 此处逐一拟合训练数据
    for i in range(data_label.size):
        clf.partial_fit(X=data_x[i:i+1], y=data_label[i:i+1], classes=(-1, 1))
    print('predict: ', clf.predict(data_test))
    print('probabilities: ', clf.predict_proba(data_test))
    print('log probabilities: ', clf.predict_log_proba(data_test))


# 多项式朴素贝叶斯：先验分布是多项式分布
def MultiNBDemo():
    #
    #    fit_prior   class_prior     prior probabilities
    #    False           -           P(Y=Ck) = 1 / K
    #    True            None        P(Y=Ck) = mk / m
    #    True         has values     class_prior
    print('\n==== Multinomial Naive Bayes ====')
    clf = MultinomialNB(fit_prior=True, class_prior=(0.2, 0.8))
    clf.fit(data_x, data_label)
    print('predict: ', clf.predict(data_test))
    print('probabilities: ', clf.predict_proba(data_test))
    print('log probabilities: ', clf.predict_log_proba(data_test))


if __name__ == '__main__':
    GaussianNBDemo()
    MultiNBDemo()
