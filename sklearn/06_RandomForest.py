import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

data = pd.read_csv('data/train_modified.csv')
# 查看Disbursed类别的信息
#print(data['Disbursed'].value_counts())
x_columns = [x for x in data.columns if x not in ['Disbursed', 'ID']]
data_x = data[x_columns]
data_label = data['Disbursed']


if __name__ == '__main__':
    # 默认参数
    # oob_score：是否采用袋外样本评估模型好坏
    model_defult = RandomForestClassifier(oob_score=True, random_state=10)
    model_defult.fit(data_x, data_label)
    print(model_defult.oob_score_)  # 0.98005
    # AOC分数
    y_predprob = model_defult.predict_proba(data_x)[:,1]
    print('AOC score:', metrics.roc_auc_score(data_label, y_predprob))

    # 网格搜索调参
    # 第一次调参
    # n_estimators：最大的弱学习器的个数
    opt_params1 = {'n_estimators': range(40, 80)}
    gsearch1 = GridSearchCV(estimator=RandomForestClassifier(min_samples_split=100,
                                                             min_samples_leaf=20,
                                                             max_depth=8,
                                                             max_features='sqrt',
                                                             random_state=10),
                            param_grid=opt_params1, scoring='roc_auc', cv=5)
    gsearch1.fit(data_x, data_label)
    print(gsearch1.best_params_, gsearch1.best_score_)  # {'n_estimators': 61} 0.8214438833841464

    # 第一次调参后模型效果评估
    model_opt1 = RandomForestClassifier(n_estimators=61,
                                        min_samples_split=100,
                                        min_samples_leaf=20,
                                        max_depth=8,
                                        max_features='sqrt',
                                        oob_score=True,
                                        random_state=10)
    model_opt1.fit(data_x, data_label)
    print(model_opt1.oob_score_)  # 0.984
    
    # 第二次调参
    # max_depth：树最大深度
    # min_samples_split：叶子节点划分所需最小样本数
    opt_params2 = {'max_depth': range(3, 14), 'min_samples_split': range(50, 201, 20)}
    gsearch2 = GridSearchCV(estimator=RandomForestClassifier(n_estimators=61,
                                                             min_samples_leaf=20,
                                                             max_features='sqrt',
                                                             random_state=10),
                            param_grid=opt_params2, scoring='roc_auc', cv=5)
    gsearch2.fit(data_x, data_label)
    print(gsearch2.best_params_, gsearch2.best_score_)  # {'max_depth': 13, 'min_samples_split': 90} 0.8248928163109757

    # 第二次调参后模型效果评估
    model_opt2 = RandomForestClassifier(n_estimators=61,
                                        max_depth=13,
                                        min_samples_split=90,
                                        min_samples_leaf=20,
                                        max_features='sqrt',
                                        oob_score=True,
                                        random_state=10)
    model_opt2.fit(data_x, data_label)
    print(model_opt2.oob_score_)  # 0.984
    
    # 第三次调参
    # min_samples_split：节点在划分所需最小样本数
    # min_samples_leaf：叶子节点最小样本数
    opt_params3 = {'min_samples_split':range(80,150,10), 'min_samples_leaf':range(10,60,10)}
    gsearch3 = GridSearchCV(estimator=RandomForestClassifier(n_estimators=61,
                                                             max_depth=13,
                                                             max_features='sqrt',
                                                             random_state=10),
                            param_grid=opt_params3, scoring='roc_auc', cv=5)
    gsearch3.fit(data_x, data_label)
    print(gsearch3.best_params_, gsearch3.best_score_)  # {'min_samples_leaf': 20, 'min_samples_split': 120} 0.8250786013719513
    
    # 第三次调参后模型效果评估
    model_opt3 = RandomForestClassifier(n_estimators=61,
                                        max_depth=13,
                                        min_samples_split=120,
                                        min_samples_leaf=20,
                                        max_features='sqrt',
                                        oob_score=True,
                                        random_state=10)
    model_opt3.fit(data_x, data_label)
    print(model_opt3.oob_score_)  # 0.984

    # 第四次调参
    # max_features：最大特征数
    opt_params4 = {'max_features':range(3, 11)}
    gsearch4 = GridSearchCV(estimator=RandomForestClassifier(n_estimators=61,
                                                             max_depth=13,
                                                             min_samples_split=120,
                                                             min_samples_leaf=20,
                                                             random_state=10),
                            param_grid=opt_params4, scoring='roc_auc', cv=5)
    gsearch4.fit(data_x, data_label)
    print(gsearch4.best_params_, gsearch4.best_score_)  # {'max_features': 7} 0.8250786013719513

    # 第四次调参后模型效果评估
    opt_model4 = RandomForestClassifier(n_estimators=61,
                                        max_depth=13,
                                        min_samples_split=120,
                                        min_samples_leaf=20,
                                        max_features=7,
                                        oob_score=True,
                                        random_state=10)
    opt_model4.fit(data_x, data_label)
    print(opt_model4.oob_score_)    # 0.984

# 参考文献：https://www.cnblogs.com/pinard/p/6160412.html
