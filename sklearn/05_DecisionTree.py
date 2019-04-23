from sklearn.datasets import load_iris
from sklearn import tree
import pydotplus


iris = load_iris()
data_x = iris.data
data_label = iris.target


if __name__ == '__main__':
    model = tree.DecisionTreeClassifier(
        criterion='gini',     # "gini"（基尼系数）或"entropy"（信息增益）
        splitter='best',      # "best"或"random"，前者在特征的所有划分点中找出最优的划分点，后者是随机的在部分划分点中找局部最优的划分点
        max_features=None,    # 划分时考虑的最大特征数，默认是"None",意味着划分时考虑所有的特征数；如果是"log2"意味着划分时最多考虑log2N个特征；如果是"sqrt"或者"auto"意味着划分时最多考虑√N个特征。如果是整数，代表考虑的特征绝对数。如果是浮点数，代表考虑特征百分比，即考虑（百分比xN）取整后的特征数。其中N为样本总特征数
        max_depth=None,       # 决策树的最大深度
        min_samples_split=2,  # 这个值限制了子树继续划分的条件，如果某节点的样本数少于min_samples_split，则不会继续再尝试选择最优特征来进行划分
        min_samples_leaf=1,   # 叶子节点最少样本数，如果某叶子节点数目小于样本数，则会和兄弟节点一起被剪枝
        min_weight_fraction_leaf=0,     # 叶子节点最小的样本权重和，这个值限制了叶子节点所有样本权重和的最小值，如果小于这个值，则会和兄弟节点一起被剪枝。默认是0，就是不考虑权重问题。一般来说，如果有较多样本有缺失值，或者分类树样本的分布类别偏差很大，就会引入样本权重，这时就要注意这个值了
        max_leaf_nodes=None,            # 最大叶子节点数，通过限制最大叶子节点数，可以防止过拟合，默认是"None”，即不限制最大的叶子节点数
        class_weight=None,              # 指定样本各类别的的权重，主要是为了防止训练集某些类别的样本过多，导致训练的决策树过于偏向这些类别
        min_impurity_split=None,        # 节点划分最小不纯度，这个值限制了决策树的增长，如果某节点的不纯度(基尼系数，信息增益，均方差，绝对差)小于这个阈值，则该节点不再生成子节点
        presort=False         # 数据是否预排序
    )
    model.fit(data_x, data_label)
    dot_data = tree.export_graphviz(model, out_file=None)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_pdf("iris.pdf")

# 参考文献：https://www.cnblogs.com/pinard/p/6056319.html
