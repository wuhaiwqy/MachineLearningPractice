from sklearn.datasets.samples_generator import make_blobs
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# 数据准备
X, y = make_blobs(n_samples=10000,
                  n_features=3,
                  centers=[[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]],
                  cluster_std=[0.2, 0.2, 0.2, 0.2],
                  random_state=1)
fig1 = plt.figure('降维前数据')
Axes3D(fig1, rect=[0, 0, 1, 1], elev=30, azim=20)
plt.scatter(X[:, 0], X[:, 1], X[:, 2], marker='o')

# 模型训练
pca = PCA(n_components='mle')
pca.fit(X)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_)

# 降维并画图
X_dec = pca.transform(X)
fig2 = plt.figure('降维数据')
if 3 == pca.n_components_:
    Axes3D(fig1, rect=[0, 0, 1, 1], elev=30, azim=20)
    plt.scatter(X_dec[:, 0], X_dec[:, 1], X_dec[:, 2], marker='o')
elif 2 == pca.n_components_:
    plt.scatter(X_dec[:, 0], X_dec[:, 1], marker='o')
elif 1 == pca.n_components_:
    plt.scatter(range(0, len(X_dec)), X_dec[:, 0], marker='o')
plt.show()

# 参考资料：
# https://www.cnblogs.com/pinard/p/6243025.html