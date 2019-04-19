import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import SpectralClustering

'''
    Data Generation
'''
# generate ring data
# ring_data_count: data amount of ONE ring
# radius_ranges: radius ranges of each ring, like ((min1, max1), (min2, max2))
# show_plot: show sample image or not
def genetrate_ring(ring_data_count, radius_ranges, show_plot=False):
    x = []
    y = []
    for radius_range in radius_ranges:
        radius_min = radius_range[0]
        radius_max = radius_range[1]
        if radius_min > radius_max:
            raise Exception("Wrong radius ranges data!")
        theta = np.random.random(size=ring_data_count) * 2 * np.pi # 角度
        radius = radius_min + (radius_max - radius_min) * np.random.random(size=ring_data_count) # 半径
        x.extend(radius * np.cos(theta))
        y.extend(radius * np.sin(theta))
    if show_plot:
        plt.plot(x, y, 'o')
        plt.show()
    return x, y

# generate X shape data
# data_count: total data amount
# xrange: range of x axis, (x_min, x_max)
# yrange: range of y axis, (y_min, y_max)
# show_plot: show sample image or not
def generate_x(data_count, xrange, yrange, show_plot=False):
    x = []
    y = []
    theta = np.arctan((yrange[1] - yrange[0]) / (xrange[1] - xrange[0]))
    data_count1 = int(data_count/2)
    x1 = xrange[0] + (xrange[1] - xrange[0]) * np.random.random(size=data_count1)
    y1 = np.tan(theta) * (x1 - xrange[0]) + yrange[0]
    x2 = xrange[0] + (xrange[1] - xrange[0]) * np.random.random(size=data_count - data_count1)
    y2 = np.tan(np.pi - theta) * (x2 - xrange[0]) + yrange[1]
    x.extend(x1)
    x.extend(x2)
    y.extend(y1)
    y.extend(y2)
    if show_plot:
        plt.plot(x, y, 'x')
        plt.show()
    return x, y


def generate_circle(circle_data_count, centers, radii, show_plot=False):
    x = []
    y = []
    for i in range(len(centers)):
        center = centers[i]
        radius = radii[i]
        theta_list = np.random.random(size=circle_data_count) * 2 * np.pi
        radius_list = radius * np.random.random(size=circle_data_count)
        x.extend(center[0] + radius_list * np.cos(theta_list))
        y.extend(center[1] + radius_list * np.sin(theta_list))
    if show_plot:
        plt.plot(x, y, 'x')
        plt.show()
    return x, y


'''
    tests
'''
def kmeans_test01_ring(k):
    x, y = genetrate_ring(1000, ((0.9, 1.1), (2.3, 2.4)), True)
    samples = np.column_stack((x, y))
    model = KMeans(n_clusters=k)
    model.fit(samples)
    labels = model.labels_

    plt.scatter(x, y, c=labels, marker='x')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("After Cluster: KMeans")
    plt.show()


def kmeans_test02_x(k):
    x, y = generate_x(1000, (2, 5), (6, 7), True)
    samples = np.column_stack((x, y))
    model = KMeans(n_clusters=k)
    model.fit(samples)
    labels = model.labels_

    plt.scatter(x, y, c=labels, marker='x')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("After Cluster: KMeans")
    plt.show()


def kmeans_test03_circle(k):
    x, y = generate_circle(circle_data_count=1000,
                    centers=((1,1), (1,2), (2, 1), (2,2)),
                    radii=(0.9, 0.9, 0.9, 0.9),
                    show_plot=True)
    samples = np.column_stack((x, y))
    model = KMeans(n_clusters=k)
    model.fit(samples)
    labels = model.labels_

    plt.scatter(x, y, c=labels, marker='x')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("After Cluster: KMeans")
    plt.show()


def dbscan_test01_ring(eps, min_samples):
    x, y = genetrate_ring(1000, ((0.9, 1.1), (2.3, 2.4)), True)
    samples = np.column_stack((x, y))
    model = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean',
                   metric_params=None, algorithm='auto', leaf_size=30,
                   p=None, n_jobs=1)
    model.fit(samples)
    labels = model.labels_

    plt.scatter(x, y, c=labels, marker='x')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("After Cluster: DBSCAN")
    plt.show()


def dbscan_test02_x(eps, min_samples):
    x, y = generate_x(1000, (2, 5), (6, 7), True)
    samples = np.column_stack((x, y))
    model = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean',
                   metric_params=None, algorithm='auto', leaf_size=30,
                   p=None, n_jobs=1)
    model.fit(samples)
    labels = model.labels_

    plt.scatter(x, y, c=labels, marker='x')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("After Cluster: DBSCAN")
    plt.show()


def dbscan_test03_circle(eps, min_samples):
    x, y = generate_circle(circle_data_count=1000,
                    centers=((1,1), (1,2), (2, 1), (2,2)),
                    radii=(0.7, 0.7, 0.7, 0.7),
                    show_plot=True)
    samples = np.column_stack((x, y))
    model = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean',
                   metric_params=None, algorithm='auto', leaf_size=30,
                   p=None, n_jobs=1)
    model.fit(samples)
    labels = model.labels_

    plt.scatter(x, y, c=labels, marker='x')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("After Cluster: DBSCAN")
    plt.show()


def spectral_test01_ring(k):
    x, y = genetrate_ring(1000, ((0.9, 1.1), (2.3, 2.4)), True)
    samples = np.column_stack((x, y))
    model = SpectralClustering(n_clusters=k)
    model.fit(samples)
    labels = model.labels_

    plt.scatter(x, y, c=labels, marker='x')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("After Cluster: Spectral")
    plt.show()


def spectral_test02_x(k):
    x, y = generate_x(1000, (2, 5), (6, 7), True)
    samples = np.column_stack((x, y))
    model = SpectralClustering(n_clusters=k)
    model.fit(samples)
    labels = model.labels_

    plt.scatter(x, y, c=labels, marker='x')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("After Cluster: Spectral")
    plt.show()


def spectral_test03_circle(k):
    x, y = generate_circle(circle_data_count=1000,
                    centers=((1,1), (1,2), (2, 1), (2,2)),
                    radii=(0.7, 0.7, 0.7, 0.7),
                    show_plot=True)
    samples = np.column_stack((x, y))
    model = SpectralClustering(n_clusters=k)
    model.fit(samples)
    labels = model.labels_

    plt.scatter(x, y, c=labels, marker='x')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("After Cluster: Spectral")
    plt.show()


if __name__ == "__main__":
    kmeans_test01_ring(2)
    kmeans_test02_x(2)
    kmeans_test03_circle(3)
    dbscan_test01_ring(0.5, 5)
    dbscan_test02_x(0.5, 5)
    dbscan_test03_circle(0.1, 33)
    spectral_test01_ring(2)
    spectral_test02_x(2)
    spectral_test03_circle(4)
