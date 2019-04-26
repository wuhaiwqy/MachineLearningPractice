import matplotlib.pyplot as plt
import numpy as np


# contour绘制等高线
def contour_function(x, y):
    return (1 - x / 2 + x ** 5 + y ** 3) * np.exp(- x ** 2 - y ** 2)


def demo01():
    x = np.linspace(-3, 3, 10)
    y = np.linspace(-3, 3, 10)
    X, Y = np.meshgrid(x, y)

    ax = plt.gca()
    ax.contour(X, Y, contour_function(X, Y), 8, colors='black')
    plt.show()


if __name__ == '__main__':
    demo01()
