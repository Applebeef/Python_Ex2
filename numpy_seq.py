import numpy as np


def half(x):
    z = x[1::2, ::2]
    z = z / 2
    return z


def outer_product(x, y):
    return x.reshape(-1, 1) * y


def extract_logical(x, arr):
    z = x[arr == np.round(arr)]
    ind = np.where(arr == np.round(arr), True, False)
    return z, ind


def extract_integer(x, arr):
    z = x[arr == np.round(arr)]
    ind = np.zeros((x.ndim, z.size), dtype=int)
    return z, ind


def calc_norm(x, axis=0):
    return np.sum(x ** 2, axis=1 if axis == 0 else 0) ** 0.5


def normalize(x, axis=0):
    return x / calc_norm(x, axis)


def matrix_norm(x, k=1000):
    n = x.shape[0]
    X = np.random.randn(n, k)
    X = normalize(X, axis=1)
    Z = np.dot(x, X)
    u = calc_norm(Z, axis=1)
    return np.max(u)


def segment(im, thresh=128):
    if im.ndim == 3:
        im = im.mean(axis=2)
    return np.where(im < thresh, 0, 255)


if __name__ == "__main__":
    x = 3  # delete this line
