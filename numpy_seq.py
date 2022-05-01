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


# Ex 1.6
def get_minor_from_mat(A, i, j):  # Getting the new sub matrix without col j.
    return [row[: j] + row[j + 1:] for row in (A[: i] + A[i + 1:])]


def det(A):  # Recursive func - Calculates the determinant of square matrix A.
    if len(A) == 2:
        value = (A[0][0] * A[1][1]) - (A[1][0] * A[0][1])
        return value

    res = 0
    for col in range(len(A)):
        sign = (-1) ** col
        subDet = det(get_minor_from_mat(A, 0, col))
        res += (sign * A[0][col]) * subDet

    return res


# Ex 1.8
#def linearville(robber, policeman):
    # robber = numpy.ndarray of shape (h,n,d) of integers.
    # policeman = numpy.ndarray of shape (h,n) of integers.
    # if check_robber_array():
    #     for hour in robber.size: # h hours int
    #         for shop in hour.size: # n shops int
    #             for day in shop.size: # d days int
    #                 if day != 1 or day != 0:
    #                     print("Wrong robber data - more than one robber")
    #                     return False
    #                 # if
    # else:
    #     print("Invalid robbery!")


# def check_robber_array():


# Ex 1.9
def is_all_elements_distinct(matrix):
    size = len(matrix)

    elementsSet = set()
    for i in range(0, size):
        for j in range(0, size):
            elementsSet.add(matrix[i][j])

    return len(matrix)*len(matrix) == len(elementsSet)


def is_magic(matrix):
    sumDiagonal1 = np.trace(matrix)

    secondDiagonal = np.diagonal(np.fliplr(matrix))
    sumDiagonal2 = sum(secondDiagonal)

    if sumDiagonal1 != sumDiagonal2:
        return False

    rowsSum = np.sum(matrix, axis=0)
    colsSum = np.sum(matrix, axis=1)

    # Check if all values in an array are equal to its first element
    isValuesInRowsAreSame = np.all(rowsSum == rowsSum[0])
    isValuesInColsAreSame = np.all(colsSum == colsSum[0])

    if isValuesInRowsAreSame and isValuesInColsAreSame and (
            rowsSum[0] == colsSum[0] == sumDiagonal1) and is_all_elements_distinct(matrix):
        return True

    return False


if __name__ == "__main__":
    x = 3  # delete this line

    # mat = [[1, 0, 2, -1],
    #        [3, 0, 0, 5],
    #        [2, 1, 4, -3],
    #        [1, 0, 5, 0]]

    mat = [[4, 9, 2],
           [3, 5, 7],
           [8, 1, 6]]

    if is_magic(mat):
        print("YESS")
    else:
        print("NOOO")

    # print('Determinant of the matrix is :', det(mat))
