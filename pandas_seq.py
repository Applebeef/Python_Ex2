import pandas as pd


def three_x_plus_1(s: pd.Series) -> pd.Series:
    # can we use numpy in this part? if so, we can use np.where()
    return s.apply(lambda x: x / 2 if x % 2 == 0 else 3 * x + 1)


if __name__ == "__main__":
    s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    print(s)
    print(three_x_plus_1(s))
