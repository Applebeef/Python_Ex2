import pandas as pd
import numpy as np


def three_x_plus_1(s):
    return pd.Series(np.where(s % 2 == 0, s / 2, 3 * s + 1))


def no_nans_idx(s):
    return pd.Series(index=s.values, data=np.where(np.isnan(s), False, True))


def partial_eq(s1, s2):
    indexes = s1.index.intersection(s2.index)
    return pd.Series(index=indexes, data=np.where(s1[indexes] == s2[indexes], True, False))


def get_n_largest(df, n=0, how='col'):
    # TODO implement 2.7
    pass


def upper(df):
    return df.apply(lambda x: x.apply(lambda y: y.upper() if isinstance(y, str) else y))


if __name__ == "__main__":
    df = pd.DataFrame(np.random.randint(0, 100, size=(5, 4)), columns=list('ABCD'))
    df.iloc[2, 2] = 'string'
    print(df)
    print(upper(df))
