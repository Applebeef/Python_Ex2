import math
import pandas as pd
import numpy as np


def three_x_plus_1(s):
    return pd.Series(np.where(s % 2 == 0, s / 2, 3 * s + 1))


def no_nans_idx(s):
    return pd.Series(index=s.values, data=np.where(np.isnan(s), False, True))


def partial_eq(s1, s2):
    indexes = s1.index.intersection(s2.index)
    return pd.Series(index=indexes, data=np.where(s1[indexes] == s2[indexes], True, False))


def get_n_largest(df: pd.DataFrame, n=0, how='col'):
    values = np.copy(df.values)
    values = values * -1
    if how == 'col':
        values.sort(axis=0)
        values = values * -1
        return pd.Series(index=df.columns, data=values[n])
    else:
        values.sort(axis=1)
        values = values * -1
        return pd.Series(index=df.index, data=values[:, n])


def upper(df):
    return df.apply(lambda x: x.apply(lambda y: y.upper() if isinstance(y, str) else y))


# Q2.2
def reindex_up_down(s):
    newIndeces = []

    for indStr in s.index:
        if indStr[0].isupper():
            newIndeces.append(indStr.upper())
        else:
            newIndeces.append(indStr.lower())

    s.index = newIndeces


# Q2.4
def partial_sum(s):
    sum = 0

    for val in s.values:
        if not pd.isna(val):
            sum += abs(val)

    return math.sqrt(sum)


# Q2.6
def dropna_mta_style(df, how='any'):
    df = df.dropna(how=how, axis=0)
    df = df.dropna(how=how, axis=1)

    return df


# Q2.8
def unique_dict(df, how='col'):
    # dropping null value columns to avoid errors
    df.dropna(inplace=True)

    # converting to dict
    if how == 'col':
        data_dict = df.to_dict('series')
    elif how == 'row':
        df = df.transpose()
        data_dict = df.to_dict('series')

    # unique all series in dict
    for col_dict in data_dict:
        data_dict[col_dict] = data_dict[col_dict].unique()

    return data_dict


# Q2.10
def stable_marriage(dames, gents, marriages):
    for couple in marriages:
        isNotAMatchPart1 = (dames.loc[couple[0]][0] != couple[1])
        isNotAMatchPart2 = (gents.loc[couple[1]][0] != couple[0])
        if isNotAMatchPart1 or isNotAMatchPart2:
            return False

    return True


if __name__ == "__main__":
    s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    # print(s)
    # print(three_x_plus_1(s))

    # Q2.2Tests
    # s = pd.Series(index=["TaL", "EdsGS", "tAGDS", "eLA"], data=[1, 2, 3, 4])
    # reindex_up_down(s)
    # print(s)

    # Q2.4Tests
    # s = pd.Series([1.5, 2.9, None, -5.6, -2, 45])
    # print(partial_sum(s))

    # Q2.6Test
    # df = pd.DataFrame({"name": [np.NAN, 'Batman', 'Catwoman'],
    #                    "toy": [np.NAN, 'Batmobile', 'Bullwhip'],
    #                    "born": [pd.NaT, pd.Timestamp("1940-04-25"), pd.NaT]})
    # df = dropna_mta_style(df)
    # print(df)

    # Q2.8Test
    # technologies = [
    #     ("Spark", 22000, '30days', 1000.0),
    #     ("PySpark", 25000, '50days', 2300.0),
    #     ("Spark", 23000, '55days', 1500.0)
    # ]
    # df = pd.DataFrame(technologies, columns=['Courses', 'Fee', 'Duration', 'Discount'])
    # print(unique_dict(df, 'row'))

    # Q2.10Test
    dames = pd.DataFrame.from_dict({'mary': ['john', 'mathew', 'dan'],
                                    'sarah': ['mathew', 'john', 'dan'],
                                    'eve': ['dan', 'mathew', 'john']}, orient='index')
    gents = pd.DataFrame.from_dict({'john': ['mary', 'sarah', 'eve'],
                                    'mathew': ['sarah', 'mary', 'eve'],
                                    'dan': ['eve', 'mary', 'sarah']}, orient='index')
    marriages = [('mary', 'john'), ('sarah', 'mathew'), ('eve', 'dan')]
    print(stable_marriage(dames, gents, marriages))
    
    
    df = pd.DataFrame(np.random.randint(0, 100, size=(5, 4)), columns=list('ABCD'))
    print(df)
    print(get_n_largest(df, n=3))