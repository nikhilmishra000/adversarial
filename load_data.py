import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize


def one_hot(i, k):
    y = np.zeros(k)
    y[i] = 1
    return y


def _load_iris(**kwargs):
    data = pd.read_csv('data/iris.data', header=None)

    X = normalize(
        data.values[:, 1:-1].astype(np.float32),
        axis=1
    )
    Y_raw = data.values[:, -1]
    classes = {y: i for i, y in enumerate(np.unique(Y_raw))}
    K = len(classes)

    Y = np.array([
        one_hot(classes[y], K) for y in Y_raw
    ])

    return X, Y

loaders = {
    'iris': _load_iris,
}


def load(name, **kwargs):
    data = loaders[name](**kwargs)
    return data
