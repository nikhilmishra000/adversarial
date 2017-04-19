import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import os
import cPickle
from os import listdir
from os.path import isfile, join


def one_hot(i, k):
    y = np.zeros(k)
    y[i] = 1
    return y


def _load_iris(**kwargs):
    data = pd.read_csv('data/iris.data', header=None)

    X = normalize(
        data.values[:, :-1].astype(np.float32),
        axis=1
    )
    Y_raw = data.values[:, -1]
    classes = {y: i for i, y in enumerate(np.unique(Y_raw))}
    K = len(classes)

    Y = np.array([
        one_hot(classes[y], K) for y in Y_raw
    ])

    return X, Y


def _load_wine(**kwargs):
    data = pd.read_csv('data/wine.data', header=None)

    X = normalize(
        data.values[:, 1:].astype(np.float32),
        axis=1
    )
    Y_raw = data.values[:, 0]
    classes = {y: i for i, y in enumerate(np.unique(Y_raw))}
    K = len(classes)

    Y = np.array([
        one_hot(classes[y], K) for y in Y_raw
    ])

    return X, Y


def _load_cifar(**kwargs):
    mypath = os.getcwd() + "/data" + "/cifar-10-batches-py"
    batches = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    X = None
    y = []
    for f in batches:
        if f.startswith("data") or f.startswith("test"):
            f = join(mypath, f)
            with open(f, 'rb') as fo:
                data = cPickle.load(fo)
                X_cur = data['data']
                y_cur = data['labels']
                if X == None:
                    X = X_cur
                else:
                    X = np.concatenate((X, X_cur), axis=0)
                # import pdb
                # pdb.set_trace()
                print(len(y))
                y.extend(y_cur)
    classes = {y: i for i, y in enumerate(np.unique(y))}
    K = len(classes)

    Y = np.array([
        one_hot(classes[y_cur], K) for y_cur in y
    ])

    return X, Y


loaders = {
    'iris': _load_iris,
    'wine': _load_wine,
    'cifar': _load_cifar,
}


def load(name, **kwargs):
    data = loaders[name](**kwargs)
    return data
