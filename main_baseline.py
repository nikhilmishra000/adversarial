from load_data import *
from features import *
from lp_solve import *


"""
Usage:
>>> clf = train()
>>> score(clf)
"""

X, y = load('cifar')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=10000)

opts = tfu.struct(
    batch_size=32,
    iters=10000,
    c_mode='ones',

    dim_x=X.shape[1],
    dim_y=y.shape[1],

    sizes=[
        {
            'kernel': (5, 5, 64),
            'bias': True,
            'batch_norm': {'decay': 0.99},
        },
        {
            'kernel': (5, 5, 64),
            'bias': True,
            'batch_norm': {'decay': 0.99},
            'pool': {
                'type': 'max',
                'kernel': (2, 2),
            },
        },
        {
            'kernel': (3, 3, 64),
            'pad': 'VALID',
            'bias': True,
            'batch_norm': {'decay': 0.99},
        },
        {
            'kernel': (3, 3, 64),
            'pad': 'VALID',
            'bias': True,
            'batch_norm': {'decay': 0.99},
            'pool': {
                'type': 'max',
                'kernel': (2, 2),
            },
        },
        [
            {'size': 256},
        ],
    ],
    dim_phi=128,
    dim_nu=128 * y.shape[1],

    solver_type='Adam',
    alpha=1e-4,
    beta1=0.9,
    beta2=0.999,
    # lr_decay=0.9,
    # lr_step=1000,
    # min_alpha=1e-4,
)

phi = DeepBaseline(opts)


def train():
    (N, dx), (_, k) = X_train.shape, y_train.shape
    Nt = len(X_test)
    assert N == len(y_train) and Nt == len(y_test)

    if opts.c_mode == 'random':
        C = np.random.uniform(size=(k, k))
    elif opts.c_mode == 'ones':
        C = np.ones((k, k))
    else:
        raise ValueError('C mode: %s' % opts.mode)
    C[range(k), range(k)] = 0

    # train phi and nu jointly with Adam
    buf = [[], []]
    for _ in range(opts.iters):
        idx = np.random.randint(N, size=(opts.batch_size))
        x, y = X_train[idx], y_train[idx]
        out = phi.train(x=x, y=y)
        buf[0].append(out.loss)

        if out.it % 50 == 0:
            loss = np.mean(buf[0][-50:])
            idx = np.random.randint(Nt, size=(opts.batch_size))
            y_up_star = phi.test(x=X_test[idx])
            test_loss = np.equal(y_up_star.argmax(-1),
                                 y_test[idx].argmax(-1)).mean()
            buf[1].append(test_loss)
            print 'iter: %d, lr: %.3e, loss: %.4f, test: %.4f' % (
                out.it, out.lr, loss, test_loss
            )

    def classifier(X):
        y_up_star = np.concatenate([
            phi.test(x=X[i:i + opts.batch_size])
            for i in range(0, len(X), opts.batch_size)
        ])
        return y_up_star

    return tfu.struct(phi=phi, C=C, predict=classifier, buf=buf)


def score(clf):
    y_hat_train = clf.predict(X_train)
    train_acc = np.equal(y_hat_train.argmax(-1), y_train.argmax(-1)).mean()
    train_loss = np.einsum('bi,ij,bj->b', y_hat_train, clf.C, y_train).mean()

    y_hat_test = clf.predict(X_test)
    test_acc = np.equal(y_hat_test.argmax(-1), y_test.argmax(-1)).mean()
    test_loss = np.einsum('bi,ij,bj->b', y_hat_test, clf.C, y_test).mean()

    print 'train / test'
    print 'accuracy:', train_acc, test_acc
    print 'loss:', train_loss, test_loss
