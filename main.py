from load_data import *
from features import *
from lp_solve import *
from scipy import optimize

"""
Usage:
>>> clf = train(X_train, Y_train, Phi, opts)
>>> score(clf, X_train, y_train, X_test, y_test)
"""

X, y = load('wine')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=36)

opts = tfu.struct(
    batch_size=32,
    iters=1000,
    bfgs_iters=100,
    c_mode='random',

    dim_x=X.shape[1],
    dim_y=y.shape[1],

    # for BasicPhi
    # dim_nu=X.shape[1] ** 2 * y.shape[1], # dim_nu = dim_x**2 * dim_y

    # for DeepPhi
    sizes=[64, 64],
    dim_phi=32,
    dim_nu=32 * y.shape[1],  # dim_nu = dim_phi * dim_y

    solver_type='Adam',
    alpha=1e-3,
    beta1=0.9,
    beta2=0.999,
    lr_decay=0.9,
    lr_step=100,
    min_alpha=1e-4,
)

# Phi = BasicPhi(opts)
Phi = DeepPhi(opts)


def train(X, Y, phi, opts):
    (nx, dx), (ny, k) = X.shape, Y.shape
    assert nx == ny

    if opts.c_mode == 'random':
        C = np.random.uniform(size=(k, k))
    elif opts.c_mode == 'ones':
        C = np.ones((k, k))
    else:
        raise ValueError('C mode: %s' % opts.mode)
    C[range(k), range(k)] = 0

    # train phi and nu jointly with Adam
    phi.reset()
    for _ in range(opts.iters):
        idx = np.random.randint(len(X), size=(opts.batch_size))
        x, y = X[idx], Y[idx]
        C_augs = phi.C(c=C, x=x, y=y)
        y_down_star, y_up_star, p_star = batch_solve_lp(C_augs)

        out = phi.train(c=C, x=x, yt=y, yu=y_up_star, yd=y_down_star)
        if out.it % 10 == 0:
            print 'iter: %d, lr: %.3e, loss: %.4f' % (out.it, out.lr, out.loss)

    # finetune nu with BFGS
    nu = phi.nu.eval(phi.session)
    phi_xy = phi.phi(x=X)

    def func_grad(nu):
        phi.session.run(phi.nu.assign(nu))
        C_augs = phi.C(c=C, x=X, y=Y)
        y_down_star, y_up_star, p_star = batch_solve_lp(C_augs)
        f = p_star.mean()
        g = np.einsum('bdk,bk->d', phi_xy, y_down_star - Y) / len(Y)
        return f, g

    def callback(nu):
        print 'bfgs:', func_grad(nu)[0]

    nu_final, v_star, info = optimize.fmin_l_bfgs_b(
        func_grad, nu, callback=callback,
        pgtol=1e-5, maxiter=opts.bfgs_iters,
    )
    info.pop('grad')
    print v_star, info

    phi.session.run(phi.nu.assign(nu_final))

    def classifier(X):
        C_aug = phi.C(c=C, x=X, y=np.zeros((len(X), k)))
        y_down_star, y_up_star, _ = batch_solve_lp(C_aug)
        return y_up_star

    return tfu.struct(phi=phi, C=C, predict=classifier)


def score(clf, X_train, y_train, X_test, y_test):
    y_hat_train = clf.predict(X_train)
    train_acc = np.equal(y_hat_train.argmax(-1), y_train.argmax(-1)).mean()
    train_loss = np.einsum('bi,ij,bj->b', y_hat_train, clf.C, y_train).mean()

    y_hat_test = clf.predict(X_test)
    test_acc = np.equal(y_hat_test.argmax(-1), y_test.argmax(-1)).mean()
    test_loss = np.einsum('bi,ij,bj->b', y_hat_test, clf.C, y_test).mean()

    print 'train / test'
    print 'accuracy:', train_acc, test_acc
    print 'loss:', train_loss, test_loss
