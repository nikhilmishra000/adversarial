from load_data import *
from features import *
from lp_solve import *
from scipy import optimize
import argparse

"""
Usage:
>>> clf = train()
>>> score(clf)
"""

parser = argparse.ArgumentParser(
    description='Choosing dataset for adversarial training.')
parser.add_argument('dataset', type=str, help='name of dataset to run')

args = parser.parse_args()
X, y = load(args.dataset)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_pct=36)

opts = tfu.struct(
    iters=100,
    tol=1e-5,
    c_mode='ones',

    dim_x=X.shape[1],
    dim_y=y.shape[1],

    dim_nu=X.shape[1] ** 2 * y.shape[1],
)

phi = BasicPhi(opts)


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

    nu = phi.nu.eval(phi.session)
    phi_xy = phi.phi(x=X_train)

    def func_grad(nu):
        phi.session.run(phi.nu.assign(nu))
        C_augs = phi.C(c=C, x=X_train, y=y_train)
        y_down_star, y_up_star, p_star = batch_solve_lp(C_augs)
        f = p_star.mean()
        g = np.einsum('bdk,bk->d', phi_xy, y_down_star - y_train) / N
        return f, g

    def callback(nu):
        print 'bfgs:', func_grad(nu)[0]

    nu_final, v_star, info = optimize.fmin_l_bfgs_b(
        func_grad, nu, callback=callback,
        pgtol=opts.tol, maxiter=opts.iters,
    )
    info.pop('grad')
    print v_star, info

    phi.session.run(phi.nu.assign(nu_final))

    def classifier(X):
        C_aug = phi.C_test(c=C, x=X)
        y_down_star, y_up_star, _ = batch_solve_lp(C_aug)
        return y_up_star

    return tfu.struct(phi=phi, C=C, predict=classifier)


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
