from load_data import *
from features import *
from lp_solve import *
from scipy import optimize

X, y = load('iris')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=30)

opts = tfu.struct(
    iters=50,
    tol=1e-5,
    c_mode='random',

    dim_x=X.shape[1],
    dim_y=y.shape[1],
    dim_phi=64,
    dim_nu=64 * y.shape[1],


    sizes=[128],
)

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

    nu = np.zeros((opts.dim_nu, 1))

    for it in range(opts.outer_iters):
        Phi_ndk = phi.phi(x=X)

        def func_grad(nu):
            f, g = 0, 0
            for phi_x, y_true in zip(Phi_ndk, Y[..., None]):
                # solve the inner LP
                y_down_star, v = solve_lp_down(C, nu, phi_x, y_true)

                # compute the subgradient
                f += v
                g += phi_x.dot(y_down_star - y_true)
            return f / len(Y), g / len(Y)

        def callback(nu):
            print func_grad(nu)[0]

        nu, v_star, info = optimize.fmin_l_bfgs_b(
            func_grad, nu, callback=callback,
            pgtol=opts.tol, maxiter=opts.inner_nu_iters
        )
        info.pop('grad')
        print v_star, info

        Y_down = []
        for phi_x, y_true in zip(Phi_ndk, Y[..., None]):
            y_down_star, _ = solve_lp_down(C, nu, phi_x, y_true)
            Y_down.append(y_down_star)
        Y_down = np.array(Y_down)
        print Y_down.shape

        subgrad = nu.dot((Y_down - Y).T)
        for _ in range(opts.inner_phi_iters):
            print phi.train(x=X, g=subgrad).loss

    def classifier(X_bd):
        Y = []
        Phi_bdk = phi.phi(x=X_bd)
        for phi_x in Phi_bdk:
            y_up_star, u = solve_lp_up(C, nu, phi_x)
            Y.append(y_up_star)
        return np.squeeze(Y)

    return tfu.struct(nu=nu, C=C, predict=classifier)


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
