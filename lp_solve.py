import numpy as np
import cvxpy as cx


def solve_lp_down(C, nu, phi_x, y_true):
    k = len(C)
    y_down = cx.Variable(k)
    y_up = cx.Variable(k)
    v = cx.Variable()
    u = cx.Variable()
    ones = np.ones((k, 1))
    nu = nu.reshape((-1, 1))

    C_aug = C + ones.dot(nu.T).dot(phi_x - phi_x.dot(y_true).dot(ones.T))
    constraints_down = [
        v <= C_aug * y_down,
        cx.sum_entries(y_down) == 1,
        y_down >= 0,
    ]
    problem_down = cx.Problem(cx.Maximize(v), constraints_down)
    problem_down.solve()

    return y_down.value, v.value


def solve_lp_up(C, nu, phi_x):
    k = len(C)
    y_up = cx.Variable(k)
    u = cx.Variable()
    ones = np.ones((k, 1))
    nu = nu.reshape((-1, 1))
    C_aug = C + ones.dot(nu.T).dot(phi_x)
    constraints = [
        u >= y_up.T * C_aug,
        cx.sum_entries(y_up) == 1,
        y_up >= 0
    ]
    problem = cx.Problem(cx.Minimize(u), constraints)
    problem.solve()
    return y_up.value, u.value
