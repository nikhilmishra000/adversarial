import numpy as np
import cvxpy as cx


def batch_solve_lp(C_augs):
    K, down_star, up_star, p_star = C_augs.shape[1], [], [], []
    for C_aug in C_augs:
        y_down = cx.Variable(K)
        v = cx.Variable()
        constr = [
            v <= C_aug * y_down,
            cx.sum_entries(y_down) == 1,
            y_down >= 0,
        ]
        prob = cx.Problem(cx.Maximize(v), constr)
        prob.solve()
        down_star.append(y_down.value)
        up_star.append(constr[0].dual_value)
        p_star.append(prob.value)
    return \
        np.array(down_star).squeeze(),\
        np.array(up_star).squeeze(), \
        np.array(p_star).ravel()
