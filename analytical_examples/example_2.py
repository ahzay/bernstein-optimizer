import numpy as np
from scipy.integrate import quad

from bernstein_optimizer import Constraint

tf = 1
u_sz = 1
x_sz = 1


def xdot(x, u, t):
    return x[0] + u[0]


def final_con_fun(x, u):
    return x(tf)[0] - 2


def costf(x, u):
    return quad(lambda t: u(t)[0] ** 2, 0, tf)[0]


def analytical_s(t):
    return [
        2 * np.sinh(t) / np.sinh(1),
        2 * np.exp(-t) / np.sinh(1),
    ]


constraints = [Constraint(final_con_fun, 0, 0)]

initial_dynamics = [0]
