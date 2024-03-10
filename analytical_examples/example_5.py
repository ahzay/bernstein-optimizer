from bernstein_optimizer import Constraint
import numpy as np
from scipy.integrate import quad, solve_ivp

tf = 3

uu = 1 / 3
ul = -1 / 3
u_sigm_bounds = [ul, uu]


x_sz = 1
u_sz = 3


def xdot(x, u, t):
    return [
        u[0] + u[1] + u[2],
    ]


def final_con_fun(x, u):
    return [x(tf)[0] - 1]


def x_con_fun(x, u):
    return [quad(lambda t: x(t)[0], 0, tf)[0]]


def costf(x, u):
    # return quad(lambda t: x(t)[0], 0, tf)[0]
    return quad(lambda t: x(t)[0], 0, tf)[0] ** 2


tf_max = 10

constraints = [Constraint(final_con_fun, 0, 0), Constraint(x_con_fun, 0, np.inf)]
# constraints = [Constraint(final_con_fun, 0, 0)]

initial_dynamics = [1]


def analytical_s(t):
    if t >= 0 and t < 1:
        return [1 - t, -1]
    if t >= 1 and t <= 2:
        return [0, 0]
    if t > 2 and t <= 3:
        return [t - 2, 1]
