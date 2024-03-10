from bernstein_optimizer import Constraint
import numpy as np
from scipy.integrate import quad

u_sz = 1
x_sz = 3

tf = 6


def xdot(x, u, t):
    return [
        (1 - x[1] ** 2) * x[0] - x[1] + u[0],
        x[0],
        0.5 * (x[0] ** 2 + x[1] ** 2 + u[0] ** 2),
    ]


df = 0.5
x1l = -0.3
x2l = 0.1

u_bounds = [-0.4, 1]


def final_con_fun(x, u):
    return x(tf)[1] - x(tf)[0] + df


def x1_min_con_fun(x, u):  # bigger or equal than 0
    return quad(lambda t: (x(t)[0] - x1l) ** 2, 0, tf)


def x2_min_con_fun(x, u):  # bigger or equal than 0
    return quad(lambda t: (x(t)[1] - x2l) ** 2, 0, tf)


def costf(x, u):
    return np.linalg.norm(
        [
            x(tf)[2],
        ]
    )


# constraints = [
#     Constraint(final_con_fun, 0, 0),
#     Constraint(x1_min_con_fun, 0, np.inf),
#     Constraint(x2_min_con_fun, 0, np.inf),
# ]

constraints = [
    Constraint(final_con_fun, 0, 0),
]

initial_dynamics = [1, 1, 0]
