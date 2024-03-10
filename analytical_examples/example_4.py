import numpy as np
from scipy.integrate import quad
import sys
from bernstein_optimizer import Constraint

u_sz = 1
x_sz = 2
tf = 1


def xdot(x, u, t):
    return [x[1], u[0]]


def final_con_fun(x, u):
    return x(tf)[0] + x(tf)[1] - 1


def costf(x, u):
    return quad(lambda t: 0.5 * u(t)[0] ** 2, 0, tf)[0]


def analytical_s(t):
    return [
        -(t**3) / 14 + 3 * t**2 / 7,
        -3 * t**2 / 14 + 6 * t / 7,
        -6 * t / 14 + 6 / 7,
    ]


constraints = [Constraint(final_con_fun, 0, 0)]

initial_dynamics = [0, 0]
