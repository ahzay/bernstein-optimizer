# 2.1.2. Brachistrochrone
from bernstein_optimizer import Constraint
import numpy as np
from scipy.integrate import quad

u_sz = 1
x_sz = 3

tf = 1  # init
tf_max = 10

g = -9.81
xb = 2
yb = -2
xa = 0
ya = 0

tf_is_variable = True


def xdot(x, u, t):
    return [x[2] * np.sin(u[0]), x[2] * np.cos(u[0]), g * np.cos(u[0])]


def costf(x, u, tf):
    return tf


def final_con_fun(x, u, tf):
    return [x(tf)[0] - xb, x(tf)[1] - yb]


constraints = [Constraint(final_con_fun, [0, 0], [0, 0])]

initial_dynamics = [xa, ya, 0]

u_bounds = [-np.pi, np.pi]
