from bernstein_optimizer import Constraint
import numpy as np
from scipy.integrate import quad


x_sz = 4
u_sz = 1

u_bounds = [-100, 100]


V = 2.5

K = -0.0173

tf = 100

tf_max = 300


def xdot(x, u, t):
    return [V * np.cos(x[2]), V * np.sin(x[2]), x[3], K * u[0]]


def Circle(xx, yy, aa, bb, rr):
    return (xx - aa) ** 2 + (yy - bb) ** 2 - rr**2


a = 200
b = 200
r = 10


def final_con_fun(x, u, tf):
    return Circle(x(tf)[0], x(tf)[1], a, b, r)


def final_con_tan_fun(x, u, tf):
    return (x(tf)[0] - a) * np.cos(x(tf)[2]) + (x(tf)[1] - b) * np.sin(x(tf)[2])


a1 = 100
b1 = 100
r1 = 50


def circle_cost_fun(x, u, t):
    c = Circle(x(t)[0], x(t)[1], a1, b1, r1)
    if c <= 0:
        return np.abs(c)
    else:
        return 0


def costf(x, u, tf):
    return np.linalg.norm(
        [
            quad(lambda t: u(t)[0] ** 2, 0, tf)[0],
            # quad(lambda t: circle_cost_fun(x, u, t), 0, tf)[0],
        ]
    )


constraints = [Constraint(final_con_fun, 0, 0), Constraint(final_con_tan_fun, 0, 0)]

initial_dynamics = [0, 0, np.pi / 2, 0]
