from bernstein_optimizer import Constraint
import numpy as np
from scipy.integrate import quad, solve_ivp

g = 9.81

y_c = 100

tf_t = 5.47414
tf = 1
ul = 0
uu = 30

T = g * tf_t / uu
u_sigm_bounds = [ul, uu]


x_sz = 2
u_sz = 1


def xdot(x, u, t):
    return [
        x[1],
        u[0] - g,
    ]


# def final_con_fun1(x, u):
def final_con_fun1(x, u, tf):
    return [x(tf)[0] - y_c]


# def final_con_fun2(x, u):
def final_con_fun2(x, u, tf):
    return [x(tf)[1]]


# def costf(x, u):
def costf(x, u, tf):
    return quad(lambda t: u(t)[0], 0, tf)[0]


def analytical_u(t):
    # T = np.sqrt(2 * g * y_c / uu**2)
    if t < T:
        return [uu]
    else:
        return [0]


tf_max = 10

constraints = [Constraint(final_con_fun1, 0, 0), Constraint(final_con_fun2, 0, 0)]

initial_dynamics = [0, 0]

analytical_x = solve_ivp(
    lambda t, x: xdot(x, analytical_u(t), t),
    [0, tf_t],
    initial_dynamics,
    dense_output=True,
    method="RK45",
    max_step=0.001,
    atol=1,
    rtol=1,
).sol


def analytical_s(t):
    x = analytical_x
    return [x(t)[0], x(t)[1], analytical_u(t)[0]]
