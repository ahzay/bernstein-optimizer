from bernstein_optimizer import BernsteinOptimizer
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
    }
)
import sys

from analytical_examples.example_1 import *

opt = BernsteinOptimizer(
    dynamicsf=xdot,
    costf=costf,
    constraints=constraints,
    u_sz=u_sz,
    x_sz=x_sz,
    u_is_sigm_limited=False,
    n_max=3,
    u_bounds=[-10, 10],
    tf_is_variable=False,
    init_is_random=False,
    n_is_adaptive=False,
    initial_dynamics=initial_dynamics,
    tf=tf,
)

# p = opt.minimize(method="SLSQP")
p = opt.minimize()
print(p)

x, u, s = opt.get_s_of_t(p)

# plotting
t = np.linspace(0, tf, 100)

s_values = np.array([s(ti) for ti in t])
analytical_s_values = np.array([analytical_s(ti) for ti in t])

for i in range(s_values.shape[1]):
    plt.figure()
    if i < x_sz:
        ylabel = "$x_" + str(i + 1) + "$"
    else:
        ylabel = "$u_" + str(i - x_sz + 1) + "$"

    plt.plot(t, analytical_s_values[:, i], label="analytical " + ylabel)
    plt.plot(t, s_values[:, i], label="experimental " + ylabel, linestyle="--")
    plt.xlabel("$t$")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid()
    plt.gca().set_aspect("auto")
    plt.savefig("figures/" + sys.argv[0][:-3] + "_" + ylabel[1:-1] + ".pdf")

plt.show()
