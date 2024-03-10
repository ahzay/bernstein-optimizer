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

from nonanalytical_examples.example_2 import *

opt = BernsteinOptimizer(
    dynamicsf=xdot,
    costf=costf,
    constraints=constraints,
    u_sz=u_sz,
    x_sz=x_sz,
    u_is_sigm_limited=True,
    u_sigm_bounds=u_bounds,
    n_max=5,
    u_bounds=[-100, 100],
    tf_is_variable=False,
    init_is_random=False,
    n_is_adaptive=True,
    initial_dynamics=initial_dynamics,
    tf=tf,
)

p = opt.minimize(method="SLSQP")
# p = opt.minimize()
if opt.tf_is_variable:
    tf = p[-1]

print(p)

x, u, s = opt.get_s_of_t(p)

# plotting
t = np.linspace(0, tf, 100)

s_values = np.array([s(ti) for ti in t])
plt.figure()
for i in range(s_values.shape[1]):
    if i < x_sz:
        ylabel = "$x_" + str(i + 1) + "$"
    else:
        if i == x_sz:
            plt.figure()
        ylabel = "$u_" + str(i - x_sz + 1) + "$"

    plt.plot(t, s_values[:, i], label=ylabel)
    plt.xlabel("$t$")
    plt.ylabel("$" + ylabel[1] + "$")
    plt.legend()
    plt.grid()
    plt.gca().set_aspect("auto")
    plt.savefig("figures/" + sys.argv[0][:-3] + "_" + ylabel[1] + ".pdf")

plt.show()
