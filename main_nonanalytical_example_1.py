from bernstein_optimizer import BernsteinOptimizer
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
    }
)
import matplotlib.patches as patches
import sys

from nonanalytical_examples.example_1 import *

opt = BernsteinOptimizer(
    dynamicsf=xdot,
    costf=costf,
    constraints=constraints,
    u_sz=u_sz,
    x_sz=x_sz,
    u_is_sigm_limited=False,
    # u_sigm_bounds=u_bounds,
    n_max=5,
    u_bounds=u_bounds,
    tf_is_variable=True,
    init_is_random=False,
    n_is_adaptive=False,
    initial_dynamics=initial_dynamics,
    tf=tf,
    tf_max=tf_max,
)

p = opt.minimize(method="SLSQP")

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
    plt.grid(visible=True)
    plt.gca().set_aspect("auto")
    plt.savefig("figures/" + sys.argv[0][:-3] + "_" + ylabel[1] + ".pdf")

# xy plot
plt.figure()
plt.plot(s_values[:, 0], s_values[:, 1], label="path")
plt.gca().add_patch(patches.Circle((a, b), r, fill=False))
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.legend()
plt.grid(visible=True)
plt.gca().set_aspect("equal")
plt.savefig("figures/" + sys.argv[0][:-3] + "_" + "xy" + ".pdf")


plt.show()
