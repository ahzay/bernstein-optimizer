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

from analytical_examples.example_4 import *

ns = [10, 20, 30]

plt.figure()

for n in ns:
    opt = BernsteinOptimizer(
        dynamicsf=xdot,
        costf=costf,
        constraints=constraints,
        u_sz=u_sz,
        x_sz=x_sz,
        u_is_sigm_limited=False,
        n_max=n,
        u_bounds=[-10, 10],
        tf_is_variable=False,
        init_is_random=False,
        n_is_adaptive=False,
        initial_dynamics=initial_dynamics,
        tf=tf,
    )

    p = opt.minimize(method="SLSQP")
    # p = opt.minimize()
    print(p)

    x, u, s = opt.get_s_of_t(p)
    i = 2
    # plotting
    t = np.linspace(0, tf, 100)

    s_values = np.array([s(ti) for ti in t])
    analytical_s_values = np.array([analytical_s(ti) for ti in t])

    ylabel = "$u_" + str(i - x_sz + 1) + "$"

    # plt.plot(t, analytical_s_values[:, i], label="analytical " + ylabel)
    plt.plot(
        t,
        s_values[:, i],
        label="experimental " + ylabel + ", $n=" + str(n) + "$",
        linestyle="--",
    )
    plt.xlabel("$t$")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid()
    plt.gca().set_aspect("auto")
    plt.savefig(
        "figures/" + sys.argv[0][:-3] + "_" + ylabel[1:-1] + "_" + str(n) + ".pdf"
    )

    # plt.show()
