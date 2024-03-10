import numpy as np
from numba import jit
import math
import inspect
from math import gamma
from scipy.misc import derivative
from scipy.integrate import odeint, quad, solve_ivp, fixed_quad, dblquad

# from scipy.special import binom
from scipy.optimize import minimize, NonlinearConstraint, dual_annealing
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional, Callable
from concurrent.futures import ProcessPoolExecutor


@jit(nopython=True)
def binom(n, k):
    return gamma(n + 1) / (gamma(k + 1) * gamma(n - k + 1))


# TODO: check this is fine
@jit(nopython=True)
def bernstein_poly(p, t, tf):
    # tf_b = p[-1]
    # weights = p[:-1]
    weights = p
    n = len(weights) - 1
    u = t / tf
    n_values = np.arange(n + 1)
    B = np.sum(
        weights
        * np.array([binom(n, i) for i in n_values])
        * (u**n_values)
        * ((1 - u) ** (n - n_values))
    )
    return B


def sigmoid_limiter(x, lower_bound, upper_bound):
    # Parameters to adjust the steepness of the sigmoid around the bounds
    x = np.array(x)
    steepness = 5 / (upper_bound - lower_bound)

    # Linear scaling to ensure the middle region is approximately equal to the input
    middle = (upper_bound + lower_bound) / 2
    scaled_x = steepness * (x - middle)
    # Apply the sigmoid function
    limited_x = (upper_bound - lower_bound) * (
        1 / (1 + np.exp(-scaled_x))
    ) + lower_bound

    return limited_x


def fun_sigmoid_limited(fun, lower_bound, upper_bound):
    return lambda t: sigmoid_limiter(fun(t), lower_bound, upper_bound)


class Constraint:
    def __init__(self, fun, lb, ub):
        self.fun = fun
        self.lb = lb
        self.ub = ub


class BernsteinOptimizer:
    def __init__(
        self,
        dynamicsf,  # f(x, u, t)
        u_sz,
        x_sz,
        u_is_sigm_limited,
        costf,  # f(x, u) or # f(x, u, tf) if tf_is_variable
        constraints,  # NonLinearConstraint
        n_max,
        u_bounds,
        tf_is_variable,
        tf,
        init_is_random,
        u_sigm_bounds=None,
        tf_max=None,
        n_is_adaptive=True,
        initial_dynamics=None,
    ):
        self.dynamicsf = dynamicsf
        self.costf = costf
        self.constraints = constraints
        self.n_max = n_max
        self.tf_is_variable = tf_is_variable
        if tf_is_variable:
            if tf_max is None:
                raise ValueError("If tf is variable then tf_max has to be set")
            else:
                self.tf_max = tf_max
        else:
            if tf is None:
                raise ValueError("If tf is not variable then it has to be set")

        self.tf = tf
        self.init_is_random = init_is_random
        self.n_is_adaptive = n_is_adaptive
        self.u_bounds = u_bounds
        self._init_checks()  # checks fun signatures of dynamics, cost and constraint functions
        self.u_sz = u_sz
        self.u_is_sigm_limited = u_is_sigm_limited
        if u_is_sigm_limited:
            if u_sigm_bounds == None:
                raise ValueError(
                    f"u is sigma limited so u sigma bounds should be defined, got None"
                )
            else:
                self.u_sigm_bounds = u_sigm_bounds
        self.x_sz = x_sz
        self.initial_dynamics = initial_dynamics
        if len(initial_dynamics) != self.x_sz:
            raise ValueError(
                f"Initial dynamics size expected {self.x_sz}, got {len(initial_dynamics)}"
            )
        self.wrapped_constraints = self._wrap_constraints()
        self.wrapped_bounds = self._wrap_bounds()
        self.wrapped_costf = self._wrap_fun(self.costf)
        self.wrapped_J = lambda p: self._wrap_J(p)
        self.p = self._init_p()

        # dynamics(x, u, t)
        # costf(x, u, t) -> tf if variable tf
        # customcon(fun(x, u, t, tf),lb, ub) -> tf if variable tf
        # an upperbound should be passed on tf if it is variable (or is assumed infinite)
        # lower and upper bounds for control should be passed or are assumed infinite
        # if the bounds passed are [lb,ub] they are extended to all dimensions of u, \
        # if they are more [[].[]] they have to be the right size
        # initial dynamics have be passed or are assumed 0
        # interally, everything has to be wrapped in terms of (p) (dynamics have to be solved)

    def _check_function_signature(self, func, expected_arg_count, func_name):
        sig = inspect.signature(func)
        arg_count = len(sig.parameters)
        if arg_count != expected_arg_count:
            raise ValueError(
                f"The function '{func_name}' must have exactly {expected_arg_count} arguments, got {arg_count}"
            )

    def _init_checks(self):
        # first check functions
        check = 3
        self._check_function_signature(self.dynamicsf, check, "dynamicsf")
        if not self.tf_is_variable:
            check = 2
        self._check_function_signature(self.costf, check, "costf")
        for constraint in self.constraints:
            self._check_function_signature(constraint.fun, check, "constraintf")

    def _from_p(self, p):
        class Answer:
            def __init__(self, tf_p=None, tf_b=None, u=None):
                self.tf_p = tf_p
                self.tf_b = tf_b
                self.u = u if u is not None else []

        ans = Answer()
        ans.tf_p = self.tf
        ans.tf_b = [self.tf] * self.u_sz
        # now 4 situations depend on the combination of n and tf adaptivity
        if self.n_is_adaptive:
            if self.tf_is_variable:  # [b, tf_b, tf_p], self.tf = None
                ans.tf_b = p[(self.u_sz) * (self.n_max + 1) : -1]
                ans.tf_p = p[-1]
            else:  # [b, tf_b], self.tf = Value
                ans.tf_b = p[(self.u_sz) * (self.n_max + 1) :]
        else:
            if self.tf_is_variable:  # [b, tf_p], self.tf = None
                ans.tf_p = p[-1]
                ans.tf_b = [ans.tf_p] * self.u_sz
            else:  # [b], self.tf = Value
                ans.tf_b = [self.tf] * self.u_sz

        # the first u_sz*(n_max+1) params are for u's
        # each u has n_max+1 parameter
        # ans.u = []
        # for i in range(self.u_sz):
        #     def b(t): return bernstein_poly(
        #         p[i*(self.n_max+1):(i+1)*(self.n_max+1)], t, ans.tf_b)
        #     ans.u.append(b)
        def b(t):
            return [
                bernstein_poly(
                    p[i * (self.n_max + 1) : (i + 1) * (self.n_max + 1)], t, ans.tf_b[i]
                )
                for i in range(self.u_sz)
            ]

        ans.u = b
        return ans

    # takes in a fun(x[i](t), u[j](t), tf (Optional)), returns fun(p)
    def _wrap_fun(self, fun, high_precision=False):
        if self.tf_is_variable:

            def fout(p):
                ipts = self._from_p(p)
                x = self._x_of_t(p, high_precision=high_precision)
                return fun(x, ipts.u, ipts.tf_p)

        else:

            def fout(p):
                ipts = self._from_p(p)
                x = self._x_of_t(p, high_precision=high_precision)
                return fun(x, ipts.u)

        return fout

    def _wrap_constraints(self):
        ans = []
        for constraint in self.constraints:
            ans.append(
                NonlinearConstraint(
                    self._wrap_fun(constraint.fun), constraint.lb, constraint.ub
                )
            )
        if self.n_is_adaptive:
            for i in range(self.u_sz):
                if self.tf_is_variable:
                    ans.append(
                        NonlinearConstraint(
                            lambda p: p[((self.n_max + 1) * self.u_sz) + i] - p[-1],
                            0,
                            np.inf,
                        )
                    )
                else:
                    ans.append(
                        NonlinearConstraint(
                            lambda p: p[((self.n_max + 1) * self.u_sz) + i] - self.tf,
                            0,
                            np.inf,
                        )
                    )

        return ans

    def _wrap_bounds(self):
        bounds = []
        for n in range((self.n_max + 1) * self.u_sz):
            bounds.append([self.u_bounds[0], self.u_bounds[1]])

        if self.n_is_adaptive:
            if self.tf_is_variable:  # [b, tf_b, tf_p], self.tf = None
                for i in range(self.u_sz):
                    bounds.append([0, self.tf_max])
                bounds.append([0, self.tf_max])
            else:
                for i in range(self.u_sz):
                    bounds.append([self.tf, self.tf + 100000])

        else:
            if self.tf_is_variable:  # [b, tf_p], self.tf = None
                bounds.append([0, self.tf_max])

        return bounds

    def _wrap_J(self, p):
        c = [self.wrapped_costf(p)]
        if self.n_is_adaptive:
            for i in range(self.u_sz):
                c.append(1 / p[((self.n_max + 1) * self.u_sz) + i])

        return np.linalg.norm(c)

    def _x_of_t(self, p, high_precision=False):
        ipts = self._from_p(p)
        if self.u_is_sigm_limited:

            def ivp(t, x):
                return self.dynamicsf(  # x, u, t -> u(t, tf_b)
                    x,
                    fun_sigmoid_limited(
                        ipts.u, self.u_sigm_bounds[0], self.u_sigm_bounds[1]
                    )(t),
                    t,
                )

        else:

            def ivp(t, x):
                return self.dynamicsf(x, ipts.u(t), t)  # x, u, t -> u(t, tf_b)

        if high_precision:
            return solve_ivp(
                ivp,
                [0, ipts.tf_p],
                self.initial_dynamics,
                dense_output=True,
                method="RK45",
                max_step=0.01,
                atol=1,
                rtol=1,
            ).sol
        else:
            return solve_ivp(
                ivp,
                [0, ipts.tf_p],
                self.initial_dynamics,
                dense_output=True,
                method="RK45",
                max_step=0.1,
                atol=1,
                rtol=1,
            ).sol

    def _init_p(self):
        if self.init_is_random:
            init_p = np.random.uniform(
                self.u_bounds[0], self.u_bounds[1], size=self.u_sz * (self.n_max + 1)
            ).tolist()
        else:
            init_p = [0] * self.u_sz * (self.n_max + 1)
        if self.n_is_adaptive:
            if self.tf_is_variable:  # [b, tf_b, tf_p], self.tf = None
                tf_b = []
                for i in range(self.u_sz):
                    tf_b.append(np.random.uniform(0, self.tf_max))
                init_p += tf_b
                init_p.append(self.tf)
            else:  # [b, tf_b], self.tf = Value
                for i in range(self.u_sz):
                    init_p.append(self.tf)
                    # init_p.append(np.random.uniform(self.tf, self.tf + 100)) # if random tf_b inits
        else:
            if self.tf_is_variable:  # [b, tf_p], self.tf = None
                # init_p.append(np.random.uniform(0, self.tf_max))
                init_p.append(self.tf)
        print(f"Init p:\n{init_p}")
        return init_p

    def minimize(self, method="trust-constr"):
        if method == "trust-constr":
            options = {"gtol": 1e-20, "verbose": 1}
        elif method == "SLSQP":
            options = {"ftol": 1e-12, "finite_diff_rel_step": "cs"}
        else:
            options = {}

        sol = minimize(
            self.wrapped_J,
            self.p,
            method=method,
            bounds=self.wrapped_bounds,
            constraints=self.wrapped_constraints,
            options=options,
        )
        self.p = sol.x
        # print(f'Constr violation: {sol.constr_violation}')
        wrapped_costf = self._wrap_fun(self.costf, high_precision=True)
        print(f"Cost: {wrapped_costf(self.p)}")
        return self.p

    def get_s_of_t(self, p):
        x = self._x_of_t(p)
        u = self._from_p(p).u

        if self.u_is_sigm_limited:
            u = fun_sigmoid_limited(u, self.u_sigm_bounds[0], self.u_sigm_bounds[1])

        s = lambda t: np.concatenate([x(t), u(t)])

        return x, u, s
