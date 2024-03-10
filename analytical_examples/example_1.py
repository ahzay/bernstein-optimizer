import numpy as np
from scipy.integrate import quad


from bernstein_optimizer import Constraint

u_sz = 2
x_sz = 2
tf = 1


def xdot(x, u, t):
    return [x[1] + u[0], -x[0] + u[1]]


def final_con_fun(x, u):
    return x(tf)[0] - x(tf)[1]


def costf(x, u):
    return (
        x(tf)[0] ** 2
        + x(tf)[1] ** 2
        + quad(lambda t: u(t)[0] ** 2 + u(t)[1] ** 2, 0, tf)[0]
    )


def analytical_s(t):
    sigma_3 = np.sin(0.5 * t) ** 2
    sigma_4 = 1.0 * np.sin(t) * 1j
    sigma_1 = -2.0 * sigma_3 + sigma_4 + 1.0
    sigma_2 = 2.0 * sigma_3 + sigma_4 - 1.0

    result = np.array(
        [
            0.99999823221123018690548178710742 * np.real(np.sin(t))
            - 2.0000037137312576263070695858914 * np.real(sigma_3)
            + np.real(
                t
                * sigma_1
                * (
                    -0.54068074149015865614842368813697
                    + 0.31335588255501350962717310721928 * 1j
                )
            )
            + np.real(
                t
                * sigma_2
                * (
                    0.54068074149015865614842368813697
                    + 0.31335588255501350962717310721928 * 1j
                )
            )
            + 1.0000018568656288131535347929457,
            0.99999823221123018690548178710742
            - 1.0000018568656288131535347929457 * np.real(np.sin(t))
            + np.real(
                t
                * sigma_1
                * (
                    -0.31335588255501350962717310721928
                    - 0.54068074149015865614842368813697 * 1j
                )
            )
            + np.real(
                t
                * sigma_2
                * (
                    0.31335588255501350962717310721928
                    - 0.54068074149015865614842368813697 * 1j
                )
            )
            - 1.9999964644224603738109635742148 * np.real(sigma_3),
            2.1627229659606346245936947525479 * np.real(sigma_3)
            - 0.62671176511002701925434621443856 * np.real(np.sin(t))
            - 1.0813614829803173122968473762739,
            1.2534235302200540385086924288771 * np.real(sigma_3)
            + 1.0813614829803173122968473762739 * np.real(np.sin(t))
            - 0.62671176511002701925434621443856,
        ]
    )

    return result


constraints = [Constraint(final_con_fun, 0, 0)]

initial_dynamics = [1, 1]
