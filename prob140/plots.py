import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
from sympy import lambdify,symbols


def Plot_continuous(x_limits, func, *args, **kwargs):
    """
    Plots a continuous distribution

    Parameters
    ----------
    x_limits : iterable
        Array, list, or tuple of size 2, containing the lower and upper bound to be plotted
    func : Sympy expression, function, or str
        Univariate density function or a String for the scipy dist name
    args : floats (optional)
        Arguments of the distribution if func was a String.

    Optional Named Parameters
    -------------------------
    tails : boolean (optional)
        If True, left_end will shade from the lower bound up to left_end, and right_end will shade from right_end up to
        the upper bound. If False, left_end will shade to right_end. (Default: False)
    left_end : float (optional)
        Left side of event to be shaded (Default: None)
    right_end : float (optional)
        Right side of event to be shaded (Default: None)
    cdf : boolean (optional)
        If True and func was string, the cdf will be plotted (Default: False)

    All pyplot named arguments (such as color) should work as well. See http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.plot

    Returns
    -------
    None

    """

    interval = x_limits
    tails = kwargs.pop("tails", False)

    if isinstance(func, str):
        func = func.lower()

        rv = getattr(stats, func)

        cdf = kwargs.pop("cdf", False)

        if cdf:
            f = rv(*args).cdf
        else:
            f = rv(*args).pdf
    else:
        assert len(func.free_symbols) <= 1, "Must have exactly 1 variable"
        if len(func.free_symbols) == 0:
            new_symbol = symbols('x')
            replace_variables = [new_symbol]
        else:
            replace_variables = list(func.free_symbols)
        f = lambdify(replace_variables,func)

    lower = interval[0]
    upper = interval[1]

    x = np.linspace(lower, upper, 100)
    y = [f(xi) for xi in x]

    left = kwargs.pop("left_end", None)
    right = kwargs.pop("right_end", None)

    options = {"lw": 2, "color": "darkblue"}

    options.update(kwargs)

    plt.plot(x, y, **options)
    plt.ylim(0, max(y) * 1.1)

    if tails:
        if left and left >= lower:
            x2 = np.linspace(lower, left, 100)
            plt.fill_between(x2, f(x2), alpha=0.7, color="gold")

        if right and right <= upper:
            x2 = np.linspace(right, upper, 100)
            plt.fill_between(x2, f(x2), alpha=0.7, color="gold")
    else:
        lb = left if left else lower
        rb = right if right else upper
        if left or right:
            x2 = np.linspace(lb, rb, 100)
            plt.fill_between(x2, f(x2), alpha=0.7, color="gold")


def Plot_expon(x_limits, lamb, **kwargs):
    """
    Plots an exponential distribution

    Parameters
    ----------
    x_limits : iterable
        Array, list, or tuple of size 2, containing the lower and upper bound to be plotted
    lamb : float
        Rate

    Optional Named Parameters
    -------------------------
    tails : boolean (optional)
        If True, left_end will shade from the lower bound up to left_end, and right_end will shade from right_end up to
        the upper bound. If False, left_end will shade to right_end. (Default: False)
    left_end : float (optional)
        Left side of event to be shaded (Default: None)
    right_end : float (optional)
        Right side of event to be shaded (Default: None)
    cdf : boolean (optional)
        If True and func was string, the cdf will be plotted (Default: False)

    All pyplot named arguments (such as color) should work as well. See http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.plot

    """
    Plot_continuous(x_limits, "expon", 0, lamb, **kwargs)


def Plot_norm(x_limits, mu, sigma, **kwargs):
    Plot_continuous(x_limits, "norm", mu, sigma, **kwargs)


def Plot_arcsine(x_limits, **kwargs):
    Plot_continuous(x_limits, "arcsine", **kwargs)


def Plot_beta(x_limits, a, b, **kwargs):
    Plot_continuous(x_limits, "beta", a, b, **kwargs)


def Plot_cauchy(x_limits, **kwargs):
    Plot_continuous(x_limits, "cauchy", **kwargs)


def Plot_chi2(x_limits, df, **kwargs):
    Plot_continuous(x_limits, "chi2", df, **kwargs)


def Plot_erlang(x_limits, r, lamb, **kwargs):
    Plot_continuous(x_limits, "erlang", r, 0, 1 / lamb, **kwargs)


def Plot_f(x_limits, dfn, dfd, **kwargs):
    Plot_continuous(x_limits, "f", dfn, dfd, **kwargs)


def Plot_gamma(x_limits, r, lamb, **kwargs):
    Plot_continuous(x_limits, "gamma", r, 0, 1 / lamb, **kwargs)


def Plot_lognorm(x_limits, mu, sigma, **kwargs):
    Plot_continuous(x_limits, "lognorm", sigma, 0, np.exp(mu), **kwargs)


def Plot_pareto(x_limits, alpha, **kwargs):
    Plot_continuous(x_limits, "pareto", alpha, **kwargs)


def Plot_powerlaw(x_limits, a, **kwargs):
    Plot_continuous(x_limits, "powerlaw", a, **kwargs)


def Plot_rayleigh(x_limits, sigma, **kwargs):
    Plot_continuous(x_limits, "rayleigh", 0, sigma, **kwargs)


def Plot_t(x_limits, df, **kwargs):
    Plot_continuous(x_limits, "t", df, **kwargs)


def Plot_triang(x_limits, a, b, c, **kwargs):
    Plot_continuous(x_limits, "triang", (c - a) / (b - a), a, b - a, **kwargs)


def Plot_uniform(x_limits, a, b, **kwargs):
    Plot_continuous(x_limits, "uniform", a, b - a, **kwargs)