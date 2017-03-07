import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt


def Plot_continuous(func, *args, **kwargs):

    if "x_limits" in kwargs:
        interval = kwargs.pop("x_limits")
    else:
        interval = args[0]
        args = args[1:]

    if isinstance(func, str):
        func = func.lower()

        rv = getattr(stats, func)

        cdf = kwargs.pop("cdf", False)
        tails = kwargs.pop("tails", False)

        if cdf:
            f = rv(*args).cdf
        else:
            f = rv(*args).pdf
    else:
        f = func

    lower = interval[0]
    upper = interval[1]

    x = np.linspace(lower, upper, 100)
    y = f(x)

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
    Plot_continuous("expon", 0, lamb, x_limits=x_limits, **kwargs)


def Plot_norm(x_limits, mu, sigma, **kwargs):
    Plot_continuous("norm", mu, sigma, x_limits=x_limits, **kwargs)


def Plot_arcsine(x_limits, **kwargs):
    Plot_continuous("arcsine", x_limits=x_limits, **kwargs)


def Plot_beta(x_limits, a, b, **kwargs):
    Plot_continuous("beta", a, b, x_limits=x_limits, **kwargs)


def Plot_cauchy(x_limits, **kwargs):
    Plot_continuous("cauchy", x_limits=x_limits, **kwargs)


def Plot_chi2(x_limits, df, **kwargs):
    Plot_continuous("chi2", df, x_limits=x_limits, **kwargs)


def Plot_erlang(x_limits, r, lamb, **kwargs):
    Plot_continuous("erlang", r, 0, 1 / lamb, x_limits=x_limits, **kwargs)


def Plot_f(x_limits, dfn, dfd, **kwargs):
    Plot_continuous("f", dfn, dfd, x_limits=x_limits, **kwargs)


def Plot_gamma(x_limits, r, lamb, **kwargs):
    Plot_continuous("gamma", r, 0, 1 / lamb, x_limits=x_limits, **kwargs)


def Plot_lognorm(x_limits, mu, sigma, **kwargs):
    Plot_continuous("lognorm", sigma, 0, np.exp(mu), x_limits=x_limits, **kwargs)


def Plot_pareto(x_limits, alpha, **kwargs):
    Plot_continuous("pareto", alpha, x_limits=x_limits, **kwargs)


def Plot_powerlaw(x_limits, a, **kwargs):
    Plot_continuous("powerlaw", a, x_limits=x_limits, **kwargs)


def Plot_rayleigh(x_limits, sigma, **kwargs):
    Plot_continuous("rayleigh", 0, sigma, x_limits=x_limits, **kwargs)


def Plot_t(x_limits, df, **kwargs):
    Plot_continuous("t", df, x_limits=x_limits, **kwargs)


def Plot_triang(x_limits, a, b, c, **kwargs):
    Plot_continuous("triang", (c - a) / (b - a), a, b - a, x_limits=x_limits, **kwargs)


def Plot_uniform(x_limits, a, b, **kwargs):
    Plot_continuous("uniform", a, b - a, x_limits=x_limits, **kwargs)