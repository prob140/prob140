import ipywidgets as widgets
from ipywidgets import interact
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.mplot3d.axes3d import Axes3D
import numpy as np
import scipy.stats as stats
from sympy import lambdify, symbols


def Plot_continuous(x_limits, func, *args, **kwargs):
    """
    Plots a continuous distribution

    Parameters
    ----------
    x_limits : iterable
        Array, list, or tuple of size 2, containing the lower and upper bound
        to be plotted.
    func : Sympy expression, function, or str
        Univariate density function or a String for the scipy dist name.
    args : floats (optional)
        Arguments of the distribution if func was a String.

    Optional Named Parameters
    -------------------------
    tails : boolean (optional)
        If True, left_end will shade from the lower bound up to left_end, and
        right_end will shade from right_end up to the upper bound. If False,
        left_end will shade to right_end. (Default: False)
    left_end : float (optional)
        Left side of event to be shaded. (Default: None)
    right_end : float (optional)
        Right side of event to be shaded. (Default: None)
    cdf : boolean (optional)
        If True and func was string, the cdf will be plotted (Default: False)

    All pyplot named arguments (such as color) should work as well. See
    http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.plot

    Returns
    -------
    None
    """

    interval = x_limits
    tails = kwargs.pop('tails', False)

    if isinstance(func, str):
        func = func.lower()

        rv = getattr(stats, func)

        cdf = kwargs.pop('cdf', False)

        if cdf:
            f = rv(*args).cdf
        else:
            f = rv(*args).pdf
    elif callable(func):
        f = func
    else:
        assert len(func.free_symbols) <= 1, 'Must have exactly 1 variable'
        if len(func.free_symbols) == 0:
            new_symbol = symbols('x')
            replace_variables = [new_symbol]
        else:
            replace_variables = list(func.free_symbols)
        f = lambdify(replace_variables, func)

    lower = interval[0]
    upper = interval[1]

    x = np.linspace(lower, upper, 100)
    y = [f(xi) for xi in x]

    left = kwargs.pop('left_end', None)
    right = kwargs.pop('right_end', None)

    options = {'lw': 2, 'color': 'darkblue'}

    options.update(kwargs)

    plt.plot(x, y, **options)
    plt.ylim(0, max(y) * 1.1)

    if tails:
        if left and left >= lower:
            x2 = np.linspace(lower, left, 100)
            plt.fill_between(x2, f(x2), alpha=0.7, color='gold')

        if right and right <= upper:
            x2 = np.linspace(right, upper, 100)
            plt.fill_between(x2, f(x2), alpha=0.7, color='gold')
    else:
        lb = left if left else lower
        rb = right if right else upper
        if left or right:
            x2 = np.linspace(lb, rb, 100)
            plt.fill_between(x2, f(x2), alpha=0.7, color='gold')

    # Multiple all y-ticks by 100 to get percent per unit.
    ax = plt.gca()
    ticks = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x * 100.))
    ax.yaxis.set_major_formatter(ticks)
    plt.ylabel('Percent per unit')


def Plot_3d(x_limits, y_limits, f, interactive=False, figsize=(12, 8), **kwargs):
    """
    Plots a 3d graph.

    Parameters
    ----------
    x_limits : iterable
        Array, list, or tuple of size 2, containing the lower and upper bound
        of the x-axis.
    y_limits : iterable
        Array, list, or tuple of size 2, containing the lower and upper bound
        of the x-axis.
    f : bivariate function
        Joint density
    interactive : boolean (optional)
        If True, creates a widget to adjust elevation and azimuth.
        (default: False)
    kwargs
        Optional named arguments for `plot_surface`.

    Returns
    -------
    None
    """
    def plot(elev, azim):

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        x = np.linspace(*x_limits, 100)
        y = np.linspace(*y_limits, 100)
        X, Y = np.meshgrid(x, y)

        v = np.vectorize(f)
        Z = v(X, Y)
        ax.plot_surface(X, Y, Z, **kwargs)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('f(x, y)')

        ax.view_init(elev, azim)

    if interactive:
        elevation_slider = widgets.FloatSlider(
            value=20,
            min=0,
            max=90,
            step=1,
            description='elevation'
        )
        azimuth_slider = widgets.FloatSlider(
            value=-100,
            min=-180,
            max=180,
            step=1,
            description='azimuth'
        )

        @interact(elev=elevation_slider, azim=azimuth_slider)
        def wrapper(elev, azim):
            plot(elev, azim)
    else:
        plot(20, -100)


def Plot_expon(x_limits, lamb, **kwargs):
    """
    Plots an exponential distribution

    Parameters
    ----------
    x_limits : iterable
        Array, list, or tuple of size 2, containing the lower and upper bound
        to be plotted.
    lamb : float
        Rate.

    Optional Named Parameters
    -------------------------
    tails : boolean (optional)
        If True, left_end will shade from the lower bound up to left_end, and
        right_end will shade from right_end up to the upper bound. If False,
        left_end will shade to right_end. (Default: False)
    left_end : float (optional)
        Left side of event to be shaded. (Default: None)
    right_end : float (optional)
        Right side of event to be shaded. (Default: None)
    cdf : boolean (optional)
        If True and func was string, the cdf will be plotted (Default: False)

    All pyplot named arguments (such as color) should work as well. See
    http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.plot

    Returns
    -------
    None
    """
    Plot_continuous(x_limits, 'expon', 0, 1/lamb, **kwargs)


def Plot_norm(x_limits, mu, sigma, **kwargs):
    """
    Plots a gaussian distribution.

    Parameters
    ----------
    x_limits : iterable
        Array, list, or tuple of size 2, containing the lower and upper bound
        to be plotted.
    mu : float
        Mean.
    sigma : float
        Standard Deviation.

    Optional Named Parameters
    -------------------------
    tails : boolean (optional)
        If True, left_end will shade from the lower bound up to left_end, and
        right_end will shade from right_end up to the upper bound. If False,
        left_end will shade to right_end. (Default: False)
    left_end : float (optional)
        Left side of event to be shaded. (Default: None)
    right_end : float (optional)
        Right side of event to be shaded. (Default: None)
    cdf : boolean (optional)
        If True and func was string, the cdf will be plotted (Default: False)

    All pyplot named arguments (such as color) should work as well. See
    http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.plot

    Returns
    -------
    None
    """
    Plot_continuous(x_limits, 'norm', mu, sigma, **kwargs)


def Plot_arcsine(x_limits, **kwargs):
    """
    Plots an arcsine distribution.

    Parameters
    ----------
    x_limits : iterable
        Array, list, or tuple of size 2, containing the lower and upper bound
        to be plotted.

    Optional Named Parameters
    -------------------------
    tails : boolean (optional)
        If True, left_end will shade from the lower bound up to left_end, and
        right_end will shade from right_end up to the upper bound. If False,
        left_end will shade to right_end. (Default: False)
    left_end : float (optional)
        Left side of event to be shaded. (Default: None)
    right_end : float (optional)
        Right side of event to be shaded. (Default: None)
    cdf : boolean (optional)
        If True and func was string, the cdf will be plotted (Default: False)

    All pyplot named arguments (such as color) should work as well. See
    http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.plot

    Returns
    -------
    None
    """
    Plot_continuous(x_limits, 'arcsine', **kwargs)


def Plot_beta(x_limits, a, b, **kwargs):
    """
    Plots a beta distribution.

    Parameters
    ----------
    x_limits : iterable
        Array, list, or tuple of size 2, containing the lower and upper bound
        to be plotted.
    a : float
        Shape.
    b : float
        Shape.

    Optional Named Parameters
    -------------------------
    tails : boolean (optional)
        If True, left_end will shade from the lower bound up to left_end, and
        right_end will shade from right_end up to the upper bound. If False,
        left_end will shade to right_end. (Default: False)
    left_end : float (optional)
        Left side of event to be shaded. (Default: None)
    right_end : float (optional)
        Right side of event to be shaded. (Default: None)
    cdf : boolean (optional)
        If True and func was string, the cdf will be plotted (Default: False)

    All pyplot named arguments (such as color) should work as well. See
    http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.plot

    Returns
    -------
    None
    """
    Plot_continuous(x_limits, 'beta', a, b, **kwargs)


def Plot_cauchy(x_limits, loc=0, scale=1, **kwargs):
    """
    Plots a cauchy distribution.

    Parameters
    ----------
    x_limits : iterable
        Array, list, or tuple of size 2, containing the lower and upper bound
        to be plotted.

    Optional Named Parameters
    -------------------------
    tails : boolean (optional)
        If True, left_end will shade from the lower bound up to left_end, and
        right_end will shade from right_end up to the upper bound. If False,
        left_end will shade to right_end. (Default: False)
    left_end : float (optional)
        Left side of event to be shaded. (Default: None)
    right_end : float (optional)
        Right side of event to be shaded. (Default: None)
    cdf : boolean (optional)
        If True and func was string, the cdf will be plotted (Default: False)

    All pyplot named arguments (such as color) should work as well. See
    http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.plot

    Returns
    -------
    None
    """
    Plot_continuous(x_limits, 'cauchy', loc, scale, **kwargs)


def Plot_chi2(x_limits, df, **kwargs):
    """
    Plots a chi-squared distribution.

    Parameters
    ----------
    x_limits : iterable
        Array, list, or tuple of size 2, containing the lower and upper bound
        to be plotted.
    df : Integer
        Number of degrees of freedom.

    Optional Named Parameters
    -------------------------
    tails : boolean (optional)
        If True, left_end will shade from the lower bound up to left_end, and
        right_end will shade from right_end up to the upper bound. If False,
        left_end will shade to right_end. (Default: False)
    left_end : float (optional)
        Left side of event to be shaded. (Default: None)
    right_end : float (optional)
        Right side of event to be shaded. (Default: None)
    cdf : boolean (optional)
        If True and func was string, the cdf will be plotted (Default: False)

    All pyplot named arguments (such as color) should work as well. See
    http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.plot

    Returns
    -------
    None
    """
    Plot_continuous(x_limits, 'chi2', df, **kwargs)


def Plot_erlang(x_limits, r, lamb, **kwargs):
    """
    Plots an erlang distribution.

    Parameters
    ----------
    x_limits : iterable
        Array, list, or tuple of size 2, containing the lower and upper bound
        to be plotted.
    r : int
        Shape.
    lamb : float
        Rate.

    Optional Named Parameters
    -------------------------
    tails : boolean (optional)
        If True, left_end will shade from the lower bound up to left_end, and
        right_end will shade from right_end up to the upper bound. If False,
        left_end will shade to right_end. (Default: False)
    left_end : float (optional)
        Left side of event to be shaded. (Default: None)
    right_end : float (optional)
        Right side of event to be shaded. (Default: None)
    cdf : boolean (optional)
        If True and func was string, the cdf will be plotted (Default: False)

    All pyplot named arguments (such as color) should work as well. See
    http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.plot

    Returns
    -------
    None
    """
    Plot_continuous(x_limits, 'erlang', r, 0, 1 / lamb, **kwargs)


def Plot_f(x_limits, dfn, dfd, **kwargs):
    """
    Plots an F distribution.

    Parameters
    ----------
    x_limits : iterable
        Array, list, or tuple of size 2, containing the lower and upper bound
        to be plotted.
    dfn : int
        Degree of freedom 1.
    dfd : int
        Degree of freedom 2.

    Optional Named Parameters
    -------------------------
    tails : boolean (optional)
        If True, left_end will shade from the lower bound up to left_end, and
        right_end will shade from right_end up to the upper bound. If False,
        left_end will shade to right_end. (Default: False)
    left_end : float (optional)
        Left side of event to be shaded. (Default: None)
    right_end : float (optional)
        Right side of event to be shaded. (Default: None)
    cdf : boolean (optional)
        If True and func was string, the cdf will be plotted (Default: False)

    All pyplot named arguments (such as color) should work as well. See
    http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.plot

    Returns
    -------
    None
    """
    Plot_continuous(x_limits, 'f', dfn, dfd, **kwargs)


def Plot_gamma(x_limits, r, lamb, **kwargs):
    """
    Plots a gamma distribution.

    Parameters
    ----------
    x_limits : iterable
        Array, list, or tuple of size 2, containing the lower and upper bound
        to be plotted.
    r : int
        Shape.
    lamb : float
        Rate.

    Optional Named Parameters
    -------------------------
    tails : boolean (optional)
        If True, left_end will shade from the lower bound up to left_end, and
        right_end will shade from right_end up to the upper bound. If False,
        left_end will shade to right_end. (Default: False)
    left_end : float (optional)
        Left side of event to be shaded. (Default: None)
    right_end : float (optional)
        Right side of event to be shaded. (Default: None)
    cdf : boolean (optional)
        If True and func was string, the cdf will be plotted (Default: False)

    All pyplot named arguments (such as color) should work as well. See
    http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.plot

    Returns
    -------
    None
    """
    Plot_continuous(x_limits, 'gamma', r, 0, 1 / lamb, **kwargs)


def Plot_lognorm(x_limits, mu, sigma, **kwargs):
    """
    Plots a log-normal distribution.

    Parameters
    ----------
    x_limits : iterable
        Array, list, or tuple of size 2, containing the lower and upper bound
        to be plotted.
    mu : float
        Mean.
    sigma : float
        Standard Deviation.

    Optional Named Parameters
    -------------------------
    tails : boolean (optional)
        If True, left_end will shade from the lower bound up to left_end, and
        right_end will shade from right_end up to the upper bound. If False,
        left_end will shade to right_end. (Default: False)
    left_end : float (optional)
        Left side of event to be shaded. (Default: None)
    right_end : float (optional)
        Right side of event to be shaded. (Default: None)
    cdf : boolean (optional)
        If True and func was string, the cdf will be plotted (Default: False)

    All pyplot named arguments (such as color) should work as well. See
    http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.plot

    Returns
    -------
    None
    """
    Plot_continuous(x_limits, 'lognorm', sigma, 0, np.exp(mu), **kwargs)


def Plot_pareto(x_limits, alpha, **kwargs):
    """
    Plots an alpha distribution.

    Parameters
    ----------
    x_limits : iterable
        Array, list, or tuple of size 2, containing the lower and upper bound
        to be plotted.
    a : float
        Shape.

    Optional Named Parameters
    -------------------------
    tails : boolean (optional)
        If True, left_end will shade from the lower bound up to left_end, and
        right_end will shade from right_end up to the upper bound. If False,
        left_end will shade to right_end. (Default: False)
    left_end : float (optional)
        Left side of event to be shaded. (Default: None)
    right_end : float (optional)
        Right side of event to be shaded. (Default: None)
    cdf : boolean (optional)
        If True and func was string, the cdf will be plotted (Default: False)

    All pyplot named arguments (such as color) should work as well. See
    http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.plot

    Returns
    -------
    None
    """
    Plot_continuous(x_limits, 'pareto', alpha, **kwargs)


def Plot_powerlaw(x_limits, a, **kwargs):
    """
    Plots a powerlaw distribution.

    Parameters
    ----------
    x_limits : iterable
        Array, list, or tuple of size 2, containing the lower and upper bound
        to be plotted.
    a : float
        Shape.

    Optional Named Parameters
    -------------------------
    tails : boolean (optional)
        If True, left_end will shade from the lower bound up to left_end, and
        right_end will shade from right_end up to the upper bound. If False,
        left_end will shade to right_end. (Default: False)
    left_end : float (optional)
        Left side of event to be shaded. (Default: None)
    right_end : float (optional)
        Right side of event to be shaded. (Default: None)
    cdf : boolean (optional)
        If True and func was string, the cdf will be plotted (Default: False)

    All pyplot named arguments (such as color) should work as well. See
    http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.plot

    Returns
    -------
    None
    """
    Plot_continuous(x_limits, 'powerlaw', a, **kwargs)


def Plot_rayleigh(x_limits, sigma, **kwargs):
    """
    Plots a rayleigh distribution.

    Parameters
    ----------
    x_limits : iterable
        Array, list, or tuple of size 2, containing the lower and upper bound
        to be plotted.
    sigma : float
        Scale.

    Optional Named Parameters
    -------------------------
    tails : boolean (optional)
        If True, left_end will shade from the lower bound up to left_end, and
        right_end will shade from right_end up to the upper bound. If False,
        left_end will shade to right_end. (Default: False)
    left_end : float (optional)
        Left side of event to be shaded. (Default: None)
    right_end : float (optional)
        Right side of event to be shaded. (Default: None)
    cdf : boolean (optional)
        If True and func was string, the cdf will be plotted (Default: False)

    All pyplot named arguments (such as color) should work as well. See
    http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.plot

    Returns
    -------
    None
    """
    Plot_continuous(x_limits, 'rayleigh', 0, sigma, **kwargs)


def Plot_t(x_limits, df, **kwargs):
    """
    Plots a t distribution.

    Parameters
    ----------
    x_limits : iterable
        Array, list, or tuple of size 2, containing the lower and upper bound
        to be plotted.
    df : int
        Degree of freedom

    Optional Named Parameters
    -------------------------
    tails : boolean (optional)
        If True, left_end will shade from the lower bound up to left_end, and
        right_end will shade from right_end up to the upper bound. If False,
        left_end will shade to right_end. (Default: False)
    left_end : float (optional)
        Left side of event to be shaded. (Default: None)
    right_end : float (optional)
        Right side of event to be shaded. (Default: None)
    cdf : boolean (optional)
        If True and func was string, the cdf will be plotted (Default: False)

    All pyplot named arguments (such as color) should work as well. See
    http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.plot

    Returns
    -------
    None
    """
    Plot_continuous(x_limits, 't', df, **kwargs)


def Plot_triang(x_limits, a, b, c, **kwargs):
    """
    Plots a triangular distribution.

    Parameters
    ----------
    x_limits : iterable
        Array, list, or tuple of size 2, containing the lower and upper bound
        to be plotted.
    a : float
        Minimum value.
    b : float
        Maximum value.
    c : float
        Intermediate value

    Optional Named Parameters
    -------------------------
    tails : boolean (optional)
        If True, left_end will shade from the lower bound up to left_end, and
        right_end will shade from right_end up to the upper bound. If False,
        left_end will shade to right_end. (Default: False)
    left_end : float (optional)
        Left side of event to be shaded. (Default: None)
    right_end : float (optional)
        Right side of event to be shaded. (Default: None)
    cdf : boolean (optional)
        If True and func was string, the cdf will be plotted (Default: False)

    All pyplot named arguments (such as color) should work as well. See
    http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.plot

    Returns
    -------
    None
    """
    Plot_continuous(x_limits, 'triang', (c - a) / (b - a), a, b - a, **kwargs)


def Plot_uniform(x_limits, a, b, **kwargs):
    """
    Plots a uniform distribution.

    Parameters
    ----------
    x_limits : iterable
        Array, list, or tuple of size 2, containing the lower and upper bound
        to be plotted.
    a : float
        Minimum value.
    b : float
        Maximum value.

    Optional Named Parameters
    -------------------------
    tails : boolean (optional)
        If True, left_end will shade from the lower bound up to left_end, and
        right_end will shade from right_end up to the upper bound. If False,
        left_end will shade to right_end. (Default: False)
    left_end : float (optional)
        Left side of event to be shaded. (Default: None)
    right_end : float (optional)
        Right side of event to be shaded. (Default: None)
    cdf : boolean (optional)
        If True and func was string, the cdf will be plotted (Default: False)

    All pyplot named arguments (such as color) should work as well. See
    http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.plot

    Returns
    -------
    None
    """
    Plot_continuous(x_limits, 'uniform', a, b - a, **kwargs)
