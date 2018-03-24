
Plots for Continuous Distributions
==================================

.. currentmodule:: prob140.plots



Importing
---------

To use the continuous plots, you must import the plots module. `Plot_continuous` and `Plot_norm` are the only
functions that don't require you to import the plots module.

.. ipython:: python

    from prob140.plots import *

Quick Reference
---------------

The normal syntax for plotting a distribution is `Plot_distribution(x_limits, parameters, optional_arguments)`

Click the links below to see detailed information for plotting any distribution. Note that we won't use most of these for Prob140


.. autosummary::
    :toctree: _autosummary

    Plot_norm
    Plot_arcsine
    Plot_beta
    Plot_cauchy
    Plot_chi2
    Plot_erlang
    Plot_expon
    Plot_f
    Plot_gamma
    Plot_lognorm
    Plot_pareto
    Plot_powerlaw
    Plot_rayleigh
    Plot_t
    Plot_triang
    Plot_uniform
    Plot_continuous

Plotting events
---------------

The optional parameters `left_end=` and `right_end=` define the left and right side to be shaded. These optional
parameters should work for all the continuous distribution plots

.. ipython:: python

    @savefig norm_left_end.png width=3in
    Plot_norm(x_limits=(-2, 2), mu=0, sigma=1, left_end=-1)

.. ipython:: python

    @savefig norm_right_end.png width=3in
    Plot_norm(x_limits=(-2, 2), mu=0, sigma=1, right_end=1)

.. ipython:: python

    @savefig norm_left_right_end.png width=3in
    Plot_norm(x_limits=(-2, 2), mu=0, sigma=1, left_end=-1, right_end=1)

We can also set the parameter `tails=True` to invert the direction to be shaded.

.. ipython:: python

    @savefig norm_left_end_tails.png width=3in
    Plot_norm(x_limits=(-2, 2), mu=0, sigma=1, left_end=-1, right_end=1, tails=True)

CDF
---

For all the plot functions except `Plot_continuous`, you can pass the parameter `cdf=True` to plot the cumulative
distribution function instead of the probability density function. This also works with `left_end/right_end`

.. ipython:: python

    @savefig norm_cdf.png width=3in
    Plot_norm(x_limits=(-2, 2), mu=0, sigma=1, cdf=True)

Plot Examples
-------------

Plot_norm
^^^^^^^^^

.. ipython:: python

    @savefig norm.png width=3in
    Plot_norm(x_limits=(-2, 2), mu=0, sigma=1)

Plot_arcsine
^^^^^^^^^^^^

.. ipython:: python

    @savefig arcsine.png width=3in
    Plot_arcsine(x_limits=(0.01, 0.99))

Plot_beta
^^^^^^^^^

.. ipython:: python

    @savefig beta.png width=3in
    Plot_beta(x_limits=(0, 1), a=2, b=2)

Plot_cauchy
^^^^^^^^^^^

.. ipython:: python

    @savefig cauchy.png width=3in
    Plot_cauchy(x_limits=(-5, 5))

Plot_chi2
^^^^^^^^^

.. ipython:: python

    @savefig chi2.png width=3in
    Plot_chi2(x_limits=(0, 8), df=3)


Plot_erlang
^^^^^^^^^^^

.. ipython:: python

    @savefig erlang.png width=3in
    Plot_erlang(x_limits=(0, 12), r=3, lamb=0.5)

Plot_expon
^^^^^^^^^^

.. ipython:: python

    @savefig expon.png width=3in
    Plot_expon(x_limits=(0, 5), lamb=1)

Plot_f
^^^^^^

.. ipython:: python

    @savefig f.png width=3in
    Plot_f(x_limits=(0.01, 5), dfn=5, dfd=2)

Plot_gamma
^^^^^^^^^^

.. ipython:: python

    @savefig gamma.png width=3in
    Plot_gamma(x_limits=(0, 20), r=5, lamb=0.5)

Plot_lognorm
^^^^^^^^^^^^

.. ipython:: python

    @savefig lognorm.png width=3in
    Plot_lognorm(x_limits=(0, 5), mu=0, sigma=0.25)


Plot_rayleigh
^^^^^^^^^^^^^

.. ipython:: python

    @savefig rayleigh.png width=3in
    Plot_rayleigh(x_limits=(0, 10), sigma=2)

Plot_pareto
^^^^^^^^^^^

.. ipython:: python

    @savefig pareto.png width=3in
    Plot_pareto(x_limits=(0, 5), alpha=3)

Plot_powerlaw
^^^^^^^^^^^^^

.. ipython:: python

    @savefig powerlaw.png width=3in
    Plot_powerlaw(x_limits=(0, 1), a=1.6)

Plot_t
^^^^^^

.. ipython:: python

    @savefig t.png width=3in
    Plot_t(x_limits=(-3, 3), df=2)


Plot_triang
^^^^^^^^^^^

.. ipython:: python

    @savefig triang.png width=3in
    Plot_triang(x_limits=(0, 10), a=2, b=10, c=3)


Plot_uniform
^^^^^^^^^^^^

.. ipython:: python

    @savefig uniform.png width=3in
    Plot_uniform(x_limits=(0, 5), a=2, b=4)