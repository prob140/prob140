``prob140`` Tutorial!
=====================

This is a brief introduction to the functionality in ``prob140``! For an
interactive guide, see the examples notebook in the GitLab directory.

.. contents:: Table of Contents
    :depth: 2
    :local:


Getting Started
---------------

Make sure you are on the most recent version of the `prob140` library. See the installation guide for more directions.

If you are using an `iPython` notebook, use this as your first cell:

.. code-block:: python

    # HIDDEN

    from datascience import *
    from prob140 import *
    %matplotlib inline
    import matplotlib.pyplot as plt
    import numpy as np
    plt.style.use('fivethirtyeight')

You may want to familiarize yourself with Data8's ``datascience`` `documentation <http://data8.org/datascience/tutorial.html>`_ first

Creating a Distribution
-----------------------

The `prob140` library adds distribution methods to the default `table` class that you should
already be familiar with. A distribution is defined as a 2-column table in which the first column
represents the domain of the distribution while the second column represents the probabilities
associated with each value in the domain.

You can specify a list or array to the methods `domain` and `probability` to specify those columns
for a distribution

.. ipython:: python

    from prob140 import *

    dist1 = Table().domain(make_array(2, 3, 4)).probability(make_array(0.25, 0.5, 0.25))

    dist1

We can also construct a distribution by explicitly assigning values for the
`domain` but applying a probability function to the values of the domain

.. ipython:: python

    def p(x):
        return 0.25

    dist2 = Table().domain(np.arange(1, 8, 2)).probability_function(p)

    dist2

This can be very useful when we have a distribution with a known probability
density function

.. ipython:: python

    from scipy.misc import comb
    def pmf(x):
        n = 10
        p = 0.3
        return comb(n,x) * p**x * (1-p)**(n-x)

    binomial = Table().domain(np.arange(11)).probability_function(pmf)
    binomial


Events
------

Often, we are concerned with specific values in a distribution rather than all the values.

Calling ``event`` allows us to see a subset of the values in a distribution and the associated probabilities

.. ipython:: python

    dist1

    dist1.event(np.arange(1,4))

    dist2

    dist2.event([1, 3, 3.5, 6])

To find the probability of an event, we can call ``prob_event``, which sums up the probabilities
of each of the values

.. ipython:: python

    dist1.prob_event(np.arange(1,4))

    dist2.prob_event([1, 3, 3.5, 6])

    binomial.prob_event(np.arange(5))

    binomial.prob_event(np.arange(11))

Note that due to the way Python handles floats, there might be some rounding errors

Plotting
--------

To visualize our distributions, we can plot a histogram of the density using the ``Plot`` function.

.. ipython:: python

    @savefig binomial.png width=4in
    Plot(binomial)

.. ipython:: python

    @savefig dist2.png width=4in
    Plot(dist2)

Width
^^^^^

If want to specify the width of every bar, we can use the optional parameter ``width=`` to specify the bin sizes.
However, this should be used very rarely, **only** when there is uniform spacing between bars.

.. ipython:: python

    @savefig binomial_width_2.png width=4in
    Plot(binomial, width=2)

.. ipython:: python

    dist3 = Table().domain(np.arange(0, 10, 2)).probability_function(lambda x: 0.2)

    @savefig dist3.png width=4in
    Plot(dist3)

.. ipython:: python

    @savefig dist3_width_2.png width=4in
    Plot(dist3, width=2)

Events
^^^^^^

Sometimes, we want to highlight an event or events in our histogram. Do make an event a different color, we can use
the optional parameter ``event=``. An event must be a list or a list of lists.

.. ipython:: python

    @savefig binomial_event_1.png width=4in
    Plot(binomial, event=[1,3,5])

.. ipython:: python

    @savefig binomial_event_2.png width=4in
    Plot(binomial, event=np.arange(0,10,2))

If we use a list of lists for the event parameter, each event will be a different color.

.. ipython:: python

    @savefig binomial_event_3.png width=4in
    Plot(binomial, event=[[0],[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]])


Plotting multiple distributions
-------------------------------

It is often useful to plot multiple histograms on top of each other. To plot multiple distributions on the same
graph, use the ``Plots`` function. ``Plots`` takes in an even number of arguments, alternating between the label of
the distribution and the distribution table itself.

.. ipython:: python

    @savefig dist1_binomial.png width=4in
    Plots("Distribution 1", dist1, "Distribution 2", dist2)

.. ipython:: python

    binomial2 = Table().domain(np.arange(11)).probability_function(lambda x: comb(10,x) * 0.5**10)

    @savefig 2_binomials.png width=4in
    Plots("Bin(n=10,p=0.3)", binomial, "Bin(n=10,p=0.5)", binomial2)

Try to avoid plotting too many distributions together because the graph starts to become unreadable

.. ipython:: python

    @savefig bad_idea.png width=4in
    Plots("dist1", dist1, "dist2", dist2, "Bin1", binomial, "Bin2", binomial2)


Empirical Distributions
-----------------------

Whenever we simulate an event, we often end up with an array of results. We can construct an empirical distribution
of the results by grouping of the possible values and assigning the frequencies are probabilities. An easy way to do
this is by calling `emp_dist`

.. ipython:: python

    x = make_array(1,1,1,1,1,2,3,3,3,4)
    emp_dist(x)
    values = make_array()
    for i in range(10000):
        num = np.random.randint(10) + np.random.randint(10) + np.random.randint(10) + np.random.randint(10)
        values = np.append(values, num)

.. ipython:: python

    @savefig emp_dist.png width=4in
    Plot(emp_dist(values))



Utilities
---------

.. ipython:: python

    print(dist1.expected_value())
    print(dist1.sd())
    print(binomial.expected_value())
    print(0.3 * 10)
    print(binomial.sd())
    import math
    print(math.sqrt(10 * 0.3 * 0.7))
    print(binomial.variance())
    print(10 * 0.3 * 0.7)


