
Joint Distributions
===================

This is a brief introduction to working with Joint Distributions from the
`prob140` library.

.. contents:: Table of Contents
    :depth: 2
    :local:


Getting Started
---------------

As always, this should be the first cell if you are using a notebook.

.. code-block:: python

    # HIDDEN

    from datascience import *
    from prob140 import *
    %matplotlib inline
    import matplotlib.pyplot as plt
    import numpy as np
    plt.style.use('fivethirtyeight')



Constructing Joint Distributions
--------------------------------

Distribution basics
^^^^^^^^^^^^^^^^^^^

We can construct a joint distribution by starting with a Table. Calling
`Table().values()` with two lists will create a Table with X and Y taking on
those values.

.. ipython:: python

    from prob140 import *

    dist = Table().values(make_array(2, 3), np.arange(1, 6, 2))
    dist

We can then assign probabilities using `probability()` if we have an explicit
list or array.

.. ipython:: python

    dist = dist.probability([0.1, 0.1, 0.2, 0.3, 0.1, 0.2])
    dist

To turn it into a Joint Distribution object, call the `to_joint()` method.

.. ipython:: python

    dist.to_joint()

By default, the joint distribution will display the Y values in reverse. To turn
this functionality off, use the optional parameter `reverse=False`.

.. ipython:: python

    dist.to_joint(reverse=False)

Naming the Variables
^^^^^^^^^^^^^^^^^^^^

When defining a distribution, you can also give a name to each random variable
rather than the default 'X' and 'Y'. You must alternate between strings and
lists when calling `values()`

.. ipython:: python

    heads_table = Table().values("H1",[0.2,0.9],"H2",[2,1,0]).probability(make_array(.75*.04, .75*.32,.75*.64,.25*.81,.25*.18,.25*.01))
    heads_table
    heads = heads_table.to_joint(reverse=False)
    heads

You can also use strings for the values of the domain.

.. ipython:: python

    coins_table = Table().values("Coin1",['H','T'],"Coin2", ['H','T']).probability(np.array([0.24, 0.36, 0.16, 0.24]))
    coins = coins_table.to_joint(reverse=False)
    coins

Probability Functions
^^^^^^^^^^^^^^^^^^^^^

You can also use a joint probability function that will take in the values of
the random variables.

.. ipython:: python

    def joint_func(dice1, dice2):
        return (dice1 + dice2)/252

    dice = Table().values("D1", np.arange(1,7),"D2", np.arange(1,7)).probability_function(joint_func).to_joint()
    dice

Marginal Distributions
----------------------

To see the marginal distribution of a variable, call the method
`marginal(label)` where label is a string for the label.

.. ipython:: python

    heads.marginal("H1")
    heads.marginal("H2")
    coins.marginal("Coin1")

You can also call `both_marginals()` to see both marginal distributions at
once.

.. ipython:: python

    heads.both_marginals()
    coins.both_marginals()

To get the marginal distribution of a variable as a single variable distribution
for plotting, call `marginal_dist(label)`.

.. ipython:: python

    heads.marginal_dist("H1")

.. ipython:: python

    @savefig marginal_dist.png width=4in
    Plot(heads.marginal_dist("H1"), width=0.1)

.. ipython:: python

    heads.marginal_dist("H2")
    coins.marginal_dist("Coin1")


Conditional Distributions
-------------------------

You can see the conditional distribution using `conditional_dist(label, given)`.
For example, to see the distribution of H1|H2, call `conditional_dist("H1", "H2")`.

.. ipython:: python

    heads.conditional_dist("H1", "H2")
    heads.conditional_dist("H2", "H1")
