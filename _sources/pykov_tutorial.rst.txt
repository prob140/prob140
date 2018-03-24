Markov Chains
=============

This is a brief introduction to working with Markov Chains from the `prob140`
library.

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



Constructing Markov Chains
--------------------------

Explicitly assigning probabilities
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To assign the possible states of a markov chain, use `Table().states()`.

.. ipython:: python

    Table().states(make_array("A", "B"))

A markov chain needs transition probabilities for each transition state `i` to
`j`. Note that the sum of the transition probabilities coming out of each state
must sum to 1

.. ipython:: python

    mc_table = Table().states(make_array("A", "B")).transition_probability(make_array(0.5, 0.5, 0.3, 0.7))
    mc_table

To convert the Table into a MarkovChain object, call `.to_markov_chain()`.

.. ipython:: python

    mc = mc_table.to_markov_chain()
    mc

Using a transition probability function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Just like single variable distributions and joint distributions, we can assign a
transition probability function.

.. ipython:: python

    def identity_transition(x,y):
        if x==y:
            return 1
        return 0

    transMatrix = Table().states(np.arange(1,4)).transition_function(identity_transition)
    transMatrix
    mc2 = transMatrix.to_markov_chain()
    mc2


Distribution
------------

To find the state of the markov chain after a certain point, we can call the
`.distribution` method which takes in a starting condition and a number of
steps. For example, to see the distribution of `mc` starting at "A" after 2
steps, we can call

.. ipython:: python

    mc.distribution("A", 2)

Sometimes it might be useful for the starting condition to be a probability
distribution. We can set the starting condition to be a single variable
distribution.

.. ipython:: python

    start = Table().states(make_array("A", "B")).probability(make_array(0.8, 0.2))
    start
    mc.distribution(start, 2)
    mc.distribution(start, 0)

Steady State
------------

.. ipython:: python

    mc.steady_state()
    mc2.steady_state()
