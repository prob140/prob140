Markov Chains
=============

This is a brief introduction to working with Markov Chains from the `prob140` library. Make sure you have read the
other tutorial first.

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

To assign the possible states of a markov chain, use `Table().states()`

.. ipython:: python

    Table().states(make_array("A", "B"))

A markov chain needs transition probabilities for each transition state `i` to `j`. Note that the sum of the
transition probabilities coming out of each state must sum to 1

.. ipython:: python

    mc_table = Table().states(make_array("A", "B")).transition_probability(make_array(0.5, 0.5, 0.3, 0.7))
    mc_table

To convert the Table into a MarkovChain object, call `.toMarkovChain()`

.. ipython:: python

    mc = mc_table.toMarkovChain()
    mc

Using a transition probability function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use like single variable distributions and joint distributions, we can assign a transition probability function.

.. ipython:: python

    def identity_transition(x,y):
        if x==y:
            return 1
        return 0

    transMatrix = Table().states(np.arange(1,4)).transition_function(identity_transition)
    transMatrix

