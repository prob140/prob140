Markov Chains (``prob140.MarkovChain``)
=======================================
.. currentmodule:: prob140

Constucting
-----------

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


Utilities
---------

.. autosummary::
    :toctree: _autosummary

    MarkovChain.distribution
    MarkovChain.steady_state
    MarkovChain.mean_first_passage_times
    MarkovChain.prob_of_path
    MarkovChain.mixing_time
    MarkovChain.accessibility_matrix



Simulations
-----------

.. autosummary::
    :toctree: _autosummary

    MarkovChain.move
    MarkovChain.simulate_chain
    MarkovChain.empirical_distribution
