Markov Chains (``prob140.MarkovChain``)
=======================================
.. currentmodule:: prob140

Constucting
-----------

Explicitly assigning probabilities

.. ipython:: python

    mc_table = Table().states(make_array("A", "B")).transition_probability(make_array(0.5, 0.5, 0.3, 0.7))
    mc_table
    mc = mc_table.toMarkovChain()
    mc

Using a transition function

.. ipython:: python

    def identity_transition(x,y):
        if x==y:
            return 1
        return 0

    transMatrix = Table().states(np.arange(1,4)).transition_function(identity_transition)
    transMatrix.toMarkovChain()


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
