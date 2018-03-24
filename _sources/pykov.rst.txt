Markov Chains (``prob140.MarkovChain``)
=======================================
.. currentmodule:: prob140

Construction
------------

Using a Table
^^^^^^^^^^^^^

You can use a 3 column table (source state, target state, transition
probability) to construct a Markov Chain. The functions
`Table.transition_probability()` or `Table.transition_function()` are helpful
for constructing such a Table. From there, call `Markov_chain.from_table()` to
construct a Markov Chain.

.. ipython:: python

    mc_table = Table().states(make_array("A", "B")).transition_probability(make_array(0.5, 0.5, 0.3, 0.7))
    mc_table
    MarkovChain.from_table(mc_table)

Using a transition function
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Often, it will be more useful to define a transition function that returns the
probability of going from a source to a target state.

.. ipython:: python

    states = ['state_1', 'state_2']
    def identity_transition(source, target):
        if source == target:
            return 1
        return 0

    MarkovChain.from_transition_function(states, identity_transition)


Using a transition matrix
^^^^^^^^^^^^^^^^^^^^^^^^^

You can also explicitly define the transition matrix.

.. ipython:: python

    import numpy
    states = ['rainy', 'sunny']
    transition_matrix = numpy.array([[0.1, 0.9],
                                     [0.8, 0.2]])
    MarkovChain.from_matrix(states, transition_matrix)

.. autosummary::
    :toctree: _autosummary

    Table.transition_probability
    MarkovChain.from_table
    MarkovChain.from_transition_function
    MarkovChain.from_matrix

Utilities
---------

.. autosummary::
    :toctree: _autosummary

    MarkovChain.distribution
    MarkovChain.steady_state
    MarkovChain.expected_return_time
    MarkovChain.prob_of_path
    MarkovChain.log_prob_of_path
    MarkovChain.get_transition_matrix
    MarkovChain.transition_matrix


Simulations
-----------

.. autosummary::
    :toctree: _autosummary

    MarkovChain.simulate_path

Visualizations
--------------

.. autosummary::
    :toctree: _autosummary

    MarkovChain.plot_path