import numpy as np
from numpy.testing import assert_array_equal
import pytest

from . import (
    assert_approx_equal,
    assert_dist_equal,
)
from prob140 import (
    MarkovChain,
    Table,
)


MC_SIMPLE = MarkovChain.from_matrix(
    states=np.array(['A', 'B']),
    transition_matrix=np.array([[0.1, 0.9], [0.8, 0.2]])
)
START_SIMPLE = Table().states(['A', 'B']).probability([0.8, 0.2])

def test_construction():
    # Transition matrix
    transition_matrix = np.array(
        [[0.5, 0.5, 0., 0., 0.],
         [0.25, 0.5, 0.25, 0., 0.],
         [0., 0.25, 0.5, 0.25, 0.],
         [0., 0., 0.25, 0.5, 0.25],
         [0., 0., 0., 0.5, 0.5]]
    )

    states = np.arange(1, 6)
    def trans_func(i, j):
        if i - j == 0:
            return 0.5
        elif 2 <= i <= 4:
            if abs(i - j) == 1:
                return 0.25
            else:
                return 0
        elif i == 1:
            if j == 2:
                return 0.5
            else:
                return 0
        elif i == 5:
            if j == 4:
                return 0.5
            else:
                return 0
    table = Table().states(states).transition_function(trans_func)
    transition_prob = transition_matrix.reshape((25))
    table2 = Table().states(states).transition_probability(transition_prob)

    table_to_mc = table.to_markov_chain()
    table2_to_mc = table2.to_markov_chain()
    mc_from_table = MarkovChain.from_table(table)
    mc_from_function = MarkovChain.from_transition_function(states, trans_func)
    mc_from_matrix = MarkovChain.from_matrix(states, transition_matrix)

    assert_array_equal(
        transition_matrix,
        table_to_mc.get_transition_matrix()
    )
    assert_array_equal(
        transition_matrix,
        table2_to_mc.get_transition_matrix()
    )
    assert_array_equal(
        transition_matrix,
        mc_from_table.get_transition_matrix()
    )
    assert_array_equal(
        transition_matrix,
        mc_from_function.get_transition_matrix()
    )
    assert_array_equal(
        transition_matrix,
        mc_from_matrix.get_transition_matrix()
    )

    # Negative probability.
    with pytest.warns(UserWarning):
        MarkovChain.from_matrix([1, 2], [[-1, 2], [0.5, 0.5]])
    # Transition probability doesn't sum to 1.
    with pytest.warns(UserWarning):
        MarkovChain.from_matrix([1, 2], [[0.2, 0.3], [1, 2]])



def test_distribution():
    assert_dist_equal(MC_SIMPLE.distribution('A'), [0.1, 0.9])
    assert_dist_equal(MC_SIMPLE.distribution(START_SIMPLE), [0.24, 0.76])
    assert_dist_equal(MC_SIMPLE.distribution(START_SIMPLE, 0), [0.8, 0.2])
    assert_dist_equal(MC_SIMPLE.distribution(START_SIMPLE, 3), [0.3576, 0.6424])


def test_log_prob_of_path():
    assert_approx_equal(
        MC_SIMPLE.log_prob_of_path('A', ['A', 'B', 'A']),
        -2.6310891599660815
    )
    assert_approx_equal(
        MC_SIMPLE.log_prob_of_path(START_SIMPLE, ['A', 'B', 'A']),
        -0.55164761828624576
    )


def test_prob_of_path():
    assert_approx_equal(
        MC_SIMPLE.prob_of_path('A', ['A', 'B', 'A']),
        0.072
    )
    assert_approx_equal(
        MC_SIMPLE.prob_of_path(START_SIMPLE, ['A', 'B', 'A']),
        0.576
    )


def test_simulate_path():
    mc_communicates = MarkovChain.from_matrix(
        states=np.array(['A', 'B']),
        transition_matrix=np.array([[0, 1], [1, 0]])
    )
    # Path should alternate values.
    path = mc_communicates.simulate_path(START_SIMPLE, 100)
    assert len(set(path)) == 2
    for i in range(len(path) - 1):
        assert path[i] != path[i + 1]

    mc_isolated = MarkovChain.from_matrix(
        states=np.array(['A', 'B']),
        transition_matrix=np.array([[1, 0], [0, 1]])
    )

    # Path should all have same values.
    path_A = mc_isolated.simulate_path('A', 100)
    assert len(set(path_A)) == 1
    path_B = mc_isolated.simulate_path('B', 100)
    assert len(set(path_B)) == 1


def test_steady_state():
    states = ['A', 'B']
    transition_matrix = np.array([[0.95, 0.05],
                                  [0.1, 0.9]])
    mc = MarkovChain.from_matrix(states, transition_matrix)
    assert_dist_equal(
        mc.steady_state(),
        [2 / 3, 1 / 3]
    )
    assert_dist_equal(
        mc.expected_return_time(),
        [1.5, 3]
    )
