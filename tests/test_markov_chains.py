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

    table_to_mc = table.to_markov_chain()
    mc_from_table = MarkovChain.from_table(table)
    mc_from_function = MarkovChain.from_transition_function(states, trans_func)
    mc_from_matrix = MarkovChain.from_matrix(states, transition_matrix)

    assert_array_equal(
        transition_matrix,
        table_to_mc.get_transition_matrix()
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


def test_distribution():
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