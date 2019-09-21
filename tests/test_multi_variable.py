import numpy as np
import pytest

from prob140 import (
    JointDistribution,
    Table,
)

from . import (
    assert_approx_equal,
    assert_dist_equal,
    assert_equal,
    assert_joint_equal,
)


DIST1 = Table().values([0, 1], [2, 3, 4]).probabilities(
    [0.1, 0.2, 0., 0.3, 0.4, 0.]).to_joint()
DIST2 = Table().values('Coin1', ['H','T'], 'Coin2', ['H','T']).probabilities(
    [0.24, 0.36, 0.16, 0.24]).to_joint()


def test_construction():
    # String values
    table_str = Table().values('X', ['H', 'T'], 'Y', ['H', 'T'])
    table_str = table_str.probabilities(np.array([0.24, 0.36, 0.16, 0.24]))
    dist = table_str.to_joint()
    assert_joint_equal(dist, np.array([[0.36, 0.24], [0.24, 0.16]]))
    assert_equal(dist.get_possible_values(), [['H', 'T'], ['T', 'H']])
    dist = table_str.to_joint(reverse=False)
    assert_joint_equal(dist, np.array([[0.24, 0.16], [0.36, 0.24]]))
    assert_equal(dist.get_possible_values(), [['H', 'T'], ['H', 'T']])

    # Int values
    table_int = Table().values('X', [1], 'Y', [2]).probabilities([1])
    dist = JointDistribution.from_table(table_int)
    x_values = dist.get_possible_values('X')
    assert_approx_equal(x_values, [1])
    assert isinstance(x_values[0], int)
    y_values = dist.get_possible_values('Y')
    assert_approx_equal(y_values, [2])
    assert isinstance(y_values[0], int)

    # Float values
    table_float = Table().values('X', [1.1], 'Y', [2.2]).probabilities([1])
    dist = JointDistribution.from_table(table_float)
    x_values = dist.get_possible_values('X')
    assert isinstance(x_values[0], float)

    # Negative probability values
    with pytest.warns(UserWarning):
        Table().values('X', [1, 2], 'Y', [3])\
            .probabilities([-0.5, 1.5]).to_joint()

    # Doesn't sum to 1
    with pytest.warns(UserWarning):
        Table().values('X', [1, 2], 'Y', [3])\
            .probabilities([1., 1.]).to_joint()


def test_both_marginals():
    assert_joint_equal(DIST1.both_marginals(),
                       np.array([[0.0, 0.0, 0.0],
                                 [0.2, 0.4, 0.6],
                                 [0.1, 0.3, 0.4],
                                 [0.3, 0.7, 1.0]]))
    assert_joint_equal(DIST2.both_marginals(),
                       np.array([[0.36, 0.24, 0.6],
                                 [0.24, 0.16, 0.4],
                                 [0.60, 0.40, 1.0]]))


def test_marginal_dist():
    assert_dist_equal(DIST1.marginal_dist('X'), [0.3, 0.7])
    assert_dist_equal(DIST1.marginal_dist('Y'), [0, 0.6, 0.4])
    assert_dist_equal(DIST2.marginal_dist('Coin1'), [0.6, 0.4])
    assert_dist_equal(DIST2.marginal_dist('Coin2'), [0.6, 0.4])

    # Nonexistent label
    with pytest.raises(AssertionError):
        DIST1.marginal_dist('Z')
    with pytest.raises(AssertionError):
        DIST2.marginal_dist('Coin3')


def test_conditional_dist():
    assert_joint_equal(DIST1.conditional_dist('X', given='Y', show_ev=True),
                       np.array([[np.nan, np.nan, np.nan, np.nan],
                                [0.33333333, 0.66666667, 1., 0.66666667],
                                [0.25, 0.75, 1., 0.75],
                                [0.3, 0.7, 1., 0.7]]))
    assert_joint_equal(DIST1.conditional_dist('Y', given='X', show_ev=True),
                       np.array([[0., 0., 0.],
                                [0.66666667, 0.57142857, 0.6],
                                [0.33333333, 0.42857143, 0.4],
                                [1., 1., 1.],
                                [2.66666667, 2.57142857, 2.6]]))

    # Nonexistent label
    with pytest.raises(AssertionError):
        DIST1.conditional_dist('A', 'B')
