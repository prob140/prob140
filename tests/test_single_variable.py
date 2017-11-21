import doctest

import numpy as np
from numpy.testing import assert_array_equal
import pytest

from prob140 import (
    emp_dist,
    Table,
)


SIMPLE_DIST = Table().values([1, 2, 3]).probability([0.2, 0.3, 0.5])
UNIFORM_DIST = Table().values(np.arange(100)).probability([0.01] * 100)
NEGATIVE_DIST = Table().values([-2, -1, 0, 1]).probability([0.25] * 4)
NONINTEGER_DIST = Table().values([-1.5, -0.5, 0.5, 1.5]).probability([0.25] * 4)


def test_construction():

    domain = Table().values(np.array([1, 2, 3]))
    assert domain.num_columns == 1

    dist1 = domain.probability(np.array([0.1, 0.2 , 0.7]))
    assert dist1.num_columns == 2

    dist2 = domain.probability_function(lambda x: x / 6)
    assert dist2.num_columns == 2

    # Negative probability.
    with pytest.warns(UserWarning):
        domain.probability([0, 1.1, -0.1])
    with pytest.warns(UserWarning):
        domain.probability_function(lambda x: -x / 6)

    # Probability doesn't sum to 1.
    with pytest.warns(UserWarning):
        domain.probability([0.3, 0.1, 0.2])
    with pytest.warns(UserWarning):
        domain.probability_function(lambda x: x)


def test_cdf():
    assert SIMPLE_DIST.cdf(1) == 0.2
    assert SIMPLE_DIST.cdf(2) == 0.5
    assert SIMPLE_DIST.cdf(3) == 1


def test_emp_dist():
    values = np.array([1] * 5 + [2] * 3 + [3] * 2)
    empirical_dist = emp_dist(values)
    assert_array_equal(
        empirical_dist.column(0),
        np.array([1, 2, 3])
    )
    assert_array_equal(
        empirical_dist.column(1),
        np.array([0.5, 0.3, 0.2])
    )
