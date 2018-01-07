import numpy as np
import pytest

from prob140 import (
    emp_dist,
    Table,
)

from . import (
    assert_approx_equal,
    assert_dist_equal,
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

    for x in np.arange(-0.5, 100, 0.5):
        assert_approx_equal(UNIFORM_DIST.cdf(x), np.ceil(x + 0.5) / 100)


def test_emp_dist():
    values = np.array([1] * 5 + [2] * 3 + [3] * 2)
    empirical_dist = emp_dist(values)
    assert_approx_equal(
        empirical_dist.column(0),
        np.array([1, 2, 3])
    )
    assert_approx_equal(
        empirical_dist.column(1),
        np.array([0.5, 0.3, 0.2])
    )


def test_events():
    # Calculating events.
    assert_approx_equal(UNIFORM_DIST.prob_event(np.arange(50)), 0.5)
    assert_dist_equal(UNIFORM_DIST.event(np.arange(0, 100, 2)), [0.01] * 50)

    # Empty event.
    assert_approx_equal(SIMPLE_DIST.prob_event([]), 0.)
    assert_dist_equal(SIMPLE_DIST.event([]), [0])

    # Partial missing events.
    assert_approx_equal(NEGATIVE_DIST.prob_event(np.arange(-2, 1, 0.5)), 0.75)
    assert_dist_equal(NEGATIVE_DIST.event(np.arange(-2, 1, 0.5)), [0.25, 0] * 4)

    # Full missing events
    assert_approx_equal(NONINTEGER_DIST.prob_event(0), 0)
    assert_dist_equal(NONINTEGER_DIST.event(0), [0])


def test_normalized():
    dist = Table().values([1, 2, 3]).probability([1] * 3)
    assert_dist_equal(dist.normalized(), [1 / 3] * 3)


def test_sample_from_dist():
    assert SIMPLE_DIST.sample_from_dist() in [1, 2, 3]
    for x in SIMPLE_DIST.sample_from_dist(100):
        assert x in [1, 2, 3]


def test_ev():
    assert_approx_equal(SIMPLE_DIST.ev(), 2.3)
    assert_approx_equal(UNIFORM_DIST.ev(), 49.5)
    assert_approx_equal(NEGATIVE_DIST.ev(), -0.5)
    assert_approx_equal(NONINTEGER_DIST.ev(), 0)


def test_var():
    assert_approx_equal(SIMPLE_DIST.var(), 0.61)
    assert_approx_equal(UNIFORM_DIST.var(), 833.25)
    assert_approx_equal(NEGATIVE_DIST.var(), 1.25)
    assert_approx_equal(NONINTEGER_DIST.var(), 1.25)


def test_sd():
    assert_approx_equal(SIMPLE_DIST.sd(), np.sqrt(0.61))
    assert_approx_equal(UNIFORM_DIST.sd(), np.sqrt(833.25))
    assert_approx_equal(NEGATIVE_DIST.sd(), np.sqrt(1.25))
    assert_approx_equal(NONINTEGER_DIST.sd(), np.sqrt(1.25))


def test_remove_zeros():
    dist = Table().values([2, 3, 4, 5]).probability([0.5, 0.0, 0.5, 0])
    assert dist.remove_zeros().num_rows == 2
