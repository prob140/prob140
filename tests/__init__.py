import collections

import numpy as np


def assert_dist_equal(table, probabilities):
    assert_approx_equal(table.column(1), probabilities)


def assert_joint_equal(joint_dist, probabilities):
    joint_prob = joint_dist.as_matrix()
    probabilities = np.array(probabilities)
    assert joint_prob.shape == probabilities.shape
    assert_approx_equal(joint_prob, probabilities)


def assert_approx_equal(actual, expected):
    if isinstance(actual, collections.Iterable):
        for x, y in zip(actual, expected):
            assert_approx_equal(x, y)
    elif np.isnan(actual):
        assert np.isnan(expected)
    else:
        assert actual is expected or round(actual - expected, 6) == 0.


def assert_equal(actual, expected):
    if isinstance(actual, collections.Iterable) and not isinstance(actual, str):
        for x, y in zip(actual, expected):
            assert_equal(x, y)
    else:
        assert actual == expected
