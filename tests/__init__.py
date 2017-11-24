import collections


def assert_dist_equal(table, probabilities):
    assert_approx_equal(table.column(1), probabilities)


def assert_approx_equal(actual, expected):
    if isinstance(actual, collections.Iterable):
        for x, y in zip(actual, expected):
            assert_approx_equal(x, y)
    else:
        assert round(actual - expected, 6) == 0.
