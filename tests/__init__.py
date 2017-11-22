def assert_dist_equal(table, probabilities):
    assert_approx_equal(table.column(1), probabilities)

def assert_approx_equal(actual, expected):
    if isinstance(actual, float):
        assert round(actual - expected, 6) == 0.
    else:
        for x, y in zip(actual, expected):
            assert_approx_equal(x, y)
