from numpy.testing import assert_array_equal

from medvedi import DataFrame


def test_unique_smoke():
    df = DataFrame({"a": [3, 2, 2, 1]})
    assert_array_equal(df.unique("a"), [1, 2, 3])

    assert_array_equal(sorted(df.unique("a", unordered=True).tolist()), [1, 2, 3])
