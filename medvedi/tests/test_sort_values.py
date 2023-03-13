import numpy as np
from numpy.testing import assert_array_equal
import pytest

from medvedi import DataFrame


def test_sort_values_one_ignore():
    df = DataFrame({"a": [3, 2, 1]})
    df.sort_values("a", inplace=True, ignore_index=True)
    assert df.index.names == ()
    assert_array_equal(df["a"], [1, 2, 3])


def test_sort_values_one_descending():
    df = DataFrame({"a": [3, 2, 1]})
    df.sort_values("a", inplace=True, ignore_index=True, ascending=False)
    assert df.index.names == ()
    assert_array_equal(df["a"], [3, 2, 1])


def test_sort_values_one_index():
    df = DataFrame({"a": [3, 2, 1]})
    df.sort_values("a", inplace=True, ignore_index=False)
    assert df.index.names == ("_index0",)
    assert_array_equal(df["a"], [1, 2, 3])
    assert_array_equal(df.index.get_level_values(0), [2, 1, 0])


def test_sort_values_one_copy():
    df = DataFrame({"a": [3, 2, 1]})
    x = df.sort_values("a", inplace=False, ignore_index=True)
    assert_array_equal(df["a"], [3, 2, 1])
    assert_array_equal(x["a"], [1, 2, 3])


def test_sort_values_mergeable():
    df = DataFrame({"a": [3, 2, 2, 1], "b": [0, 0, 1, 2]})
    df.sort_values(["a", "b"], inplace=True, ignore_index=True, non_negative_hint=True)
    assert_array_equal(df["a"], [1, 2, 2, 3])
    assert_array_equal(df["b"], [2, 0, 1, 0])

    df = DataFrame({"a": [3, 2, 2, 1], "b": [0, 0, -1, 2]})
    df.sort_values(["a", "b"], inplace=True, ignore_index=True)
    assert_array_equal(df["a"], [1, 2, 2, 3])
    assert_array_equal(df["b"], [2, -1, 0, 0])


def test_sort_values_unmergeable():
    df = DataFrame({"a": [3, 2, 2, 1], "b": np.array(["0", "0", "1", "2"], dtype=object)})
    df.sort_values(("a", "b"), inplace=True, ignore_index=True)
    assert_array_equal(df["a"], [1, 2, 2, 3])
    assert_array_equal(df["b"], ["2", "0", "1", "0"])


def test_sort_values_two_m():
    df = DataFrame(
        {"a": np.array([3, 2, 2, 1, "NaT", -2, -1], dtype="timedelta64[s]"), "b": [0] * 7},
    )
    df1 = df.sort_values(["b", "a"], ignore_index=True)
    assert_array_equal(df1["a"], np.array([-2, -1, 1, 2, 2, 3, "NaT"], dtype="timedelta64[s]"))

    df2 = df.sort_values(["b", "a"], ignore_index=True, na_position="first")
    assert_array_equal(df2["a"], np.array(["NaT", -2, -1, 1, 2, 2, 3], dtype="timedelta64[s]"))


def test_sort_values_one_m():
    df = DataFrame({"a": np.array([3, 2, 2, 1, "NaT", -2, -1], dtype="timedelta64[s]")})
    df1 = df.sort_values("a", ignore_index=True)
    assert_array_equal(df1["a"], np.array([-2, -1, 1, 2, 2, 3, "NaT"], dtype="timedelta64[s]"))

    df2 = df.sort_values("a", ignore_index=True, na_position="first")
    assert_array_equal(df2["a"], np.array(["NaT", -2, -1, 1, 2, 2, 3], dtype="timedelta64[s]"))


def test_sort_values_hint_two_m():
    df = DataFrame({"a": np.array([3, 2, 2, 1], dtype="timedelta64[s]"), "b": [0] * 4})
    df1 = df.sort_values(["b", "a"], ignore_index=True, non_negative_hint=True)
    assert_array_equal(df1["a"], np.array([1, 2, 2, 3], dtype="timedelta64[s]"))


def test_sort_values_empty_by():
    df = DataFrame({"a": [3, 2, 1]})
    with pytest.raises(ValueError):
        df.sort_values([])


def test_sort_index_levels_none():
    df = DataFrame({"a": [3, 2, 1]}, index="a")
    df.sort_index(inplace=True)
    assert_array_equal(df["a"], [1, 2, 3])


def test_sort_index_levels_single():
    df = DataFrame({"a": [3, 2, 1]}, index="a")
    df.sort_index(0, inplace=True)
    assert_array_equal(df["a"], [1, 2, 3])


def test_sort_index_levels_list():
    df = DataFrame({"a": [3, 2, 1]}, index="a")
    df.sort_index([0], inplace=True)
    assert_array_equal(df["a"], [1, 2, 3])


def test_sort_index_bad_type():
    df = DataFrame({"a": [3, 2, 1]}, index="a")
    with pytest.raises(TypeError):
        df.sort_index("a", inplace=True)
