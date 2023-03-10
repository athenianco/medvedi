import numpy as np
from numpy.testing import assert_array_equal
import pytest

from medvedi import DataFrame


def test_set_index_name_existing():
    df = DataFrame({"a": [0, 1, 2]})
    new_df = df.set_index("a")
    assert new_df.index.names == ("a",)
    assert str(new_df.index) == "(a)"
    assert new_df.index.__sentry_repr__() == str(new_df.index)
    assert new_df is not df


def test_set_index_name_not_existing():
    df = DataFrame({"a": [0, 1, 2]})
    with pytest.raises(KeyError):
        df.set_index("b")


def test_set_index_name_clash_1d():
    df = DataFrame({"a": [0, 1, 2]})
    with pytest.raises(KeyError):
        df.set_index([5, 6, 7], inplace=True)
    df.set_index(np.array([5, 6, 7]), inplace=True)
    assert len(df.index.names) == 1
    with pytest.raises(ValueError):
        df.set_index(np.array([5, 6, 7]), inplace=True, drop=False)
    df.set_index(np.array([5, 6, 7]), inplace=True, drop=True)
    assert len(df.index.names) == 1


def test_set_index_name_clash_2d():
    df = DataFrame({"a": [0, 1, 2]})
    df.set_index([[5, 6, 7], ["a", "b", "c"]], inplace=True)
    assert len(df.index.names) == 2
    with pytest.raises(ValueError):
        df.set_index([[5, 6, 7], ["a", "b", "c"]], inplace=True, drop=False)
    df.set_index([[5, 6, 7], ["x", "y", "z"]], inplace=True, drop=True)
    assert len(df.index.names) == 2
    assert_array_equal(df.index.get_level_values(1), ["x", "y", "z"])


def test_set_index_drop():
    df = DataFrame({"a": [0, 1, 2], "b": ["a", "b", "c"]}, index="a")
    new_df = df.set_index("b", drop=True)
    assert df.columns == ("a", "b")
    assert_array_equal(df["a"], [0, 1, 2])
    assert new_df.columns == ("b",)

    new_df = df.set_index((), drop=True)
    assert new_df.index.names == ()
    assert new_df.columns == ("b",)

    new_df = df.set_index([], drop=True)
    assert new_df.index.names == ()
    assert new_df.columns == ("b",)

    df.set_index("b", drop=True, inplace=True)
    assert df.columns == ("b",)


def test_reset_index_drop():
    df = DataFrame({"a": [0, 1, 2], "b": ["a", "b", "c"]}, index="a")
    new_df = df.reset_index(drop=True)
    assert df.index.names == ("a",)
    assert df.columns == ("a", "b")
    assert_array_equal(df["a"], [0, 1, 2])
    assert new_df.columns == ("b",)
    assert new_df.index.names == ()

    df.reset_index(drop=True, inplace=True)
    assert df.columns == ("b",)
    assert df.index.names == ()


def test_reset_index_leave():
    df = DataFrame({"a": [0, 1, 2], "b": ["a", "b", "c"]}, index="a")
    df.reset_index(drop=False, inplace=True)
    assert df.columns == ("a", "b")
    assert_array_equal(df["a"], [0, 1, 2])
    assert df.index.names == ()
