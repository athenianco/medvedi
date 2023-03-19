import numpy as np
from numpy.testing import assert_array_equal
import pytest

from medvedi import DataFrame
from medvedi.testing import assert_index_equal


def test_index_properties():
    df = DataFrame({"a": [0, 1, 2]}, index=("a",))
    assert not df.index.empty
    assert df.index.nlevels == 1
    assert df.index.is_unique
    assert df.index.duplicated().sum() == 0
    assert df.index.name == "a"


def test_set_index_name_existing():
    df = DataFrame({"a": [0, 1, 2]})
    new_df = df.set_index("a")
    assert new_df.index.names == ("a",)
    assert str(new_df.index) == "(a)"
    assert new_df.index.__sentry_repr__() == str(new_df.index)
    assert new_df is not df
    with pytest.raises(AssertionError):
        assert_index_equal(df.index, new_df.index)


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
        df.index.name
    with pytest.raises(ValueError):
        df.set_index([[5, 6, 7], ["a", "b", "c"]], inplace=True, drop=False)
    df.set_index([[5, 6, 7], ["x", "y", "z"]], inplace=True, drop=True)
    assert len(df.index.names) == 2
    assert df.index.nlevels == 2
    assert_array_equal(df.index.get_level_values(1), ["x", "y", "z"])


def test_set_index_drop():
    df = DataFrame({"a": [0, 1, 2], "b": ["a", "b", "c"]}, index="a")
    new_df = df.set_index("b", drop=True)
    assert df.columns == ("a", "b")
    assert_array_equal(df["a"], [0, 1, 2])
    assert new_df.columns == ("b",)

    new_df = df.set_index((), drop=True)
    assert new_df.index.names == ()
    assert new_df.index.nlevels == 0
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


def test_index_values():
    df = DataFrame({"a": [0, 1, 2], "b": ["a", "b", "c"]}, index="a")
    assert_array_equal(df.index.values, [0, 1, 2])
    df.set_index(["a", "b"], inplace=True)
    with pytest.raises(AttributeError):
        df.index.values


@pytest.mark.parametrize("df", [DataFrame({"a": []}, index="a"), DataFrame()])
def test_index_empty(df):
    assert df.index.empty
    assert df.index.is_unique
    assert df.index.duplicated().sum() == 0
    assert df.index.nlevels == (1 if df.columns else 0)
    if not df.columns:
        with pytest.raises(ValueError):
            df.index.name


def test_index_is_monotonic_increasing_true():
    df = DataFrame({"a": [0, 1, 2]}, index="a")
    assert df.index.is_monotonic_increasing

    df = DataFrame({"a": [0, 1, 1]}, index="a")
    assert df.index.is_monotonic_increasing

    df = DataFrame({"a": [0, 1, 1], "b": [50, 5, 6]}, index=("a", "b"))
    assert df.index.is_monotonic_increasing

    df = DataFrame({"a": [0, 1, 1], "b": [50, 5, 5]}, index=("a", "b"))
    assert df.index.is_monotonic_increasing

    df = DataFrame({"a": [0, 1, 2], "b": [50, 6, 5]}, index=("a", "b"))
    assert df.index.is_monotonic_increasing


def test_index_is_monotonic_increasing_false():
    df = DataFrame({"a": [0, 3, 2]}, index="a")
    assert not df.index.is_monotonic_increasing

    df = DataFrame({"a": [0, 1, 1], "b": [50, 6, 5]}, index=("a", "b"))
    assert not df.index.is_monotonic_increasing


def test_index_is_monotonic_increasing_3d():
    df = DataFrame(
        {"a": [0, 1, 1, 2], "b": [50.3, 5.1, 5.1, 0], "c": ["x", "a", "b", ""]},
        index=("a", "b", "c"),
    )
    assert df.index.is_monotonic_increasing

    df = DataFrame(
        {"a": [0, 1, 1, 2], "b": [50.3, 5.1, 5.1, 0], "c": ["x", "b", "a", ""]},
        index=("a", "b", "c"),
    )
    assert not df.index.is_monotonic_increasing


def test_index_is_monotonic_increasing_empty():
    df = DataFrame({"a": []})
    assert df.index.is_monotonic_increasing

    df = DataFrame({"a": []}, index="a")
    assert df.index.is_monotonic_increasing

    df = DataFrame({"a": [1]}, index="a")
    assert df.index.is_monotonic_increasing


def test_index_is_monotonic_decreasing_true():
    df = DataFrame({"a": [2, 1, 0]}, index="a")
    assert df.index.is_monotonic_decreasing

    df = DataFrame({"a": [1, 1, 0]}, index="a")
    assert df.index.is_monotonic_decreasing

    df = DataFrame({"a": [1, 1, 0], "b": [6, 5, 50]}, index=("a", "b"))
    assert df.index.is_monotonic_decreasing

    df = DataFrame({"a": [1, 1, 0], "b": [5, 5, 50]}, index=("a", "b"))
    assert df.index.is_monotonic_decreasing

    df = DataFrame({"a": [2, 1, 0], "b": [5, 6, 50]}, index=("a", "b"))
    assert df.index.is_monotonic_decreasing


def test_index_is_monotonic_decreasing_false():
    df = DataFrame({"a": [2, 3, 0]}, index="a")
    assert not df.index.is_monotonic_decreasing

    df = DataFrame({"a": [1, 1, 0], "b": [5, 6, 50]}, index=("a", "b"))
    assert not df.index.is_monotonic_decreasing


def test_index_is_monotonic_decreasing_3d():
    df = DataFrame(
        {"a": [2, 1, 1, 0], "b": [0, 5.1, 5.1, 50.3], "c": ["", "b", "a", "x"]},
        index=("a", "b", "c"),
    )
    assert df.index.is_monotonic_decreasing

    df = DataFrame(
        {"a": [2, 1, 1, 0], "b": [0, 5.1, 5.1, 50.3], "c": ["", "a", "b", "x"]},
        index=("a", "b", "c"),
    )
    assert not df.index.is_monotonic_decreasing


def test_index_is_monotonic_increasing_2d_two():
    df = DataFrame(
        {"a": [0, 1, 1, 2, 3, 3], "b": [50.3, 5.1, 6.1, 0, 2.1, 3.2]},
        index=("a", "b"),
    )
    assert df.index.is_monotonic_increasing


def test_index_assert_equal():
    df1 = DataFrame({"a": [0, 1, 2]}, index="a")
    df2 = DataFrame({"a": [0, 1, 2], "b": [5, 6, 7]}, index="a")
    assert_index_equal(df1.index, df2.index)


def test_index_duplicates():
    df = DataFrame({"a": [0, 1, 1, 2]}, index=("a",))
    assert not df.index.is_unique
    assert_array_equal(df.index.duplicated(), [False, False, True, False])
    assert_array_equal(df.index.duplicated(keep="last"), [False, True, False, False])


def test_index_diff_smoke():
    df1 = DataFrame({"a": [0, 1, 1, 4, 2]}, index="a")
    df2 = DataFrame({"a": [1, 2, 2, 3]}, index="a")
    assert_array_equal(df1.index.diff(df2.index), [0, 3])


def test_index_diff_bad_type():
    df1 = DataFrame({"a": [0, 1, 1, 2]}, index="a")
    df2 = DataFrame({"a": [1, 2, 2, 3]}, index="a")
    with pytest.raises(TypeError):
        df1.index.diff(df2)
