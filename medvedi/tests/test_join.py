from numpy.testing import assert_array_equal
import pytest

from medvedi import DataFrame
from medvedi.testing import assert_frame_equal


def test_join_zero():
    assert DataFrame.join().empty


def test_join_one():
    df = DataFrame({"a": [0, 1, 2]})
    assert DataFrame.join(df) is df
    assert DataFrame.join(df, copy=True) is not df
    assert_frame_equal(DataFrame.join(df, copy=True), df)


def test_join_method():
    df = DataFrame({"a": [0, 1, 2]})
    with pytest.raises(AttributeError):
        df.join(df)


def test_join_two_disjoint():
    df1 = DataFrame({"i": [0, 1, 2], "a": [3, 4, 5]}, index="i")
    df2 = DataFrame({"i": [2, 1, 0], "b": [6, 7, 8]}, index="i")
    df = DataFrame.join(df1, df2)
    assert df._index == ("i",)
    assert_array_equal(df.index.get_level_values(0), [0, 1, 2])
    assert_array_equal(df["a"], [3, 4, 5])
    assert_array_equal(df["b"], [8, 7, 6])
    df["b"][1] = 100
    assert df2["b"][1] == 7


def test_join_two_missing():
    df1 = DataFrame({"i": [0, 1, 2], "a": [3, 4, 5]}, index="i")
    df2 = DataFrame({"i": [2, 1, 7], "b": [6, 7, 8]}, index="i")
    df = DataFrame.join(df1, df2)
    assert df._index == ("i",)
    assert_array_equal(df.index.get_level_values(0), [0, 1, 2])
    assert_array_equal(df["a"], [3, 4, 5])
    assert_array_equal(df["b"], [0, 7, 6])


def test_join_three_disjoint():
    df1 = DataFrame({"i": [0, 1, 2], "a": [3, 4, 5]}, index="i")
    df2 = DataFrame({"i": [2, 1, 0], "b": [6, 7, 8]}, index="i")
    df3 = DataFrame({"i": [1, 2, 0], "c": ["a", "b", "c"]}, index="i")
    df = DataFrame.join(df1, df2, df3)
    assert df._index == ("i",)
    assert_array_equal(df.index.get_level_values(0), [0, 1, 2])
    assert_array_equal(df["a"], [3, 4, 5])
    assert_array_equal(df["b"], [8, 7, 6])
    assert_array_equal(df["c"], ["c", "a", "b"])


def test_join_three_inner():
    df1 = DataFrame({"i": [0, 1, 2], "a": [3, 4, 5]}, index="i")
    df2 = DataFrame({"i": [2, 1, 0], "b": [6, 7, 8]}, index="i")
    df3 = DataFrame({"i": [1, 2, 7], "c": ["a", "b", "c"]}, index="i")
    df = DataFrame.join(df1, df2, df3, how="inner")
    assert df._index == ("i",)
    assert_array_equal(df.index.get_level_values(0), [1, 2])
    assert_array_equal(df["a"], [4, 5])
    assert_array_equal(df["b"], [7, 6])
    assert_array_equal(df["c"], ["a", "b"])


def test_join_three_outer():
    df1 = DataFrame({"i": [0, 1, 2], "a": [3, 4, 5]}, index="i")
    df2 = DataFrame({"i": [2, 1, 0], "b": [6, 7, 8]}, index="i")
    df3 = DataFrame({"i": [1, 2, 7], "c": ["a", "b", "c"]}, index="i")
    df = DataFrame.join(df1, df2, df3, how="outer")
    assert df._index == ("i",)
    assert_array_equal(df.index.get_level_values(0), [0, 1, 2, 7])
    assert_array_equal(df["a"], [3, 4, 5, 0])
    assert_array_equal(df["b"], [8, 7, 6, 0])
    assert_array_equal(df["c"], ["", "a", "b", "c"])


def test_join_three_right():
    df1 = DataFrame({"i": [0, 1, 2], "a": [3, 4, 5]}, index="i")
    df2 = DataFrame({"i": [2, 1, 0], "b": [6, 7, 8]}, index="i")
    df3 = DataFrame({"i": [1, 2, 7], "c": ["a", "b", "c"]}, index="i")
    df = DataFrame.join(df1, df2, df3, how="right")
    assert df._index == ("i",)
    assert_array_equal(df.index.get_level_values(0), [1, 2, 7])
    assert_array_equal(df["a"], [4, 5, 0])
    assert_array_equal(df["b"], [7, 6, 0])
    assert_array_equal(df["c"], ["a", "b", "c"])


def test_join_two_multilevel():
    df1 = DataFrame({"i1": [0, 1, 2], "i2": [0, -1, -2], "a": [3, 4, 5]}, index=("i1", "i2"))
    df2 = DataFrame({"i1": [2, 1, 0], "i2": [-2, -1, 7], "b": [6, 7, 8]}, index=("i1", "i2"))
    df = DataFrame.join(df1, df2)
    assert df._index == ("i1", "i2")
    assert_array_equal(df.index.get_level_values(0), [0, 1, 2])
    assert_array_equal(df.index.get_level_values(1), [0, -1, -2])
    assert_array_equal(df["a"], [3, 4, 5])
    assert_array_equal(df["b"], [0, 7, 6])


def test_join_two_suffix():
    df1 = DataFrame({"i": [0, 1, 2], "a": [3, 4, 5]}, index="i")
    df2 = DataFrame({"i": [2, 1, 0], "a": [6, 7, 8]}, index="i")
    df = DataFrame.join(df1, df2, suffixes=(None, "_"))
    assert df._index == ("i",)
    assert_array_equal(df.index.get_level_values(0), [0, 1, 2])
    assert_array_equal(df["a"], [3, 4, 5])
    assert_array_equal(df["a_"], [8, 7, 6])


def test_join_two_overwrite():
    df1 = DataFrame({"i": [0, 1, 2], "a": [3, 4, 5]}, index="i")
    df2 = DataFrame({"i": [2, 1, 0], "a": [6, 7, 8]}, index="i")
    df = DataFrame.join(df1, df2)
    assert df._index == ("i",)
    assert_array_equal(df.index.get_level_values(0), [0, 1, 2])
    assert_array_equal(df["a"], [8, 7, 6])
