import numpy as np
from numpy.testing import assert_array_equal
import pytest

from medvedi import DataFrame
from medvedi.pure_static import PureStaticDataFrameMethods
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

    with pytest.raises(NotImplementedError):
        PureStaticDataFrameMethods._join(DataFrame)


def test_join_bullshit():
    with pytest.raises(TypeError):
        DataFrame.join("a", "b")

    with pytest.raises(TypeError):
        DataFrame.join(DataFrame(), "b")

    with pytest.raises(TypeError):
        DataFrame.join(DataFrame(), suffixes="b")

    with pytest.raises(ValueError):
        DataFrame.join(DataFrame(), DataFrame({"a": [0, 1, 2]}))

    with pytest.raises(ValueError):
        DataFrame.join(DataFrame({"a": [3, 4, 5]}), DataFrame({"a": [0, 1, 2]}, index="a"))

    with pytest.raises(ValueError):
        DataFrame.join(*(DataFrame() for _ in range(1000)))

    with pytest.raises(ValueError):
        DataFrame.join(how="xxx")


@pytest.mark.parametrize("how", ["left", "inner"])
def test_join_empty(how):
    assert DataFrame.join(
        DataFrame({"a": []}, index="a"), DataFrame({"a": [0, 1, 2]}, index="a"), how=how,
    ).empty

    assert DataFrame.join(
        DataFrame({"a": []}, index="a"), DataFrame({"a": [0.1, 1.2, 2.3]}, index="a"), how=how,
    ).empty

    assert len(
        DataFrame.join(
            DataFrame({"a": [0.1, 1.2, 2.3]}, index="a"), DataFrame({"a": []}, index="a"), how=how,
        )["a"],
    ) == (3 if how == "left" else 0)


@pytest.mark.parametrize("how", ["left", "inner"])
def test_join_total_empty(how):
    joined = DataFrame.join(
        DataFrame({"a": []}, index="a"), DataFrame({"a": [], "b": []}, index="a"), how=how,
    )
    assert joined.empty
    assert joined.columns == ("a", "b")


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
    with pytest.raises(ValueError):
        DataFrame.join(df1, df2, suffixes=(None,))
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


def test_join_two_object_index():
    df1 = DataFrame(
        {"i": [0, 1, 2], "a": np.array(["0", "1", "2"], dtype=object)}, index=("i", "a"),
    )
    df2 = DataFrame(
        {"i": [2, 1, 0], "a": np.array(["2", "1", "0"], dtype=object)}, index=("i", "a"),
    )
    df = DataFrame.join(df1, df2)
    assert df._index == ("i", "a")
    assert_array_equal(df.index.get_level_values(0), [0, 1, 2])
    assert_array_equal(df.index.get_level_values(1), ["0", "1", "2"])


def test_join_two_float_object():
    df1 = DataFrame({"i": [0.1, 1.2, 2.3], "a": [3.1, 4.2, 5.3]}, index="i")
    df2 = DataFrame(
        {"i": [2.3, 1.2, 0.1], "b": [6.1, 7.2, 8.3], "c": np.array(["a", "b", "c"], dtype=object)},
        index="i",
    )
    df = DataFrame.join(df1, df2)
    assert df._index == ("i",)
    assert_array_equal(df.index.get_level_values(0), [0.1, 1.2, 2.3])
    assert_array_equal(df["a"], [3.1, 4.2, 5.3])
    assert_array_equal(df["b"], [8.3, 7.2, 6.1])
    assert_array_equal(df["c"], ["c", "b", "a"])


def test_join_two_incompatible_index():
    df1 = DataFrame(
        {"i": [0, 1, 2], "a": np.array(["0", "1", "2"], dtype=object)}, index=("i", "a"),
    )
    df2 = DataFrame({"i": [2, 1, 0], "a": np.array(["2", "1", "0"], dtype="S")}, index=("i", "a"))
    with pytest.raises(ValueError):
        DataFrame.join(df1, df2)
