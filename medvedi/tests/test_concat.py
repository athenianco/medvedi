import numpy as np
from numpy.testing import assert_array_equal
import pytest

from medvedi import DataFrame
from medvedi.pure_static import PureStaticDataFrameMethods


def test_concat_zero():
    assert DataFrame._concat().empty


def test_concat_one():
    df = DataFrame({"a": [1, 2, 3]})
    assert DataFrame.concat(df) is df


def test_concat_method():
    df = DataFrame({"a": [1, 2, 3]})
    with pytest.raises(AttributeError):
        df.concat(df)

    with pytest.raises(NotImplementedError):
        PureStaticDataFrameMethods._concat(DataFrame)


def test_concat_bullshit():
    with pytest.raises(TypeError):
        DataFrame.concat("a", "b")

    with pytest.raises(TypeError):
        DataFrame.concat(DataFrame(), "b")

    with pytest.raises(ValueError):
        DataFrame.concat(DataFrame(), DataFrame({"a": [0, 1, 2]}))

    with pytest.raises(ValueError):
        DataFrame.concat(DataFrame({"a": [3, 4, 5]}), DataFrame({"a": [0, 1, 2]}, index="a"))

    DataFrame.concat(
        DataFrame({"a": [3, 4, 5]}), DataFrame({"a": [0, 1, 2]}, index="a"), ignore_index=True,
    )


def test_concat_two():
    df = DataFrame({"a": [1, 2, 3]})
    df = DataFrame.concat(df, df.take([1, 2]))
    assert len(df) == 5
    assert_array_equal(df["a"], [1, 2, 3, 2, 3])


def test_concat_three():
    df = DataFrame({"a": [1, 2, 3], "b": ["a", "b", "c"]})
    df = DataFrame.concat(df, df, df)
    assert len(df) == 9
    assert_array_equal(df["a"], [1, 2, 3] * 3)
    assert_array_equal(df["b"], ["a", "b", "c"] * 3)


def test_concat_bad():
    with pytest.raises(ValueError):
        DataFrame.concat(DataFrame({"a": [1, 2, 3]}), DataFrame({"b": [1, 2, 3]}))


def test_concat_index():
    df = DataFrame({"a": [1, 2, 3]}, index="a")
    df = DataFrame.concat(df, df.take([1, 2]))
    assert len(df) == 5
    assert_array_equal(df["a"], [1, 2, 3, 2, 3])
    assert df.index.names == ("a",)

    df = DataFrame({"a": [1, 2, 3]}, index="a")
    df = DataFrame.concat(df, df.take([1, 2]), ignore_index=True)
    assert len(df) == 5
    assert_array_equal(df["a"], [1, 2, 3, 2, 3])
    assert df.index.names == ()


def test_concat_copy():
    df = DataFrame({"a": [1, 2, 3]})
    assert DataFrame.concat(df, copy=True) is not df


def test_concat_dtypes():
    df = DataFrame({"a": [1, 2, 3]})
    df = DataFrame.concat(df, DataFrame({"a": np.array([], dtype=object)}))
    assert len(df) == 3
    assert_array_equal(df["a"], [1, 2, 3])
    assert df["a"].dtype == int


def test_concat_empty():
    assert DataFrame.concat(DataFrame({"a": []}), DataFrame({"a": []})).empty


def test_concat_not_strict_not_empty():
    df = DataFrame({"a": [1, 2, 3]})
    df = DataFrame.concat(df, DataFrame({"b": np.array(["x", "y"], dtype=object)}), strict=False)
    assert len(df) == 5
    assert_array_equal(df["a"], [1, 2, 3, 0, 0])
    assert_array_equal(df["b"], [None, None, None, "x", "y"])


def test_concat_not_strict_empty():
    df1 = DataFrame({"a": [1, 2, 3]})
    df2 = DataFrame({"b": np.array([], dtype=object)})
    for _df1, _df2 in [(df1, df2), (df2, df1)]:
        df = DataFrame.concat(_df1, _df2, strict=False)
        assert len(df) == 3
        assert_array_equal(df["a"], [1, 2, 3])
        assert_array_equal(df["b"], [None, None, None])
