from numpy.testing import assert_array_equal
import pytest

from medvedi import DataFrame


def test_iloc_scalar():
    df = DataFrame({"a": [1, 2], "b": [3, 4]})
    row = df.iloc[1]
    assert row["a"] == 2
    assert row["b"] == 4

    row = df.iloc[-1]
    assert row["a"] == 2
    assert row["b"] == 4


def test_iloc_oor():
    df = DataFrame({"a": [1, 2], "b": [3, 4]})
    with pytest.raises(IndexError):
        df.iloc[-3]
    with pytest.raises(IndexError):
        df.iloc[2]


def test_iloc_range():
    df = DataFrame({"a": [1, 2], "b": [3, 4]})
    result = df.iloc[1:2]
    assert isinstance(result, DataFrame)
    assert_array_equal(result["a"], [2])
    assert_array_equal(result["b"], [4])


def test_iloc_garbage():
    df = DataFrame({"a": [1, 2], "b": [3, 4]})
    with pytest.raises(TypeError):
        df.iloc["test"]
