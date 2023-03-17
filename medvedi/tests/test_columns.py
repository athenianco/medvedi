import numpy as np
from numpy.testing import assert_array_equal
import pytest

from medvedi import DataFrame
from medvedi.testing import assert_frame_equal


def test_set_column_errors():
    df = DataFrame()
    with pytest.raises(ValueError):
        df["a"] = np.ones((2, 2))
    df["a"] = [0, 1, 2]
    with pytest.raises(ValueError):
        df["b"] = [0, 1]


def test_delete_column_smoke():
    df = DataFrame({"a": [0, 1, 2]})
    del df["a"]
    assert len(df) == 0


def test_delete_column_bad():
    df = DataFrame({"a": [0, 1, 2]})
    with pytest.raises(KeyError):
        del df["b"]


def test_delete_column_index():
    df = DataFrame({"a": [0, 1, 2]}, index="a")
    with pytest.raises(ValueError):
        del df["a"]


def test_set_column_scalar():
    df = DataFrame({"a": [0, 1, 2]})
    df["b"] = 2
    assert len(df["b"]) == 3
    assert (df["b"] == 2).all()


@pytest.mark.parametrize("value", [2, None])
def test_set_column_empty_scalar(value):
    df = DataFrame()
    df["b"] = value
    assert len(df["b"]) == 0

    df = DataFrame({"b": []})
    df["b"] = value
    assert len(df["b"]) == 0


def test_set_column_existing_scalar():
    df = DataFrame({"a": np.array([0, 1, 2], dtype=object)})
    df["a"] = 2
    assert_array_equal(df["a"], [2, 2, 2])
    assert df["a"].dtype == object


def test_get_column_tuple():
    df = DataFrame({"a": np.array([0, 1, 2], dtype=object), "b": [5, 6, 7]}, index="b")
    assert_frame_equal(df[("a",)], df)
