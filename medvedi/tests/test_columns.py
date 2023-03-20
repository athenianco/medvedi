import numpy as np
from numpy.testing import assert_array_equal
import pytest

from medvedi import DataFrame
from medvedi.accelerators import array_of_objects
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
    old_arr = df["a"]
    df["a"] = 2
    assert_array_equal(df["a"], [2, 2, 2])
    assert df["a"].dtype == object
    assert_array_equal(old_arr, [0, 1, 2])

    df = DataFrame({"a": np.array([0, 1, 2], dtype=int)})
    df["a"] = 2
    assert_array_equal(df["a"], [2, 2, 2])
    assert df["a"].dtype == int


@pytest.mark.parametrize("dtype", ["S", "U"])
def test_set_column_string_scalar(dtype):
    df = DataFrame({"a": np.array(["0", "1", "2"], dtype=dtype)})
    df["a"] = "test"
    assert_array_equal(df["a"], np.array(["test"] * 3, dtype=dtype))
    with pytest.raises(ValueError):
        df["a"] = 45


def test_get_column_tuple():
    df = DataFrame({"a": np.array([0, 1, 2], dtype=object), "b": [5, 6, 7]}, index="b")
    assert_frame_equal(df[("a",)], df)


def test_array_of_objects_smoke():
    arr = array_of_objects(10, None)
    assert_array_equal(arr, [None] * 10)
    arr = array_of_objects(10, [])
    assert len(arr) == 10
    assert arr.dtype == object
    arr[0].append(1)
    assert arr[1] == [1]
