import numpy as np
from numpy.testing import assert_array_equal
import pytest

from medvedi import DataFrame
from medvedi.testing import assert_frame_equal


def test_drop_duplicates_one_first():
    df = DataFrame({"a": [1, 2, 2, 3], "b": [0, 1, 2, 3]})
    unique_df = df.copy()
    unique_df.drop_duplicates("a", inplace=True)
    assert_array_equal(unique_df["a"], [1, 2, 3])
    assert_array_equal(unique_df["b"], [0, 1, 3])
    assert_array_equal(df["a"], [1, 2, 2, 3])

    unique_df = df.drop_duplicates("a")
    assert_array_equal(unique_df["a"], [1, 2, 3])
    assert_array_equal(unique_df["b"], [0, 1, 3])
    assert_array_equal(df["a"], [1, 2, 2, 3])


def test_drop_duplicates_one_last():
    df = DataFrame({"a": [1, 2, 2, 3], "b": [0, 1, 2, 3]})
    df.drop_duplicates("a", inplace=True, keep="last")
    assert_array_equal(df["a"], [1, 2, 3])
    assert_array_equal(df["b"], [0, 2, 3])

    df = DataFrame(
        {
            "a": [1, 2, 2, 3, 5, 3, 6],
            "b": [0, 1, 2, 3, 4, 5, 6],
        },
    )
    df.drop_duplicates("a", inplace=True, keep="last")
    assert_array_equal(df["a"], [1, 2, 3, 5, 6])
    assert_array_equal(df["b"], [0, 2, 5, 4, 6])


def test_drop_duplicates_bad_column():
    df = DataFrame({"a": [1, 2, 2, 3], "b": [0, 1, 2, 3]})
    with pytest.raises(KeyError):
        df.drop_duplicates("c")


def test_drop_duplicates_one_index():
    df = DataFrame({"a": [1, 2, 2, 3], "b": [0, 1, 2, 3]}, index="b")
    df.drop_duplicates("a", inplace=True, ignore_index=True)
    assert_array_equal(df["a"], [1, 2, 3])
    assert_array_equal(df["b"], [0, 1, 3])
    assert_array_equal(df.index.names, ())


def test_drop_duplicates_two():
    df = DataFrame(
        {"a": [1, 2, 2, 3], "b": np.array(["0", "1", "1", "3"], dtype="S1"), "c": [5, 6, 7, 8]},
    )
    df.drop_duplicates(("a", "b"), inplace=True)
    assert_array_equal(df["a"], [1, 2, 3])
    assert_array_equal(df["b"], np.array(["0", "1", "3"], dtype="S1"))
    assert_array_equal(df["c"], [5, 6, 8])


def test_drop_duplicates_noop():
    df = DataFrame({"a": [1, 2, 3], "b": [0, 10, 3]}, index="a")
    assert_frame_equal(df.drop_duplicates("a"), df)
