import numpy as np
from numpy.testing import assert_array_equal
import pytest

from medvedi import DataFrame
from medvedi.testing import assert_frame_equal


@pytest.mark.parametrize("ignore_index", [False, True])
def test_explode_ignore_index(ignore_index):
    df = DataFrame(
        {"a": np.array([[1, 2], [3], [4, 5, 6], 7, "a"], dtype=object), "b": [9, 8, 7, 6, 5]},
        index="b",
    )
    new_df = df.explode("a", ignore_index=ignore_index)
    assert_array_equal(new_df["a"], [1, 2, 3, 4, 5, 6, 7, "a"])
    assert_array_equal(new_df["b"], [9, 9, 8, 7, 7, 7, 6, 5])
    if ignore_index:
        assert new_df._index == ()
    else:
        assert new_df._index == ("b",)


def test_explode_non_object():
    df = DataFrame({"a": [0, 1, 2]})
    new_df = df.explode("a")
    assert new_df is not df
    assert_frame_equal(new_df, df)

    new_df = df.explode("a", ignore_index=True)
    assert_array_equal(new_df["a"], df["a"])
    assert new_df._index == ()
