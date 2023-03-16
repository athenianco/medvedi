import numpy as np
from numpy.testing import assert_array_equal
import pytest

from medvedi import DataFrame


def test_groupby_1d_iter():
    df = DataFrame({"a": [1, 1, 2, 2, 3, 3, 3], "b": [4, 5, 6, 7, 8, 9, 10]})
    a_values, b_values = df["a"], df["b"]
    for i, group in enumerate(df.groupby("a")):
        assert a_values[group[0]] == i + 1
        assert b_values[group].tolist() == ([4, 5], [6, 7], [8, 9, 10])[i]


def test_groupby_2d_int():
    df = DataFrame({"a": [1, 1, 2, 2, 3, 3, 3], "b": [4, 4, 6, 7, 10, 8, 8]})
    grouper = df.groupby("a", "b")
    assert grouper.order.tolist() == [0, 1, 2, 3, 5, 6, 4]
    assert grouper.counts.tolist() == [2, 1, 1, 2, 1]


@pytest.mark.parametrize("dtype", ["S2", object])
def test_groupby_2d_mixed_mergeable(dtype):
    df = DataFrame(
        {
            "a": [1, 1, 2, 2, 3, 3, 3],
            "b": np.array(["4", "4", "6", "7", "10", "8", "8"], dtype=dtype),
        },
    )
    grouper = df.groupby("a", "b")
    assert grouper.order.tolist() == [0, 1, 2, 3, 4, 5, 6]
    assert grouper.counts.tolist() == [2, 1, 1, 1, 2]


def test_groupby_bad_column():
    with pytest.raises(KeyError):
        DataFrame({"a": [1, 1, 2, 2, 3, 3, 3]}).groupby("c")
    with pytest.raises(TypeError):
        DataFrame({"a": [1, 1, 2, 2, 3, 3, 3]}).groupby(set())


def test_groupby_reduceat():
    df = DataFrame({"a": [3, 3, 3, 2, 2, 1, 1], "b": [9, 10, 7, 8, 4, 5, 6]})
    grouper = df.groupby("a")
    agg = np.add.reduceat(df["b"][grouper.order], grouper.reduceat_indexes())
    assert_array_equal(agg, [11, 12, 26])


def test_groupby_external():
    df = DataFrame({"a": [1, 1, 2, 2, 3, 3, 3], "b": [4, 5, 6, 7, 8, 9, 10]})
    g = df.groupby([0, 1, 0, 1, 0, 1, 0])
    assert_array_equal(g.counts, [4, 3])
    assert_array_equal(g.order, [0, 2, 4, 6, 1, 3, 5])

    with pytest.raises(ValueError):
        df.groupby([0, 1, 0, 1, 0, 1])


def test_groupby_group_indexes():
    df = DataFrame({"a": [1, 1, 2, 2, 3, 3, 3], "b": [4, 5, 6, 7, 8, 9, 10]})
    assert_array_equal(df.groupby("a").group_indexes(), [0, 2, 4])
    assert_array_equal(df["a"][df.groupby("a").group_indexes()], [1, 2, 3])


def test_groupby_empty():
    df = DataFrame({"a": []})
    g = df.groupby("a")
    assert len(g.reduceat_indexes()) == 0
    assert len(g.group_indexes()) == 0
    assert list(g) == []
