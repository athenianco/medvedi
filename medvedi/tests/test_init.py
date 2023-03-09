from numpy.testing import assert_array_equal
import pytest

from medvedi import DataFrame


def test_init_dict_smoke():
    df = DataFrame(
        {
            "a": [1, 2, 3],
            "b": ["1", "2", "3"],
        },
    )
    assert_array_equal(df["a"], [1, 2, 3])
    assert_array_equal(df["b"], ["1", "2", "3"])
    assert df._index == ()


def test_init_dict_columns():
    with pytest.raises(ValueError):
        DataFrame(
            {
                "a": [1, 2, 3],
                "b": ["1", "2", "3"],
            },
            columns=["a", "b"],
        )


def test_init_iterable_columns():
    df = DataFrame(
        [[1, 2, 3], ["1", "2", "3"]],
        columns=["a", "b"],
    )
    assert_array_equal(df["a"], [1, 2, 3])
    assert_array_equal(df["b"], ["1", "2", "3"])
    assert df._index == ()


def test_init_iterable_default_columns():
    df = DataFrame([[1, 2, 3], ["1", "2", "3"]])
    assert_array_equal(df["0"], [1, 2, 3])
    assert_array_equal(df["1"], ["1", "2", "3"])
    assert df._index == ()


@pytest.mark.parametrize("index", ["a", ("a", "b"), [5, 6, 7]])
def test_init_iterable_index(index):
    df = DataFrame(
        [[1, 2, 3], ["1", "2", "3"]],
        columns=["a", "b"],
        index=index,
    )
    assert_array_equal(df["a"], [1, 2, 3])
    assert_array_equal(df["b"], ["1", "2", "3"])
    if isinstance(index, list):
        index = ("_index0",)
    if not isinstance(index, tuple):
        index = (index,)
    assert df._index == index
    assert df.index.names == index
    assert_array_equal(df.index.get_level_values(0), df[index[0]])
    if len(index) > 1:
        assert_array_equal(df.index.get_level_values(1), df[index[1]])


def test_init_empty():
    assert DataFrame()._columns == {}
    assert DataFrame()._index == ()


def test_init_wrong_dict():
    with pytest.raises(ValueError):
        DataFrame(
            {
                "a": [1, 2, 3],
                "b": ["1", "2"],
            },
        )
    DataFrame(
        {
            "a": [1, 2, 3],
            "b": ["1", "2"],
        },
        check=False,
    )


def test_init_bad_iterable():
    with pytest.raises(ValueError):
        DataFrame(
            [[1, 2, 3], ["1", "2"]],
            columns=["a", "b"],
        )


def test_init_dict_index():
    df = DataFrame(
        {
            "a": [1, 2, 3],
            "b": ["1", "2", "3"],
        },
        index="b",
    )
    df = DataFrame(
        {
            "a": [1, 2, 3],
            "c": ["1", "2", "3"],
        },
        index=df.index,
    )
    assert_array_equal(df["a"], [1, 2, 3])
    assert_array_equal(df["c"], ["1", "2", "3"])
    assert "b" not in df
    assert df._index == ("_index0",)
    assert_array_equal(df.index.get_level_values(0), ["1", "2", "3"])
