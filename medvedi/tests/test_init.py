import numpy as np
from numpy.testing import assert_array_equal
import pytest

from medvedi import DataFrame
from medvedi.testing import assert_frame_equal


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
    assert df.columns == ("a", "b")
    assert len(str(df)) > 0
    assert df.shape == (3, 2)
    assert len(repr(df)) > 0
    assert len(df.__sentry_repr__()) > 0


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


@pytest.mark.parametrize("index", ["a", ("a", "b"), np.array([5, 6, 7])])
def test_init_iterable_index(index):
    df = DataFrame(
        [[1, 2, 3], ["1", "2", "3"]],
        columns=["a", "b"],
        index=index,
    )
    assert_array_equal(df["a"], [1, 2, 3])
    assert_array_equal(df["b"], ["1", "2", "3"])
    if isinstance(index, np.ndarray):
        index = ("_index0",)
    if not isinstance(index, tuple):
        index = (index,)
    assert df._index == index
    assert df.index.names == index
    assert len(df.index) == len(df)
    assert_array_equal(df.index.get_level_values(0), df[index[0]])
    if len(index) > 1:
        assert_array_equal(df.index.get_level_values(1), df[index[1]])
    with pytest.raises(IndexError):
        df.index.get_level_values(2)
    with pytest.raises(IndexError):
        df.index.get_level_values(-1)


def test_init_empty():
    assert DataFrame()._columns == {}
    assert DataFrame()._index == ()
    assert len(DataFrame()) == 0


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


def test_init_bad_shape():
    with pytest.raises(ValueError):
        DataFrame(
            {
                "a": [[1, 2, 3], [5, 6, 7]],
            },
            index="a",
        )


def test_init_bad_type():
    with pytest.raises(TypeError):
        DataFrame(
            {
                "a": "xxx",
            },
            index="a",
        )


def test_get_columns():
    df = DataFrame(
        {
            "a": [1, 2, 3],
            "b": ["1", "2", "3"],
        },
        index="b",
    )
    assert_frame_equal(
        df[["b"]],
        DataFrame(
            {
                "b": ["1", "2", "3"],
            },
            index="b",
        ),
    )
    assert_frame_equal(df[["a"]], df)


def test_init_dict_dtype():
    df = DataFrame(
        {
            "a": [1, 2, 3],
            "b": ["1", "2", "3"],
        },
        dtype={"a": object, "b": "S1"},
    )
    assert df["a"].dtype == object
    assert df["b"].dtype == "S1"


def test_init_dict_part_dtype():
    df = DataFrame(
        {
            "a": [1, 2, 3],
            "b": ["1", "2", "3"],
        },
        dtype={"a": object},
    )
    assert df["a"].dtype == object
    assert df["b"].dtype == "U1"


def test_init_iterable_dtype():
    df = DataFrame(
        [[1, 2, 3], ["1", "2", "3"]],
        columns=["a", "b"],
        dtype={"a": object, "b": "S1"},
    )
    assert df["a"].dtype == object
    assert df["b"].dtype == "S1"


def test_init_empty_dtype():
    df = DataFrame(columns=["a", "b"], dtype={"a": int})
    assert df["a"].dtype == int
    assert df["b"].dtype == object
