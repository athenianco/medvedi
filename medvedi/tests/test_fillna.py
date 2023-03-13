import numpy as np
import pytest

from medvedi import DataFrame


@pytest.mark.parametrize("inplace", [False, True])
def test_fillna_all(inplace):
    df = DataFrame(
        {
            "a": np.array([1.1, None, 2.4], dtype=float),
            "b": np.array([None, "test", None], dtype=object),
        },
    )
    new_df = df.fillna(5, inplace=inplace)
    assert new_df["a"][1] == 5
    assert new_df["b"][0] == 5
    assert new_df["b"][2] == 5
    new_df["a"][1] = 10
    if inplace:
        assert df["a"][1] == 10
    else:
        assert df["a"][1] != df["a"][1]


@pytest.mark.parametrize("inplace", [False, True])
def test_fillna_column(inplace):
    df = DataFrame(
        {
            "a": np.array([1.1, None, 2.4], dtype=float),
            "b": np.array([None, "test", None], dtype=object),
        },
    )
    new_df = df.fillna(5, "a", inplace=inplace)
    assert new_df["a"][1] == 5
    assert new_df["b"][0] is None
    assert new_df["b"][2] is None
    new_df["a"][1] = 10
    if inplace:
        assert df["a"][1] == 10
    else:
        assert df["a"][1] != df["a"][1]
