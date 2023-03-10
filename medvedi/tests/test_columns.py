import numpy as np
import pytest

from medvedi import DataFrame


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
