import pytest

from medvedi import DataFrame


def test_iloc_smoke():
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
