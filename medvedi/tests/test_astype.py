import pytest

from medvedi import DataFrame


@pytest.mark.parametrize("copy", [False, True])
def test_astype_copy(copy):
    df = DataFrame({"a": [1, 2], "b": [3, 4]})
    new_df = df.astype(float, copy=copy)
    assert new_df["a"].dtype == float
    assert new_df["b"].dtype == float
    assert new_df.dtype == {
        "a": float,
        "b": float,
    }
    new_df["a"][0] = 0
    assert df["a"][0] == copy


def test_astype_mapping():
    df = DataFrame({"a": [1, 2], "b": [3, 4]})
    df.astype({"a": float}, copy=False)
    assert df["a"].dtype == float
    assert df["b"].dtype == int


def test_astype_errors():
    df = DataFrame({"a": ["a", "b"], "b": [3, 4]})
    with pytest.raises(ValueError):
        df.astype({"a": int}, copy=False)

    df.astype({"a": int, "b": float}, copy=False, errors="ignore")
    assert df["a"].dtype == "U1"
    assert df["b"].dtype == float
