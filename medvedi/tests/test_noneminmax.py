import numpy as np
import pytest

from medvedi import DataFrame


@pytest.mark.parametrize("attr", ["nonemin", "nonemax"])
def test_noneminmax_empty(attr):
    df = DataFrame({"a": []})
    assert getattr(df, attr)("a") is None


@pytest.mark.parametrize("attr", ["nonemin", "nonemax"])
@pytest.mark.parametrize("dtype", [object, float, "datetime64[s]"])
def test_noneminmax_all_nan(attr, dtype):
    df = DataFrame({"a": np.array([None], dtype=dtype)})
    assert getattr(df, attr)("a") is None


@pytest.mark.parametrize("attr", ["nonemin", "nonemax"])
@pytest.mark.parametrize("dtype", [object, float, "datetime64[s]"])
def test_noneminmax_some_nan(attr, dtype):
    df = DataFrame({"a": np.array([None, 0, 1], dtype=dtype)})
    assert getattr(df, attr)("a") == df["a"][1 + (attr == "nonemax")]
