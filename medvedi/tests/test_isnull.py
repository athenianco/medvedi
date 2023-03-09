import numpy as np
from numpy import typing as npt
from numpy.testing import assert_array_equal
import pytest

from medvedi import DataFrame

methods = ("isnull", "notnull")


def inverse_as_needed(arr: npt.ArrayLike, method: str) -> npt.ArrayLike:
    return ~np.asarray(arr) if method == "notnull" else arr


@pytest.mark.parametrize("method", methods)
@pytest.mark.parametrize(
    "arr",
    [
        [1.2, 2.5, np.nan],
        np.array([1, 2, "NaT"], dtype="timedelta64[s]"),
        np.array([1000000, 2000000, "NaT"], dtype="datetime64[s]"),
        np.array([1, 2, None], dtype=object),
    ],
)
def test_isnull_notnull_exists(method, arr):
    df = DataFrame({"a": arr})
    assert_array_equal(
        getattr(df, method)("a"),
        inverse_as_needed([False, False, True], method),
        err_msg=f"{method}({arr})",
    )


@pytest.mark.parametrize("method", methods)
def test_isnull_notnull_not_exists(method):
    df = DataFrame({"a": [1, 2, 3]})
    assert_array_equal(
        getattr(df, method)("a"), inverse_as_needed([False, False, False], method), err_msg=method,
    )


@pytest.mark.parametrize("method", methods)
def test_isnull_notnull_bad_column(method):
    df = DataFrame({"a": [1, 2, 3]})
    with pytest.raises(KeyError):
        getattr(df, method)("b")
