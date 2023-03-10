import numpy as np
from numpy.testing import assert_array_equal
import pytest

from medvedi import DataFrame


def test_in_bad_column():
    with pytest.raises(KeyError):
        DataFrame({"a": [1, 2, 3]}).in_("b", [1, 2])


def test_in_bad_dtype():
    with pytest.raises(ValueError):
        DataFrame({"a": [1, 2, 3]}).in_("a", [1.3, 2.5])


def test_in_int():
    df = DataFrame({"a": [1, 2, 3]})
    assert_array_equal(df.in_("a", [2, 3, 4]), [False, True, True])


def test_in_s():
    df = DataFrame({"a": np.array(["1", "2", "3"], dtype="S1")})
    assert_array_equal(df.in_("a", [b"2", b"3", b"4"]), [False, True, True])


def test_in_u():
    df = DataFrame({"a": np.array(["1", "2", "3"], dtype="U1")})
    assert_array_equal(df.in_("a", ["2", "3", "4"]), [False, True, True])


def test_in_object():
    df = DataFrame({"a": np.array([1, 2, 3], dtype=object)})
    assert_array_equal(df.in_("a", np.array([2, 3, 4], dtype=object)), [False, True, True])


def test_in_invert():
    df = DataFrame({"a": [1, 2, 3]})
    assert_array_equal(df.in_("a", [2, 3, 4], invert=True), [True, False, False])

    df = DataFrame({"a": np.array(["1", "2", "3"], dtype="S1")})
    assert_array_equal(
        df.in_("a", np.array(["2", "3", "4"], dtype="S1"), invert=True), [True, False, False],
    )
