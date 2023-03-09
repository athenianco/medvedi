import numpy as np
import pytest

from medvedi.merge_to_str import merge_to_str


def test_merge_to_str_ints():
    arr = merge_to_str(np.array([1, 2, 3], dtype=int), np.array([4, 5, 6], dtype=int))
    assert (
        bytes(arr.data)
        == b"\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x04;"
        b"\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x05;"
        b"\x00\x00\x00\x00\x00\x00\x00\x03\x00\x00\x00\x00\x00\x00\x00\x06;"
    )


def test_merge_to_str_types():
    arr = merge_to_str(
        np.array([1, 2, 3], dtype=np.uint32),
        np.array(["a", "b", "c"], dtype="S1"),
        np.array([1, 2, -3], dtype="timedelta64[s]"),
        np.array(["NaT", "2020-01-01", "1970-01-01"], dtype="datetime64[s]"),
    )
    assert (
        bytes(arr.data)
        == b"\x00\x00\x00\x01a\x00\x00\x00\x00\x00\x00\x00\x01\x80\x00\x00\x00\x00\x00\x00\x00;"
        b"\x00\x00\x00\x02b\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00^\x0b\xe1\x00;"
        b"\x00\x00\x00\x03c\xff\xff\xff\xff\xff\xff\xff\xfd\x00\x00\x00\x00\x00\x00\x00\x00;"
    )


def test_merge_to_str_size_mismatch():
    with pytest.raises(ValueError):
        merge_to_str(np.array([1, 2, 3], dtype=int), np.array([4, 5], dtype=int))


def test_merge_to_str_bad_dtype():
    with pytest.raises(ValueError):
        merge_to_str(np.array([1, 2, 3], dtype=int), np.array([1.0, 2.5, 3.3]))
