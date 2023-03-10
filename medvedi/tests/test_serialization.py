import json
from lzma import LZMAFile
from pathlib import Path

import numpy as np
import pytest

from medvedi import DataFrame
from medvedi.io import CorruptedBuffer, json_dumps
from medvedi.testing import assert_frame_equal


def test_zero_rows():
    df = DataFrame(columns=("a", "b", "c"))
    assert df.empty
    assert_frame_equal(df, DataFrame.deserialize_unsafe(df.serialize_unsafe()))


def test_json_torture():
    with LZMAFile(Path(__file__).with_name("torture.json.xz")) as fin:
        obj = json.loads(fin.read().decode())
    json.loads(json_dumps(obj).decode())


def test_json_smoke():
    obj = {
        "aaa": ["bb", 123, 100, 1.25, None],
        "bbb": {
            "x": True,
            "y": False,
            "áббц": "zz",
        },
    }
    assert obj == json.loads(json_dumps(obj).decode())


def array_of_objects(n, value):
    arr = np.empty(n, dtype=object)
    for i in range(n):
        arr[i] = value
    return arr


def test_smoke():
    df = DataFrame(
        {
            "a": np.array(["x", "yy", "zzz"], dtype=object),
            "b": np.array([1, 2002, 3000000003], dtype=int),
            "c": np.array([b"aaa", b"bb", b"c"], dtype="S3"),
            "d": [None, "mom", "dad"],
            "e": [101, None, 303],
            "f": array_of_objects(3, np.array(["x", "yy", "zzz"], dtype=object)),
            "g": array_of_objects(3, np.array(["x", "yy", "zzz"], dtype="S3")),
            "h": array_of_objects(3, np.array([1, 2002, 3000000003], dtype=int)),
            "i": [
                {
                    "api": {"helm_chart": "0.0.103"},
                    "web-app": {"helm_chart": "0.0.53"},
                    "push-api": {"helm_chart": "0.0.29"},
                    "precomputer": {"helm_chart": "0.0.1"},
                },
                {
                    "build": 619,
                    "checksum": "ae46ff6e9059cc1f71086fafd81f0d894deb15d4d18169031df4a5204f434bbc",
                    "job-name": "athenian/metadata/olek",
                },
                {
                    "build": 1448,
                    "author": "gkwillie",
                    "checksum": "4ef9840a007a2187b665add635a9d95daaa26a6165288175d891525f0d70cc6e",
                    "job-name": "athenian/infrastructure/production",
                },
            ],
            "j": np.array([1, -2, 3], dtype="timedelta64[s]"),
            "k": np.array(["NaT"] * 3, dtype="datetime64"),
        },
    )
    new_df = DataFrame.deserialize_unsafe(df.serialize_unsafe())
    assert_frame_equal(df, new_df)


def test_corrupted():
    df = DataFrame(
        {
            "a": np.array(["x", "yy", "zzz"], dtype=object),
            "b": np.array([1, 2002, 3000000003], dtype=int),
            "c": np.array([b"aaa", b"bb", b"c"], dtype="S3"),
            "d": [None, "mom", "dad"],
        },
    )
    buffer = df.serialize_unsafe()
    assert DataFrame.deserialize_unsafe(buffer).columns == ("a", "b", "c", "d")
    for i in range(len(buffer)):
        with pytest.raises(CorruptedBuffer):
            DataFrame.deserialize_unsafe(buffer[:i])
    with pytest.raises(CorruptedBuffer):
        DataFrame.deserialize_unsafe(buffer + b"\x00")


def test_all_null_rows():
    df = DataFrame({"a": [None, None]})
    assert_frame_equal(df, DataFrame.deserialize_unsafe(df.serialize_unsafe()))


def test_all_empty_list_rows():
    arr = np.empty(2, dtype=object)
    arr[0] = arr[1] = []
    df = DataFrame({"a": arr})
    assert_frame_equal(df, DataFrame.deserialize_unsafe(df.serialize_unsafe()))


def test_fortran():
    ints = np.empty((2, 10), dtype=int, order="F")
    objs = np.empty((2, 10), dtype=object, order="F")
    ints[0] = np.arange(10)
    ints[1] = np.arange(10, 20)
    objs[0] = [f"1_{i}" for i in range(10)]
    objs[1] = [f"2_{i}" for i in range(10)]
    with pytest.raises(AssertionError):
        DataFrame(ints).serialize_unsafe()
    df = DataFrame(objs)
    assert_frame_equal(df, DataFrame.deserialize_unsafe(df.serialize_unsafe()))
