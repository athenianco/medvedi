from typing import Hashable

import numpy as np
from numpy.testing import assert_array_equal

from medvedi import DataFrame, Index


def assert_frame_equal(df1: DataFrame, df2: DataFrame) -> None:
    """Assert whether two DataFrame-s are equivalent."""
    assert df1._index == df2._index, "indexes mismatch"
    assert df1._columns.keys() == df2._columns.keys(), "column names mismatch"
    for k in df1.columns:
        _assert_array_equal(df1[k], df2[k], k)


def _assert_array_equal(v1: np.ndarray, v2: np.ndarray, path: Hashable) -> None:
    if v1.dtype == object or v2.dtype == object:
        assert v1.dtype == v2.dtype, path
        assert v1.shape == v2.shape, path
        for i in range(len(v1)):
            vi1 = v1[i]
            vi2 = v2[i]
            if isinstance(vi1, np.ndarray) or isinstance(vi2, np.ndarray):
                assert isinstance(vi1, np.ndarray), (path, i)
                assert isinstance(vi2, np.ndarray), (path, i)
                _assert_array_equal(vi1, vi2, f"{path}[{i}]")
            else:
                assert vi1 == vi2, f"{path}[{i}]"
    else:
        assert_array_equal(v1, v2, err_msg=str(path))


def assert_index_equal(i1: Index, i2: Index) -> None:
    """Assert whether two Index-es are equivalent."""
    assert i1.names == i2.names
    assert_frame_equal(i1._parent[list(i1.names)], i2._parent[list(i2.names)])
