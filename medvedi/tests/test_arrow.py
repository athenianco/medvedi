import numpy as np
import pyarrow as pa

from medvedi import DataFrame
from medvedi.testing import assert_frame_equal


def test_arrow_roundtrip():
    df = DataFrame(
        {
            "a": [1, 2, 3],
            "b": [1.5, 2.3, 3.1],
            "c": np.array(["a", "b", "c"], dtype="S1"),
            "d": np.array(["a", "b", "c"], dtype="U1"),
            "e": np.array([10000000, 20000000, 30000000], dtype="datetime64[s]"),
            "f": np.array([1, 2, -3], dtype="timedelta64[s]"),
            "g": [False, True, False],
        },
    )
    table = df.to_arrow()
    assert isinstance(table, pa.Table)
    df_back = DataFrame.from_arrow(table)
    df_back["c"] = df_back["c"].astype("S1")
    df_back["d"] = df_back["d"].astype("U1")
    assert_frame_equal(df, df_back)
