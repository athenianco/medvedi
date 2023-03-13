from numpy.testing import assert_array_equal
import pytest

from medvedi import DataFrame


@pytest.mark.parametrize("inplace", [False, True])
def test_rename_inplace(inplace):
    df = DataFrame({"a": [0, 1, 2]}, index="a")
    new_df = df.rename({"a": "b"}, inplace=inplace)
    assert new_df.columns == ("b",)
    assert_array_equal(new_df["b"], [0, 1, 2])
    assert new_df.index.names == ("b",)
    if inplace:
        assert df is new_df
    else:
        new_df["b"][0] = 10
        assert df["a"][0] == 0


def test_rename_errors():
    df = DataFrame({"a": [0, 1, 2]})
    with pytest.raises(KeyError):
        df.rename({"a": "b", "b": "c"}, errors="raise")
    new_df = df.rename({"a": "b", "b": "c"}, errors="ignore")
    assert new_df.columns == ("b",)
    assert_array_equal(new_df["b"], [0, 1, 2])


def test_rename_mapping():
    df = DataFrame({"a": [0, 1, 2]})
    with pytest.raises(TypeError):
        df.rename([("a", "b")])
