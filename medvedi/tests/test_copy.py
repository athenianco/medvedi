import pytest

from medvedi import DataFrame


@pytest.mark.parametrize("shallow", [False, True])
def test_copy_depth(shallow):
    df = DataFrame({"a": [0, 1, 2]}, index="a")
    copy = df.copy(shallow=shallow)
    assert copy.index.names == ("a",)
    copy["a"][1] = 10
    assert copy["a"][1] == 10
    assert df["a"][1] == 10 if shallow else 1
