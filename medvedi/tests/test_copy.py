from medvedi import DataFrame


def test_copy_smoke():
    df = DataFrame({"a": [0, 1, 2]}, index="a")
    copy = df.copy()
    assert copy.index.names == ("a",)
    copy["a"][1] = 10
    assert copy["a"][1] == 10
    assert df["a"][1] == 1
