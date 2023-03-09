from medvedi import DataFrame


def test_groupby_1d_int():
    df = DataFrame({"a": [1, 1, 2, 2, 3, 3, 3], "b": [4, 5, 6, 7, 8, 9, 10]})
    a_values, b_values = df["a"], df["b"]
    for i, group in enumerate(df.groupby("a")):
        assert a_values[group[0]] == i + 1
        assert b_values[group].tolist() == ([4, 5], [6, 7], [8, 9, 10])[i]


def test_groupby_2d_int():
    df = DataFrame({"a": [1, 1, 2, 2, 3, 3, 3], "b": [4, 4, 6, 7, 10, 8, 8]})
    grouper = df.groupby("a", "b")
    assert grouper.order.tolist() == [0, 1, 2, 3, 5, 6, 4]
    assert grouper.counts.tolist() == [2, 1, 1, 2, 1]
