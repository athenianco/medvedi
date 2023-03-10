from medvedi import DataFrame


def test_iterrows_smoke():
    df = DataFrame({"a": [0, 1, 2], "b": ["a", None, 10]})
    for i, (a, b) in enumerate(df.iterrows("a", "b")):
        assert a == df["a"][i]
        assert b == df["b"][i]
