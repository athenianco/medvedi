from medvedi import DataFrame


def test_take_empty():
    df = DataFrame({"a": [0, 1]})
    assert df.take([]).empty
