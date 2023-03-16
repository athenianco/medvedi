import pytest

from medvedi import DataFrame


@pytest.mark.parametrize("ignore_index", [False, True])
def test_sample_n(ignore_index):
    df = DataFrame({"a": [0, 1, 2]}, index="a")
    sampled = df.sample(n=2, ignore_index=ignore_index)
    assert len(sampled) == 2
    assert len(sampled.unique("a")) == 2
    assert len(sampled.index.names) == (not ignore_index)


def test_sample_frac():
    df = DataFrame({"a": [0, 1, 2, 3]})
    sampled = df.sample(frac=0.5)
    assert len(sampled) == 2
    assert len(sampled.unique("a")) == 2
