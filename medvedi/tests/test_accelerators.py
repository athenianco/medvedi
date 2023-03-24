import numpy as np
import pytest

from medvedi.accelerators import in1d_str


@pytest.mark.parametrize("kwarg", [{"verbatim": True}, {}])
@pytest.mark.parametrize("invert", [False, True])
def test_in1d_str_flags(kwarg, invert):
    mask = in1d_str(
        np.array(
            [
                b"\x00\x00\x00\x00\x00\x02}Dsrc-d/go-git;",
                b"\x00\x00\x00\x00\x00\x02}Gsrc-d/go-git;",
                b"\x00\x00\x00\x00\x00\x02|\xe7src-d/go-git;",
            ],
        ),
        np.array([b"\x00\x00\x00\x00\x00\x02|\xe7src-d/go-git;"]),
        invert=invert,
        **kwarg,
    )
    assert mask.sum() == 2 if invert else 1


@pytest.mark.parametrize(
    "dtype_left, dtype_right",
    [("S", "S"), ("U", "U"), ("S33", "S100"), ("U17", "U100")],
)
def test_in1d_str_dtype(dtype_left, dtype_right):
    mask = in1d_str(
        np.array(["A", "BB", "CCC"], dtype=dtype_left),
        np.array(["BB", "DDDD"], dtype=dtype_right),
    )
    assert mask.sum() == 1


def test_in1d_str_trailing():
    mask = in1d_str(
        np.array(
            [
                b"\x00\x00\x00\x00\x00\x02}Dsrc-d/go-git",
                b"\x00\x00\x00\x00\x00\x02}Gsrc-d/go-git",
                b"\x00\x00\x00\x00\x00\x02|\xe7src-d/go-git",
            ],
        ),
        np.array([b"\x00\x00\x00\x00\x00\x02|\xe7src-d/go-git\x00\x00"]),
    )
    assert mask.sum() == 1
