import numpy as np
import pytest

from medvedi.accelerators import in1d_str


@pytest.mark.parametrize("kwarg", [{"verbatim": True}, {"skip_leading_zeros": True}])
def test_in1d_str_verbatim(kwarg):
    mask = in1d_str(
        np.array(
            [
                b"\x00\x00\x00\x00\x00\x02}Dsrc-d/go-git;",
                b"\x00\x00\x00\x00\x00\x02}Gsrc-d/go-git;",
                b"\x00\x00\x00\x00\x00\x02|\xe7src-d/go-git;",
            ],
        ),
        np.array([b"\x00\x00\x00\x00\x00\x02|\xe7src-d/go-git;"]),
        **kwarg,
    )
    assert mask.sum() == 1
