import numpy as np

mergeable_dtype_kinds = frozenset(("i", "u", "m", "M", "S"))


def merge_to_str(*arrs: np.ndarray) -> np.ndarray:
    """
    Convert one or more arrays of integers, bytes, datetime64, timedelta64 to \
    S(total_bytes + (last array is not "S")).

    We cannot use arr.byteswap().view("S8") because the trailing zeros get discarded \
    in np.char.add. Thus, we have to pad with ";" unless the last array is bytes.

    We copy bytes ("S"), datetime64, timedelta64 verbatim.
    """
    assert len(arrs) > 0
    size = len(arrs[0])
    lock = arrs[-1].dtype.kind != "S"
    str_len = sum(arr.dtype.itemsize for arr in arrs) + lock
    arena = np.full((size, str_len), ord(";") if lock else 0, dtype=np.uint8)
    pos = 0
    for arr in arrs:
        kind = arr.dtype.kind
        if kind not in mergeable_dtype_kinds:
            raise ValueError(
                f"array's dtype.kind {arr.dtype} must be one of "
                f"{', '.join(mergeable_dtype_kinds)}",
            )
        if len(arr) != size:
            raise ValueError(f"all arrays must have the same length: {len(arr)} != {size}")
        itemsize = arr.dtype.itemsize
        if itemsize > 1 and kind != "S":
            col = arr.byteswap()
        else:
            col = arr
        arena[:, pos : pos + itemsize] = col.view(np.uint8).reshape(size, itemsize)
        pos += itemsize
    return arena.ravel().view(f"S{str_len}")
