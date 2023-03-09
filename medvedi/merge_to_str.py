import numpy as np

mergeable_dtype_kinds = frozenset(("i", "u", "m", "M", "S"))


def merge_to_str(*arrs: np.ndarray, map_nat_to: int | None = None) -> np.ndarray:
    """
    Convert one or more arrays of integers, bytes, datetime64, timedelta64 to S(total_bytes + 1).

    We cannot use arr.byteswap().view("S8") because the trailing zeros get discarded \
    in np.char.add. Thus, we have to pad with ";".

    We copy bytes ("S"), datetime64, timedelta64 verbatim.

    :param map_nat_to: Replace all not-a-times with this value.
    """
    assert len(arrs) > 0
    size = len(arrs[0])
    str_len = sum(arr.dtype.itemsize for arr in arrs) + 1
    arena = np.full((size, str_len), ord(";"), dtype=np.uint8)
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
        if kind == "S":
            arena[:, pos : pos + itemsize] = arr.view(np.uint8)
        else:
            if itemsize > 1:
                col = arr.byteswap()
            else:
                col = arr
            arena[:, pos : pos + itemsize] = col.view(np.uint8).reshape(size, itemsize)
            if (kind == "m" or kind == "M") and map_nat_to is not None:
                arena[col != col, pos : pos + itemsize] = map_nat_to
        pos += itemsize
    return arena.ravel().view(f"S{str_len}")
