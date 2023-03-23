from collections import defaultdict
from dataclasses import dataclass
from itertools import repeat
from typing import (
    Any,
    Collection,
    Hashable,
    Iterable,
    KeysView,
    Literal,
    Mapping,
    Sequence,
    Sized,
    Union,
    overload,
)

import numpy as np
from numpy import typing as npt

try:
    import pyarrow as pa
except ImportError:  # pragma: no cover
    pa = None  # pragma: no cover

from medvedi.accelerators import array_of_objects, in1d_str, is_not_null, is_null, unordered_unique
from medvedi.io import deserialize_df, serialize_df
from medvedi.merge_to_str import merge_to_str, mergeable_dtype_kinds
from medvedi.pure_static import PureStaticDataFrameMethods


class Index:
    """DataFrame multi-level index."""

    __slots__ = ("_parent",)

    def __init__(self, parent: "DataFrame"):
        """Initialize a new instance of `Index` class."""
        self._parent = parent

    def __len__(self) -> int:
        """Return the number of rows in the bound DataFrame."""
        return len(self._parent)

    def __str__(self) -> str:
        """Support str()."""
        return "(" + ", ".join(map(str, self.names)) + ")"

    def __sentry_repr__(self):
        """Support Sentry."""
        return str(self)

    @property
    def nlevels(self) -> int:
        """Return the number of index levels."""
        return len(self._parent._index)

    @property
    def empty(self) -> bool:
        """Return value indicating whether the number of rows is zero."""
        return self._parent.empty

    @property
    def is_unique(self) -> bool:
        """Check whether the index doesn't contain duplicates."""
        columns = self._parent._columns
        if not columns:
            return True
        order, merged = DataFrame._order([columns[c] for c in self._parent._index])
        return len(np.unique(merged[order])) == len(merged)

    @property
    def is_monotonic_increasing(self) -> bool:
        """Check whether the index is sorted in ascending order."""
        return self._is_monotonic("__ge__")

    @property
    def is_monotonic_decreasing(self) -> bool:
        """Check whether the index is sorted in descending order."""
        return self._is_monotonic("__le__")

    def _is_monotonic(self, op: str) -> bool:
        """Check whether the index is sorted."""
        if len(self._parent) <= 1:
            return True
        index = self._parent._index
        columns = self._parent._columns
        last_level = len(index) - 1
        mask: npt.NDArray[np.bool_] | None = None
        for i, level in enumerate(index):
            values = columns[level]
            if mask is not None:
                indexes = np.flatnonzero(mask)
            else:
                indexes = np.arange(len(values) - 1)
            if getattr(values[indexes + 1], op)(values[indexes]).all():
                if i == last_level:
                    break
                if mask is None:
                    mask = np.ones(len(values) - 1, dtype=bool)
                zero_mask = values[indexes + 1] == values[indexes]
                if not zero_mask.any():
                    return True
                mask[mask] &= zero_mask
                continue
            else:
                return False
        return True

    @property
    def name(self) -> Hashable:
        """
        Return the column key that forms the index.

        If the index is multi-level, raise ValueError.
        """
        if len(index := self._parent._index) != 1:
            raise ValueError(f"Index must be 1-D to use .name, have {len(index)}-D")
        return index[0]

    @property
    def names(self) -> tuple[Hashable, ...]:
        """Return the column keys that form the index."""
        return self._parent._index

    @property
    def values(self) -> np.ndarray:
        """Return the only index level if the index is 1-D."""
        if (nlevels := len(self._parent._index)) != 1:
            raise AttributeError(
                "Index.values require a single index level but got "
                f"{nlevels} {self._parent._index}, use get_level_values(0) instead",
            )
        return self._parent[self._parent._index[0]]

    def get_level_values(self, n: int) -> np.ndarray:
        """
        Extract the specified index level values.

        :param n: Index level number, 0-based.
        :return: Numpy array with the referenced column values.
        """
        columns = self._parent._index
        if n >= len(columns) or n < 0:
            raise IndexError(f"Level out of range: {n} >= {len(columns)}")
        return self._parent[columns[n]]

    def levels(self) -> tuple[np.ndarray, ...]:
        """Return all the index levels as numpy arrays."""
        parent = self._parent
        return tuple(parent[c] for c in self._parent._index)

    def duplicated(self, keep: Literal["first", "last"] = "first") -> npt.NDArray[np.bool_]:
        """
        Return the positions of the duplicate index records.

        :param keep: Select all subsequent occurrences of the same record as duplicates if \
                     "first". "last" selects all preceding occurrences of the same record as \
                     duplicates.
        :return: Positions of the duplicates.
        """
        columns = self._parent._columns
        if not columns:
            return np.array([], dtype=bool)
        return self._parent.duplicated(self._parent._index, keep)

    def diff(self, other: "Index") -> npt.NDArray[np.int_]:
        """
        Calculate positions of the unique index values that are not present in the other index.

        :param other: Index with values which should be excluded.
        :return: Positions of unique values in the index not present in the other index.
        """
        if not isinstance(other, Index):
            raise TypeError(f"other must be a medvedi.Index, got {type(other)}")
        columns_self = self._parent._columns
        columns_other = other._parent._columns
        columns_joint = [
            np.concatenate([columns_other[i_other], columns_self[i_self]], casting="unsafe")
            for i_self, i_other in zip(self._parent._index, other._parent._index)
        ]
        _, merged = DataFrame._order(columns_joint, kind="none")
        _, first_found = np.unique(merged, return_index=True)
        return first_found[first_found >= len(other)] - len(other)


@dataclass(frozen=True, slots=True)
class Grouper:
    """
    Grouped DataFrame pointers.

    order: Order of rows in which the DataFrame columns must be sorted to form sequential groups.
    counts: Size of each group.
    """

    order: npt.NDArray[np.int_]
    counts: npt.NDArray[np.int_]

    def reduceat_indexes(self) -> npt.NDArray[np.int_]:
        """
        Calculate the indexes for ufunc.reduceat aggregation.

        Usage:
        >>> df = DataFrame({"a": [3, 3, 3, 2, 2, 1, 1], "b": [9, 10, 7, 8, 4, 5, 6]})
        >>> grouper = df.groupby("a")
        >>> np.add.reduceat(df["b"][grouper.order], grouper.reduceat_indexes())
        array([11, 12, 26])
        """
        if (size := len(self.counts)) == 0:
            return np.array([], dtype=int)
        indexes = np.empty(size, dtype=int)
        indexes[0] = 0
        np.cumsum(self.counts[:-1], out=indexes[1:])
        return indexes

    def group_indexes(self) -> npt.NDArray[np.int_]:
        """
        Calculate the group key indexes.

        Usage:
        >>> df = DataFrame({"a": [1, 1, 2, 2, 3, 3, 3], "b": [4, 5, 6, 7, 8, 9, 10]})
        >>> df.groupby("a").group_indexes()
        array([0, 2, 4])
        >>> df["a"][df.groupby("a").group_indexes()]
        array([1, 2, 3])
        """
        if (size := len(self.counts)) == 0:
            return np.array([], dtype=int)
        indexes = np.empty(size, dtype=int)
        indexes[0] = 0
        np.cumsum(self.counts[:-1], out=indexes[1:])
        return self.order[indexes]

    def __iter__(self):
        """
        Iterate over indexes of each group.

        This is lightweight and avoids materializing column chunks.

        Usage:

        >>> df = DataFrame({"a": [1, 1, 2, 2, 3, 3, 3], "b": [4, 5, 6, 7, 8, 9, 10]})
        >>> a_values, b_values = df["a"], df["b"]
        >>> for group in df.groupby("a"):
        ...     print("a", a_values[group[0]])
        ...     print("b", b_values[group])
        a 1
        b [4 5]
        a 2
        b [6 7]
        a 3
        b [ 8  9 10]
        """
        pos = 0
        order = self.order
        for count in self.counts:
            indexes = order[pos : pos + count]
            pos += count
            yield indexes


class Iloc:
    """Retrieve rows by absolute index."""

    __slots__ = ("_parent",)

    def __init__(self, parent: "DataFrame"):
        """Initialize a new instance of `Iloc` class."""
        self._parent = parent

    def __getitem__(
        self,
        item: int | slice | list[int] | npt.NDArray[np.int_],
    ) -> Union[dict[Hashable, Any], "DataFrame"]:
        """Get the column values at the specified absolute index."""
        length = len(self._parent)
        if isinstance(item, slice):
            item = np.arange(*item.indices(length), dtype=int)
        if isinstance(item, (list, np.ndarray)):
            return self._parent.take(item)
        if not isinstance(item, (int, np.int_)):
            raise TypeError(f"iloc[{item}] is not supported")
        if item >= length or (length + item) < 0:
            raise IndexError(f"iloc[{item}] is out of range [-{length}, {length})")
        return {k: v[item] for k, v in self._parent._columns.items()}


class DataFrame(metaclass=PureStaticDataFrameMethods):
    """
    Core class of the library - a table with several columns.

    The interface tries to mimic Pandas. The major difference is that columns here are raw numpy
    arrays versus Series objects, so we provide common operations like `in_()` or `isnull()`
    directly in the DataFrame methods.

    DataFrame offers an optimized (de)serialization mechanism via `serialize_unsafe()` and
    `deserialize_unsafe()`. This mechanism cannot digest every possible object dtype of
    the columns, however, is 10-100x faster for common objects. If all the columns are non-objects,
    there is little speedup and regular pickling serves well.

    Operations with some column dtypes, e.g. "S", are accelerated by native code.
    """

    __slots__ = ("_columns", "_index", "__weakref__")

    def __init__(
        self,
        data: Any = None,
        columns: Iterable[Hashable] | None = None,
        index: Any = None,
        copy: bool = False,
        dtype: Mapping[Hashable, npt.DTypeLike] | None = None,
        check: bool = True,
    ):
        """
        Initialize a new instance of the `DataFrame` class.

        If the specified arguments are wrong, we raise `TypeError` or `ValueError`.

        :param data: Either a mapping from column names to column values or an iterable with \
                     column values. We do *not* support row iterables.
        :param columns: Override column names. May not be specified if `data` is a mapping.
        :param index: Index column keys or values.
        :param copy: Do not copy any arrays unless have to.
        :param dtype: Mapping from column names to the desired dtypes.
        :param check: Validate the integrity after constructing the instance.
        """
        if dtype is None:
            dtype = {}
        if data is None:
            if columns is None:
                built_columns = {}
            else:
                built_columns = {c: np.array([], dtype=dtype.get(c, object)) for c in columns}
        elif isinstance(data, Mapping):
            if columns is not None:
                raise ValueError("Either `data` or `columns` must be specified")
            built_columns = {k: self._asarray(v, dtype.get(k), copy) for k, v in data.items()}
        else:
            if columns is None:
                columns = repeat(None)
            built_columns = {}
            for i, (arr, name) in enumerate(zip(data, columns)):
                if name is None:
                    name = str(i)
                built_columns[name] = self._asarray(arr, dtype.get(name), copy)
        self._columns: dict[Hashable, np.ndarray] = built_columns
        if check:
            self._check_columns(built_columns)

        self._index: tuple[Hashable, ...] = ()
        if isinstance(index, Index):
            index = index.levels()
        if index is not None and index != ():
            self.set_index(index, inplace=True)

    def __len__(self) -> int:
        """Return the number of rows in the `DataFrame`."""
        if not self._columns:
            return 0
        return len(next(iter(self._columns.values())))

    @overload
    def __getitem__(self, item: Hashable) -> np.ndarray:  # noqa: D105
        ...  # pragma: no cover

    @overload
    def __getitem__(self, item: Collection[Hashable]) -> "DataFrame":  # noqa: D105
        ...  # pragma: no cover

    def __getitem__(self, item: Hashable | Collection[Hashable]) -> Union[np.ndarray, "DataFrame"]:
        """
        Extract the column values by key. Raise `KeyError` if the column doesn't exist.

        If the item is a collection of keys, return the DataFrame with the corresponding columns.

        :param item: Column key.
        :return: Column values, zero-copy. The user is welcome to mutate the internals.
        """
        scalar = isinstance(item, Hashable)
        if scalar and isinstance(item, (tuple, frozenset)) and item not in self._columns:
            scalar = False
        if not scalar:
            assert isinstance(item, Iterable)
            keys = list(item)
            columns = self._columns
            for i in self._index:
                if i not in keys:
                    keys.append(i)
            return DataFrame({k: columns[k] for k in keys}, index=self._index)
        return self._columns[item]  # type: ignore

    def __setitem__(self, key: Hashable, value: Any) -> None:
        """
        Add or replace a column.

        :param key: Column key to add or replace.
        :param value: Numpy array with column values. We set without copying, so the user should \
                      provide an array copy as needed.
        """
        if np.isscalar(value) or value is None:
            if key in self._columns:
                dtype = self._columns[key].dtype
                if dtype.kind == "S" or dtype.kind == "U":
                    if not isinstance(value, (str, bytes)):
                        raise ValueError(f"Invalid scalar type: {type(value)}")
                    dtype = f"{dtype.kind}{len(value)}"
            else:
                dtype = None
            if dtype == object:
                value = array_of_objects(len(self), value)
            else:
                value = np.full(len(self), value, dtype=dtype)
        else:
            value = np.atleast_1d(np.squeeze(np.asarray(value)))
        if value.ndim != 1:
            raise ValueError(f"numpy array must be one-dimensional, got shape {value.shape}")
        if len(self) != len(value) and self._columns:
            raise ValueError(f"new column must have the same length {len(self)}, got {len(value)}")
        self._columns[key] = value

    def __delitem__(self, key: Hashable) -> None:
        """
        Delete a column. If the column doesn't exist, raise `KeyError`.

        :param key: Column key to delete.
        """
        if key in self._index:
            raise ValueError("cannot drop a part of the index, please use set_index()")
        del self._columns[key]

    def __contains__(self, item: Hashable) -> bool:
        """Check whether the column key is present."""
        return item in self._columns

    def __iter__(self) -> Iterable[Hashable]:
        """Iterate over columns."""
        return iter(self._columns.keys())

    def __str__(self) -> str:
        """Support str()."""
        # fmt: off
        return (
            f"DataFrame with {len(self._columns)} columns, {len(self)} rows\n"
            +
            "\n".join(f"{c}: {v.dtype}" for c, v in self._columns.items())
        )
        # fmt: on

    def __repr__(self) -> str:
        """Support repr()."""
        return f"DataFrame({self._columns}, index={self._index})"

    def __sentry_repr__(self) -> str:
        """Format the DataFrame for Sentry."""
        return str(self)

    @property
    def empty(self):
        """Check whether the DataFrame has zero length."""
        if not self._columns:
            return True
        return len(next(iter(self._columns.values()))) == 0

    def iterrows(self, *columns: Hashable) -> zip:
        """
        Iterate column values.

        :param columns: Column keys. We iterate only the specified columns.
        """
        return zip(*(self[c] for c in columns))

    def take(
        self,
        mask_or_indexes: npt.ArrayLike,
        inplace: bool = False,
    ) -> "DataFrame":
        """
        Extract a part of the DataFrame addressed by row indexes or by a row selection mask.

        :param mask_or_indexes: Either a boolean mask or a sequence of indexes.
        :param inplace: Value indicating whether we must update the DataFrame instead of creating \
                        a new one.
        """
        mask_or_indexes = np.asarray(mask_or_indexes)

        if not inplace:
            return type(self)(
                {
                    k: v[mask_or_indexes] if len(mask_or_indexes) else v[:0]
                    for k, v in self._columns.items()
                },
                index=self._index,
                check=False,
            )
        columns = self._columns
        for k, v in columns.items():
            columns[k] = v[mask_or_indexes]
        return self

    def copy(self, shallow: bool = False) -> "DataFrame":
        """Produce a deep copy of the DataFrame.

        :param shallow: We copy underlying numpy arrays by default; when `shallow=True`, \
                        the produced DataFrame clone shares arrays with the origin.
        """
        df = type(self)()
        df._columns = {c: v if shallow else v.copy() for c, v in self._columns.items()}
        df._index = self._index
        return df

    def sample(
        self,
        n: int | None = None,
        frac: float | int | None = None,
        replace: bool = False,
        weights: npt.NDArray[np.float_] | None = None,
        ignore_index: bool = False,
    ):
        """
        Return a random sample of rows in a new DataFrame.

        :param n: Number of rows to sample. Cannot be passed together with `frac`.
        :param frac: Ratio of rows to sample. 1 == length. Cannot be passed together with `n`.
        :param replace: Sample with replacement.
        :param weights: The probabilities associated with each row.
        :param ignore_index: Reset the index in the sampled DataFrame.
        """
        if (frac is None) == (n is None):
            raise ValueError("Must define one and only one of `n` and `frac`")
        if frac is not None:
            n = int(len(self) * frac)
        indexes = np.random.choice(np.arange(len(self), dtype=int), n, replace=replace, p=weights)
        df = self.take(indexes)
        if ignore_index:
            df.reset_index(inplace=True)
        return df

    def astype(
        self,
        dtype: np.dtype | str | type | Mapping[Hashable, np.dtype | str | type],
        copy: bool = True,
        errors: Literal["raise", "ignore"] = "raise",
    ) -> "DataFrame":
        """
        Convert the data type of columns.

        :param dtype: Global data type for all the columns or mapping from column keys to the new \
                      data types.
        :param copy: Value indicating whether the operation copies the original DataFrame.
        :param errors: Either "raise" (default) or "ignore". If set to "ignore", silently skip \
                       columns which failed dtype conversion.
        :return: Converted DataFrame.
        """
        if not isinstance(dtype, Mapping):
            dtype = {k: dtype for k in self._columns}

        df = self.copy() if copy else self

        columns = df._columns
        for k, k_dt in dtype.items():
            try:
                columns[k] = columns[k].astype(k_dt, copy=False)
            except ValueError as e:
                if errors == "raise":
                    raise e from None

        return df

    def explode(self, column: Hashable, ignore_index: bool = False) -> "DataFrame":
        """
        Flatten a column with list-like elements, replicating other column values.

        :param column: Column with list-likes inside that should be transformed.
        :param ignore_index: Value indicating whether the index must be reset.
        :return: DataFrame with >= rows than the origin.
        """
        values = self[column]
        if values.dtype != object:
            return self.reset_index() if ignore_index else self.copy()
        checked_types = (tuple, list, np.ndarray)
        repeat_counts = np.ones(len(values), dtype=int)
        new_col: list[Any] = []
        for i, v in enumerate(values):
            if isinstance(v, checked_types):
                repeat_counts[i] = len(v)
                new_col.extend(v)
            else:
                new_col.append(v)
        columns = {
            k: np.repeat(v, repeat_counts) if k != column else np.asarray(new_col)
            for k, v in self._columns.items()
        }
        df = DataFrame(columns, index=self._index if not ignore_index else None)
        return df

    @property
    def index(self) -> Index:
        """Return the associated index."""
        return Index(self)

    @property
    def columns(self) -> tuple[Hashable, ...]:
        """Return the column keys."""
        return tuple(self._columns)

    @property
    def dtype(self) -> dict[Hashable, np.dtype]:
        """Return the mapping from column keys to value dtypes."""
        return {k: v.dtype for k, v in self._columns.items()}

    def sort_values(
        self,
        by: Hashable | list[Hashable],
        *,
        ascending: bool = True,
        inplace: bool = False,
        kind: Literal["quicksort", "stable"] | None = None,
        na_position: Literal["first", "last"] = "last",
        ignore_index: bool = False,
        non_negative_hint: bool = False,
    ) -> "DataFrame":
        """
        Order the DataFrame by values of one or more columns defined by `by`.

        :param by: One or more column keys by which we must sort.
        :param ascending: Value indicating whether the sorting order is from the smallest to \
                          the biggest (default) or from the biggest to the smallest.
        :param inplace: Value indicating whether we must update the current DataFrame instead of \
                        returning a copy.
        :param kind: Sorting algorithm.
        :param na_position: Where we must place the nulls (NaNs, NaTs, etc.).
        :param ignore_index: Value indicating whether the index in the resulting DataFrame will \
                             be empty.
        :param non_negative_hint: The column values referenced by `by` are all non-negative. \
                                  This enables low-level optimization in multi-column mode.
        :return: Sorted DataFrame.
        """
        if not isinstance(by, (tuple, list)):
            by = (by,)
        elif len(by) == 0:
            raise ValueError("must specify at least one column")

        df = self if inplace else self.copy()
        if ignore_index:
            df._index = ()

        order, _ = self._order(
            [self[c] for c in by],
            kind,
            strict=not non_negative_hint,
            na_position=na_position if ascending else "first" if na_position == "last" else "last",
        )
        if not ascending:
            order = order[::-1]

        for k, v in df._columns.items():
            df._columns[k] = v[order]

        return df

    def sort_index(
        self,
        level: int | Sequence[int] | None = None,
        ascending: bool = True,
        inplace: bool = False,
        kind: Literal["quicksort", "stable"] | None = None,
        na_position: Literal["first", "last"] = "last",
        ignore_index: bool = False,
        non_negative_hint: bool = False,
    ) -> "DataFrame":
        """
        Sorts according to the index values.

        :param level: Index level index or several index levels. None means using the whole index.
        :param ascending: Value indicating whether the sorting order is from the smallest to \
                          the biggest (default) or from the biggest to the smallest.
        :param inplace: Value indicating whether we must update the current DataFrame instead of \
                        returning a copy.
        :param kind: Sorting algorithm.
        :param na_position: Where we must place the nulls (NaNs, NaTs, etc.).
        :param ignore_index: Value indicating whether the index in the resulting DataFrame will \
                             be reset.
        :param non_negative_hint: The index values referenced by `level` are all non-negative. \
                                  This enables low-level optimization in multi-level mode.
        :return: Sorted DataFrame.
        """
        index = self._index
        levels: Hashable | list[Hashable]
        if level is None:
            levels = index
        elif isinstance(level, (tuple, list)):
            levels = [index[i] for i in level]
        else:
            if not isinstance(level, (int, np.int_)):
                raise TypeError(f"Invalid index level type: {type(level)}")
            levels = index[level]
        return self.sort_values(
            by=levels,
            ascending=ascending,
            inplace=inplace,
            kind=kind,
            na_position=na_position,
            ignore_index=ignore_index,
            non_negative_hint=non_negative_hint,
        )

    def set_index(self, index: Any, inplace=False, drop=False) -> "DataFrame":
        """
        Install a new index to the DataFrame.

        :param index: One or more column keys to form the multi-level index.
        :param inplace: Value indicating whether we must update the current DataFrame instead of \
                        returning a copy.
        :param drop: Erase the old indexed columns, except those mentioned in the new index.
        """
        df = self if inplace else self.copy()
        old_index = df._index

        if isinstance(index, list) or (isinstance(index, np.ndarray) and index.ndim > 1):
            index = tuple(index)

        if isinstance(index, tuple) and len(index) == 0:
            df._index = index
        elif isinstance(index, np.ndarray):
            if "_index0" in df and not drop:
                raise ValueError('Cannot set an unnamed index "_index0": column already exists')
            df._index = ("_index0",)
            df[df._index[0]] = index
        else:
            if not isinstance(index, tuple):
                index = (index,)
            if isinstance(index[0], (list, np.ndarray)):
                index_names = []
                for i, level in enumerate(index):
                    name = f"_index{i}"
                    if name in df and not drop:
                        raise ValueError(
                            f'Cannot set an unnamed index "{name}": column already exists',
                        )
                    df[name] = np.asarray(level)
                    index_names.append(name)
                df._index = tuple(index_names)
            else:
                for c in index:
                    if c not in self._columns:
                        raise KeyError(
                            f"index '{c}' must be one of the existing columns "
                            f"{list(self._columns)}",
                        )
                df._index = index

        if old_index != () and drop:
            new_index = df._index
            for c in old_index:
                if c not in new_index:
                    del df._columns[c]

        return df

    def reset_index(self, inplace=False, drop=False) -> "DataFrame":
        """
        Erase the current index.

        :param inplace: Value indicating whether we must update the current DataFrame instead of \
                        returning a copy.
        :param drop: Delete the columns referenced by `index`.
        """
        df = self if inplace else self.copy()

        if df._index and drop:
            for c in df._index:
                del df._columns[c]

        df._index = ()
        return df

    def rename(
        self,
        columns: Mapping[Hashable, Hashable],
        inplace: bool = False,
        errors: Literal["raise", "ignore"] = "ignore",
    ) -> "DataFrame":
        """
        Rename columns in bulk.

        :param columns: Mapping from old to new column keys.
        :param inplace: Value indicating whether we must update the current DataFrame instead of \
                        returning a copy.
        :param errors: "ignore" suppresses missing columns, "raise" forwards KeyError.
        :return: DataFrame with renamed columns.
        """
        df = self if inplace else self.copy()
        if not isinstance(columns, Mapping):
            raise TypeError(f"columns must be a mapping, got {type(columns)}")
        new_columns = {}
        for old, new in columns.items():
            try:
                new_columns[new] = df._columns[old]
            except KeyError as e:
                if errors == "raise":
                    raise e from None
                continue
            del df._columns[old]
        for key, val in new_columns.items():
            df[key] = val

        df._index = tuple(columns.get(i, i) for i in df._index)

        return df

    def drop_duplicates(
        self,
        subset: Hashable | Iterable[Hashable] | None = None,
        keep: Literal["first", "last"] = "first",
        inplace: bool = False,
        ignore_index: bool = False,
    ) -> "DataFrame":
        """
        Remove duplicate rows from the DataFrame.

        We consider duplicates those rows which have the same values of columns in `subset`.

        :param subset: Column key or several column keys. None means all the columns.
        :param keep: "first" leaves the first encounters, while "last" leaves the last encounters.
        :param inplace: Update the current DataFrame instead of returning a deduplicated copy.
        :param ignore_index: Value indicating whether the resulting index will be reset.
        :return: DataFrame with unique rows.
        """
        df = self if inplace else self.copy()
        first_found = df._unique(subset, keep)
        if ignore_index:
            df._index = ()
        if len(first_found) < len(df):
            return df.take(first_found, inplace=inplace)
        return df

    def duplicated(
        self,
        subset: Hashable | Iterable[Hashable] | None = None,
        keep: Literal["first", "last"] = "first",
    ) -> npt.NDArray[np.bool_]:
        """
        Return the boolean mask of duplicate rows.

        :param subset: Column key or several column keys. None means all the columns.
        :param keep: "first" leaves the first encounters, while "last" leaves the last encounters.
        :return: Boolean mask with True standing at the positions of duplicates.
        """
        first_found = self._unique(subset, keep)
        mask = np.ones(len(self), dtype=bool)
        mask[first_found] = False
        return mask

    def groupby(self, *by: Hashable | npt.ArrayLike) -> Grouper:
        """
        Group rows by one or more columns in `by`.

        Complexity: N log(N) where `N = len(self)`.

        Usage:

        >>> df = DataFrame({"a": [1, 1, 2, 2, 3, 3, 3], "b": [4, 5, 6, 7, 8, 9, 10]})
        >>> a_values, b_values = df["a"], df["b"]
        >>> for group in df.groupby("a"):
        ...     print("a", a_values[group[0]])
        ...     print("b", b_values[group])
        a 1
        b [4 5]
        a 2
        b [6 7]
        a 3
        b [ 8  9 10]
        """
        seed = []
        for c in by:
            if isinstance(c, (list, np.ndarray)):
                c = np.asarray(c)
                if c.shape != (len(self),):
                    raise ValueError(f"shape mismatch {c.shape} != ({len(self)},)")
                seed.append(c)
            else:
                if not isinstance(c, Hashable):
                    raise TypeError(f"Invalid column key: {c}")
                seed.append(self[c])
        order, values = self._order(seed, "stable")
        _, counts = np.unique(values[order], return_counts=True)
        return Grouper(order, counts)

    def isin(
        self,
        column: Hashable,
        haystack: npt.ArrayLike | set | frozenset | dict | KeysView,
        assume_unique: bool = False,
        invert: bool = False,
    ) -> npt.NDArray[np.bool_]:
        """
        Check whether each element in `column` is a member of `haystack` array.

        :param column: Column key.
        :param haystack: Array where we search, must be of the same dtype as `column`.
        :param assume_unique: Both `column` and `haystack` doesn't have duplicates.
        :param invert: Value indicating whether the result must be an inverse: check that each \
                       element in `column` is *not* present in `haystack`.
        :return: Boolean mask, same shape as `column`.
        """
        values = self[column]
        if isinstance(haystack, (set, frozenset, dict, KeysView)):
            # do not shoot in the foot
            haystack = _NumpyIterableProxy(haystack)
        haystack = np.asarray(haystack, dtype=values.dtype)
        kind = values.dtype.kind
        if kind == "S" or kind == "U":
            mask = in1d_str(values, haystack)
            if invert:
                mask = ~mask
            return mask
        return np.in1d(values, haystack, assume_unique=assume_unique, invert=invert)

    def unique(self, column: Hashable, unordered: bool = False) -> np.ndarray:
        """
        Return the unique elements of `column`.

        :param column: Column key to select.
        :param unordered: Value indicating whether the result must be sorted or the caller \
                          doesn't care.
        :return: Unique column values.
        """
        values = self[column]
        if unordered:
            try:
                return unordered_unique(values)
            except AssertionError:
                pass
        return np.unique(values)

    def isnull(self, column: Hashable) -> npt.NDArray[np.bool_]:
        """
        Calculate the boolean mask indicating whether each element of `column` is null.

        `dtype=object`: we check `is None`.
        `dtype=float`: we check `== NaN`.
        `dtype=timedelta64 or datetime64`: we check `== NaT`.
        """
        values = self[column]
        if values.dtype == object:
            return is_null(values)
        kind = values.dtype.kind
        if kind == "m" or kind == "M" or kind == "f":
            return values != values
        return np.zeros(len(values), dtype=bool)

    def notnull(self, column: Hashable) -> npt.NDArray[np.bool_]:
        """
        Calculate the boolean mask indicating whether each element of `column` is not null.

        `dtype=object`: we check `is not None`.
        `dtype=float`: we check `!= NaN`.
        `dtype=timedelta64 or datetime64`: we check `!= NaT`.
        """
        values = self[column]
        if values.dtype == object:
            return is_not_null(values)
        kind = values.dtype.kind
        if kind == "m" or kind == "M" or kind == "f":
            return values == values
        return np.ones(len(values), dtype=bool)

    def nonemin(self, column: Hashable) -> Any | None:
        """Return the minimum value along the given column, ignoring NaN-s, or None if \
        it doesn't exist."""
        values = self[column]
        if values.dtype == object:
            values = values[is_not_null(values)]
        if len(values) == 0:
            return None
        m = np.nanmin(values)
        if m != m:
            return None
        return m

    def nonemax(self, column: Hashable) -> Any | None:
        """Return the maximum value along the given column, ignoring NaN-s, or None if \
        it doesn't exist."""
        values = self[column]
        if values.dtype == object:
            values = values[is_not_null(values)]
        if len(values) == 0:
            return None
        m = np.nanmax(values)
        if m != m:
            return None
        return m

    def fillna(
        self,
        value: Any,
        column: Hashable | None = None,
        inplace: bool = False,
    ) -> "DataFrame":
        """
        Set null-like elements (None, NaN, NaT) to `value`.

        :param value: Scalar replacement value for null-likes.
        :param column: Affected column key. The default is all columns.
        :param inplace: Update the current DataFrame instead of returning a copy.
        :return: Filled DataFrame.
        """
        df = self if inplace else self.copy()
        filled: Iterable[Hashable]
        if column is None:
            filled = df._columns
        else:
            filled = (column,)
        columns = df._columns
        for c in filled:
            v = columns[c]
            v[df.isnull(c)] = value
        return df

    @property
    def iloc(self) -> Iloc:
        """Get row by absolute index."""
        return Iloc(self)

    def serialize_unsafe(self, alloc: object | None = None) -> bytes:
        """
        Constrained serialization, doesn't work for arbitrary object columns.

        The binary format is unstable and is only suitable for short-term caching.
        """
        return serialize_df(self, alloc)

    @classmethod
    def deserialize_unsafe(cls, buffer: bytes) -> "DataFrame":
        """Reverse of serialize_unsafe(): load the DataFrame from buffer."""
        return deserialize_df(buffer)

    def to_arrow(self) -> "pa.Table":
        """Convert to a PyArrow Table."""
        if pa is None:
            raise ImportError("pyarrow")  # pragma: no cover
        return pa.table(
            [pa.array(arr) for arr in self._columns.values()],
            names=list(self.columns),
        )

    @classmethod
    def from_arrow(cls, table: "pa.Table") -> "DataFrame":
        """Convert a PyArrow Table to DataFrame."""
        if pa is None:
            raise ImportError("pyarrow")  # pragma: no cover
        if not isinstance(table, pa.Table):
            raise TypeError(f"Expected a PyArrow Table, got {type(table)}")
        return DataFrame({k: table[k].to_numpy() for k in table.column_names})

    @classmethod
    def _concat(
        cls,
        *dfs: "DataFrame",
        ignore_index: bool = False,
        copy: bool = False,
        strict: bool = True,
    ) -> "DataFrame":
        if len(dfs) == 0:
            return cls()
        if not isinstance(dfs[0], cls):
            raise TypeError(f"Can only concatenate medvedi.DataFrame-s, got {type(dfs[0])}")
        if len(dfs) == 1:
            if not copy:
                return dfs[0]
            return dfs[0].copy()
        index = dfs[0]._index
        columns = dfs[0]._columns.keys()
        column_chunks = defaultdict(list)
        empty_column_dtypes: dict[Hashable, np.dtype] = {}
        pos = 0
        for df in dfs:
            if not isinstance(df, cls):
                raise TypeError(f"Can only concatenate medvedi.DataFrame-s, got {type(df)}")
            if strict and df._columns.keys() != columns:
                raise ValueError(f"Columns must match: {columns} vs. {df._columns.keys()}")
            if df._index != index and not ignore_index:
                raise ValueError(f"Indexes must match: {index} vs. {df._index}")
            if df_len := len(df):
                arange = None if strict else np.arange(pos, pos + df_len)
                pos += df_len
                for k, v in df._columns.items():
                    column_chunks[k].append((v, arange))
            else:
                for k, v in df._columns.items():
                    empty_column_dtypes.setdefault(k, v.dtype)
                    column_chunks.setdefault(k, [])
        concat_columns = {}
        for key, chunks in column_chunks.items():
            if not chunks:
                column = cls._empty_array(pos, empty_column_dtypes[key])
            else:
                values = np.concatenate([arr for arr, _ in chunks], casting="unsafe")
                if strict:
                    column = values
                else:
                    indexes = np.concatenate([i for _, i in chunks])
                    if len(indexes) == pos:
                        column = values
                    else:
                        column = cls._empty_array(pos, values.dtype)
                        column[indexes] = values
            concat_columns[key] = column
        concat_df = cls()
        concat_df._index = index if not ignore_index else ()
        concat_df._columns = concat_columns
        return concat_df

    @classmethod
    def _join(
        cls,
        *dfs: "DataFrame",
        how: Literal["left", "right", "inner", "outer"] = "left",
        suffixes: tuple[str | None, ...] = (),
        copy: bool = False,
    ):
        if how == "right":
            return cls.join(*dfs[::-1], how="left", suffixes=suffixes, copy=copy)
        if how not in ("left", "inner", "outer"):
            raise ValueError(
                f"`how` must be either 'left', 'right', 'inner', or 'outer', got {how}",
            )
        if len(dfs) == 0:
            return cls()
        if len(dfs) > 255:
            raise ValueError(f"Cannot join more than 255 DataFrame-s, got {len(dfs)}")
        for df in dfs:
            if not isinstance(df, cls):
                raise TypeError(f"Can only concatenate medvedi.DataFrame-s, got {type(df)}")
        if not isinstance(suffixes, tuple):
            raise TypeError(f"`suffixes` must be a tuple, got {type(suffixes)}")
        if suffixes:
            if len(suffixes) != len(dfs):
                raise ValueError(
                    "`suffixes` must have the same length as the number of joined DataFrame-s: "
                    f"{len(suffixes)} vs. {len(dfs)}",
                )
        else:
            suffixes = tuple(None for _ in dfs)
        if len(dfs) == 1:
            if not copy:
                return dfs[0]
            return dfs[0].copy()
        indexes = [dfs[0]._index]
        dtypes = [dfs[0]._columns[c].dtype for c in indexes[0]]
        first_empty = dfs[0].empty
        for df in dfs[1:]:
            if (
                [df._columns[c].dtype for c in df._index] != dtypes
                and not df.empty
                and not first_empty
            ):
                raise ValueError(f"Incompatible indexes: {df._index} vs. {indexes[0]}")
            indexes.append(df._index)
        if indexes[0] == ():
            raise ValueError("Joining requires an index")
        transposed_resolved_indexes_builder: list[list[np.ndarray]] = [[] for _ in indexes[0]]
        for index, df in zip(indexes, dfs):
            if not df.empty:
                for i, c in enumerate(index):
                    transposed_resolved_indexes_builder[i].append(df[c])
        transposed_resolved_indexes: list[np.ndarray] = [
            np.concatenate(vals, casting="unsafe") if vals else dfs[0]._columns[c]
            for vals, c in zip(transposed_resolved_indexes_builder, indexes[0])
        ]
        del transposed_resolved_indexes_builder
        if len(transposed_resolved_indexes) == 1:
            merged_index = transposed_resolved_indexes[0]
        else:
            if not ({c.dtype.kind for c in transposed_resolved_indexes} - mergeable_dtype_kinds):
                merged_index = merge_to_str(*transposed_resolved_indexes)
            else:
                mapped_indexes = [
                    np.unique(c, return_inverse=True)[1] for c in transposed_resolved_indexes
                ]
                merged_index = merge_to_str(*mapped_indexes)
        _, index_map, inverse_map = np.unique(merged_index, return_index=True, return_inverse=True)
        first_length = len(dfs[0])
        if len(
            np.unique(
                merge_to_str(
                    inverse_map[first_length:],
                    np.repeat(
                        np.arange(len(dfs) - 1),
                        np.fromiter((len(df) for df in dfs[1:]), int, len(dfs) - 1),
                    ),
                )
                if len(dfs) > 1
                else inverse_map[first_length:],
            ),
        ) < (len(inverse_map) - first_length):
            raise NotImplementedError  # fallback pairwise
        _, repeats = np.unique(inverse_map[:first_length], return_counts=True)
        if (repeats == 1).all():
            repeats = None
        leave_mask = None
        if how == "left":
            leave_mask = index_map < first_length
        elif how == "inner":
            encounters = np.zeros(len(index_map), dtype=np.uint8)
            pos = 0
            for df in dfs:
                df_len = len(df)
                encounters[inverse_map[pos : pos + df_len]] += 1
                pos += df_len
            leave_mask = encounters == len(dfs)
            if repeats is not None:
                augmented_repeats = np.empty(len(index_map), int)
                first_mask = index_map < first_length
                augmented_repeats[first_mask] = repeats
                repeats = augmented_repeats[leave_mask & first_mask]
                first_length = repeats.sum()
        if leave_mask is not None and not leave_mask.all():
            index_map = index_map[leave_mask]
            inverse_index_map = np.full(len(leave_mask), -1, dtype=int)
            inverse_index_map[leave_mask] = np.arange(leave_mask.sum())
            inverse_map = inverse_index_map[inverse_map]
            del inverse_index_map
        joined_columns = {
            i: c[index_map if repeats is None else np.repeat(index_map, repeats)]
            if len(index_map)
            else c[:0]
            for i, c in zip(indexes[0], transposed_resolved_indexes)
        }
        pos = 0
        mask = None
        left_added_columns = set()
        for i, (df, suffix) in enumerate(zip(dfs, suffixes)):
            df_len = len(df)
            values_order = inverse_map[pos : pos + df_len]
            pos += df_len
            if must_mask := (how != "outer" and (i > 0 or how == "inner")):
                mask = values_order >= 0
                if i == 0 and repeats is not None:
                    mask = np.flatnonzero(mask)[np.argsort(values_order[mask], kind="stable")]
                    values_order = slice(None)
                else:
                    values_order = values_order[mask]
            elif must_mask := (repeats is not None):
                mask = np.argsort(values_order, kind="stable")
                values_order = slice(None)
            for c, values in df._columns.items():
                if c in df._index:
                    continue
                if suffix is not None:
                    c = str(c) + suffix
                if i > 0 and repeats is not None:
                    left_added_columns.add(c)
                if c not in joined_columns:
                    joined_columns[c] = joined_values = cls._empty_array(
                        len(index_map) if (i > 0 or repeats is None) else first_length,
                        values.dtype,
                    )
                else:
                    joined_values = joined_columns[c]
                if must_mask:
                    values = values[mask]
                joined_values[values_order] = values

        if repeats is not None:
            for c in left_added_columns:
                joined_columns[c] = np.repeat(joined_columns[c], repeats)

        joined = DataFrame()
        joined._index = indexes[0]
        joined._columns = joined_columns
        return joined

    @staticmethod
    def _order(
        by: Sequence[np.ndarray],
        kind: Literal["quicksort", "stable", "none"] | None = None,
        strict: bool = False,
        na_position: Literal["first", "last"] = "last",
    ) -> tuple[npt.NDArray[np.int_], np.ndarray]:
        if len(by) == 1:
            if kind == "none":
                return np.array([], int), by[0]
            result = np.argsort(by[0], kind=kind)
            if na_position == "first":
                if (na_count := (by[0] != by[0]).sum()) > 0:
                    result = np.concatenate([result[-na_count:], result[:-na_count]])
            return result, by[0]
        if not strict and not ({c.dtype.kind for c in by} - mergeable_dtype_kinds):
            merged = merge_to_str(*by)
            return np.argsort(merged, kind=kind) if kind != "none" else np.array([], int), merged
        mapped_bys = []
        for c in by:
            unique_values, inverse_indexes = np.unique(c, return_inverse=True)
            if (
                na_position == "first"
                and len(unique_values)
                and unique_values[-1] != unique_values[-1]
            ):
                inverse_indexes += 1
                inverse_indexes[inverse_indexes == len(unique_values)] = 0
            mapped_bys.append(inverse_indexes)
        merged = merge_to_str(*mapped_bys)
        return np.argsort(merged, kind=kind) if kind != "none" else np.array([], int), merged

    def _unique(
        self,
        subset: Hashable | Iterable[Hashable] | None,
        keep: Literal["first", "last"],
    ) -> npt.NDArray[np.int_]:
        if subset is None:
            subset = self._columns.keys()
            if not subset:
                return np.array([], int)
        elif isinstance(subset, Hashable) and subset in self._columns:
            subset = (subset,)
        if not isinstance(subset, Iterable):
            raise TypeError(f"subset of columns is not recognized: {type(subset)}")
        if not (by := [self[c] for c in subset]):
            raise ValueError(f"subset of columns cannot be empty, got {subset}")
        order, merged = self._order(by, "stable")
        if keep == "last":
            order = order[::-1]
        _, first_found = np.unique(merged[order], return_index=True)
        return order[first_found]

    @staticmethod
    def _empty_array(length: int, dtype: np.dtype) -> np.ndarray:
        if dtype == object:
            return np.empty(length, dtype=dtype)
        kind = dtype.kind
        if kind == "f" or kind == "m" or kind == "M":
            return np.full(length, None, dtype)
        return np.zeros(length, dtype=dtype)

    @staticmethod
    def _asarray(value: Sequence | np.ndarray, dtype: npt.DTypeLike, copy: bool) -> np.ndarray:
        if dtype == object and not isinstance(value, np.ndarray):
            r = np.empty(len(value), dtype)
            r[:] = value
        else:
            r = np.array(value, dtype, copy=copy)
        return r

    @staticmethod
    def _check_columns(columns: dict[Hashable, np.ndarray]) -> None:
        length = None
        for k, v in columns.items():
            if length is None:
                length = len(v)
            elif length != len(v):
                raise ValueError(f"all columns must have equal length: {k}")
            if v.ndim != 1:
                raise ValueError(f"column {k} must be 1-dimensional, got {v.ndim}")


class _NumpyIterableProxy(Sequence[Any]):
    __slots__ = ("obj",)

    def __init__(self, obj):
        assert isinstance(obj, (Iterable, Sized))
        self.obj = obj

    def __len__(self):
        return len(self.obj)

    def __getitem__(self, item):
        raise AssertionError("numpy shouldn't call this")

    def __iter__(self):
        return iter(self.obj)
