from dataclasses import dataclass
from itertools import repeat
from typing import Any, Hashable, Iterable, Literal, Mapping, Sequence, Union

import numpy as np
from numpy import typing as npt

try:
    import pyarrow as pa
except ImportError:  # pragma: no cover
    pa = None  # pragma: no cover

from medvedi.accelerators import in1d_str, is_not_null, is_null, unordered_unique
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
    def names(self) -> tuple[Hashable, ...]:
        """Return the column keys that form the index."""
        return self._parent._index

    @property
    def values(self) -> np.ndarray:
        """Return the only index level if the index is 1-D."""
        if len(self._parent._index) != 1:
            raise AttributeError(
                "Index.values require a single index level, use get_level_values() instead",
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
        >>> df = DataFrame({"a": [1, 1, 2, 2, 3, 3, 3], "b": [4, 5, 6, 7, 8, 9, 10]})
        >>> np.add.reduceat(df["b"], df.groupby("a").reduceat_indexes())
        array([ 9, 13, 27])
        """
        indexes = np.empty(len(self.counts), dtype=int)
        indexes[0] = 0
        np.cumsum(self.counts[:-1], out=indexes[1:])
        return indexes

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
        :param check: Validate the integrity after constructing the instance.
        """
        if data is None:
            if columns is None:
                built_columns = {}
            else:
                built_columns = {c: np.array([], dtype=object) for c in columns}
        elif isinstance(data, Mapping):
            if columns is not None:
                raise ValueError("Either `data` or `columns` must be specified")
            built_columns = {k: np.array(v, copy=copy) for k, v in data.items()}
        else:
            if columns is None:
                columns = repeat(None)
            built_columns = {}
            for i, (arr, name) in enumerate(zip(data, columns)):
                if name is None:
                    name = str(i)
                built_columns[name] = np.array(arr, copy=copy)
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

    def __getitem__(self, item: Hashable) -> np.ndarray:
        """
        Extract the column values by key. Raise `KeyError` if the column doesn't exist.

        :param item: Column key.
        :return: Column values, zero-copy. The user is welcome to mutate the internals.
        """
        return self._columns[item]

    def __setitem__(self, key: Hashable, value: npt.ArrayLike) -> None:
        """
        Add or replace a column.

        :param key: Column key to add or replace.
        :param value: Numpy array with column values. We set without copying, so the user should \
                      provide an array copy as needed.
        """
        value = np.squeeze(np.asarray(value))
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
        del self._columns[key]

    def __contains__(self, item: Hashable) -> bool:
        """Check whether the column key is present."""
        return item in self._columns

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
                {k: v[mask_or_indexes] for k, v in self._columns.items()},
                index=self._index,
                check=False,
            )
        columns = self._columns
        for k, v in columns.items():
            columns[k] = v[mask_or_indexes]
        return self

    def copy(self) -> "DataFrame":
        """Produce a deep copy of the DataFrame."""
        df = type(self)()
        df._columns = {c: v.copy() for c, v in self._columns.items()}
        df._index = self._index
        return df

    @property
    def index(self) -> Index:
        """Return the associated index."""
        return Index(self)

    @property
    def columns(self) -> tuple[Hashable, ...]:
        """Return the column keys."""
        return tuple(self._columns)

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
                             be reset.
        :param non_negative_hint: The column values referenced by `by` are all non-negative. \
                                  This enables low-level optimization in multi-column mode.
        """
        df = self.copy() if not inplace else self

        if not ignore_index and df._index == ():
            df["_index0"] = np.arange(len(self), dtype=int)
            df._index = ("_index0",)

        if not isinstance(by, (tuple, list)):
            by = (by,)

        order, _ = self._order(
            [self[c] for c in by],
            kind,
            strict=not non_negative_hint,
            na_position=na_position,
        )
        if not ascending:
            order = order[::-1]

        for k, v in df._columns.items():
            df._columns[k] = v[order]

        if ignore_index:
            df._index = ()

        return df

    def set_index(self, index: Any, inplace=False, drop=False) -> "DataFrame":
        """
        Install a new index to the DataFrame.

        :param index: One or more column keys to form the multi-level index.
        :param inplace: Value indicating whether we must update the current DataFrame instead of \
                        returning a copy.
        :param drop: Erase the old indexed columns, except those mentioned in the new index.
        """
        df = self.copy() if not inplace else self
        old_index = df._index

        if isinstance(index, list) or (isinstance(index, np.ndarray) and index.ndim > 1):
            index = tuple(index)

        if index == ():
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
        df = self.copy() if not inplace else self

        if df._index and drop:
            for c in df._index:
                del df._columns[c]

        df._index = ()
        return df

    def drop_duplicates(
        self,
        subset: Iterable[Hashable],
        keep: Literal["first", "last"] = "first",
        inplace: bool = False,
        ignore_index: bool = False,
    ) -> "DataFrame":
        """
        Remove duplicate rows from the DataFrame.

        We consider duplicates those rows which have the same values of columns in `subset`.

        :param subset: Column key or several column keys.
        :param keep: "first" leaves the first encounters, while "last" leaves the last encounters.
        :param inplace: Update the current DataFrame instead of returning a deduplicated copy.
        :param ignore_index: Value indicating whether the resulting index will be reset.
        """
        df = self if inplace else self.copy()
        if ignore_index:
            df._index = ()
        by = [df[c] for c in subset]
        order, merged = df._order(by, "stable")
        if keep == "last":
            order = order[::-1]
        _, first_found = np.unique(merged[order], return_index=True)
        if len(first_found) < len(merged):
            if keep == "last":
                first_found = np.arange(len(merged), dtype=int)[order][first_found]
            return df.take(first_found, inplace=inplace)
        return df

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
                seed.append(self[c])
        order, values = self._order(seed, "stable")
        _, counts = np.unique(values[order], return_counts=True)
        return Grouper(order, counts)

    def in_(
        self,
        column: Hashable,
        haystack: npt.ArrayLike,
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
        haystack = np.asarray(haystack)
        if values.dtype != haystack.dtype:
            raise ValueError(f"dtypes mismatch: {values.dtype} vs. {haystack.dtype}")
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

        :param unordered: Value indicating whether the result must be sorted or the caller \
                          doesn't care.
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
        concat_columns = {k: [v] for k, v in dfs[0]._columns.items()}
        for df in dfs[1:]:
            if not isinstance(df, cls):
                raise TypeError(f"Can only concatenate medvedi.DataFrame-s, got {type(df)}")
            if df._columns.keys() != columns:
                raise ValueError(f"Columns must match: {columns} vs. {df._columns.keys()}")
            if df._index != index and not ignore_index:
                raise ValueError(f"Indexes must match: {index} vs. {df._index}")
            for k, v in df._columns.items():
                concat_columns[k].append(v)
        concat_df = cls()
        concat_df._index = index if not ignore_index else ()
        concat_df._columns = {
            k: np.concatenate(v, casting="unsafe") for k, v in concat_columns.items()
        }
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
        if not isinstance(dfs[0], cls):
            raise TypeError(f"Can only concatenate medvedi.DataFrame-s, got {type(dfs[0])}")
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
        for df in dfs[1:]:
            if not isinstance(df, DataFrame):
                raise TypeError(f"Joined objects must be DataFrame-s, got {type(df)}")
            if df._index != indexes[0]:
                raise ValueError(f"Incompatible indexes: {df._index} vs. {indexes[0]}")
            indexes.append(df._index)
        if indexes[0] == ():
            raise ValueError("Joining requires an index")
        transposed_resolved_indexes_builder: list[list[np.ndarray]] = [[] for _ in indexes[0]]
        for index, df in zip(indexes, dfs):
            for i, c in enumerate(index):
                transposed_resolved_indexes_builder[i].append(df[c])
        transposed_resolved_indexes: list[np.ndarray] = [
            np.concatenate(vals, casting="unsafe") for vals in transposed_resolved_indexes_builder
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
        leave_mask = None
        if how == "left":
            leave_mask = index_map < len(dfs[0])
        elif how == "inner":
            encounters = np.zeros(len(index_map), dtype=np.uint8)
            pos = 0
            for df in dfs:
                df_len = len(df)
                encounters[inverse_map[pos : pos + df_len]] += 1
                pos += df_len
            leave_mask = encounters == len(dfs)
        if leave_mask is not None and not leave_mask.all():
            index_map = index_map[leave_mask]
            inverse_index_map = np.full(len(leave_mask), -1, dtype=int)
            inverse_index_map[leave_mask] = np.arange(leave_mask.sum())
            inverse_map = inverse_index_map[inverse_map]
            del inverse_index_map
        joined_columns = {i: c[index_map] for i, c in zip(indexes[0], transposed_resolved_indexes)}
        pos = 0
        mask = None
        for i, (df, suffix) in enumerate(zip(dfs, suffixes)):
            df_len = len(df)
            values_order = inverse_map[pos : pos + df_len]
            pos += df_len
            if must_mask := (how != "outer" and (i > 0 or how != "left")):
                mask = values_order >= 0
                values_order = values_order[mask]
            for c, values in df._columns.items():
                if c in df._index:
                    continue
                if suffix is not None:
                    c = str(c) + suffix
                if c not in joined_columns:
                    joined_columns[c] = joined_values = cls._empty_array(
                        len(index_map), values.dtype,
                    )
                else:
                    joined_values = joined_columns[c]
                if must_mask:
                    values = values[mask]
                joined_values[values_order] = values
        joined = DataFrame()
        joined._index = indexes[0]
        joined._columns = joined_columns
        return joined

    @staticmethod
    def _order(
        by: Sequence[np.ndarray],
        kind: Literal["quicksort", "stable"] | None = None,
        strict: bool = False,
        na_position: Literal["first", "last"] = "last",
    ) -> tuple[npt.NDArray[np.int_], np.ndarray]:
        if len(by) == 1:
            result = np.argsort(by[0], kind=kind)
            if na_position == "first":
                if (na_count := (by[0] != by[0]).sum()) > 0:
                    result = np.concatenate([result[-na_count:], result[:-na_count]])
            return result, by[0]
        if not strict and not ({c.dtype.kind for c in by} - mergeable_dtype_kinds):
            merged = merge_to_str(*by)
            return np.argsort(merged, kind=kind), merged
        mapped_bys = []
        for c in by:
            unique_values, inverse_indexes = np.unique(c, return_inverse=True)
            if na_position == "first" and unique_values[-1] != unique_values[-1]:
                inverse_indexes += 1
                inverse_indexes[inverse_indexes == len(unique_values)] = 0
            mapped_bys.append(inverse_indexes)
        merged = merge_to_str(*mapped_bys)
        return np.argsort(merged, kind=kind), merged

    @staticmethod
    def _empty_array(length: int, dtype: np.dtype) -> np.ndarray:
        if dtype == object:
            return np.empty(length, dtype=dtype)
        kind = dtype.kind
        if kind == "f" or kind == "m" or kind == "M":
            return np.full(length, None, dtype)
        return np.zeros(length, dtype=dtype)

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
