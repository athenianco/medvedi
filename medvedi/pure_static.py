from typing import Literal, TypeVar

DataFrame = TypeVar("DataFrame")


class PureStaticDataFrameMethods(type):
    """
    Metaclass of DataFrame.

    Used to define class-only static methods.
    """

    def concat(
        cls,
        *dfs: DataFrame,
        ignore_index: bool = False,
        copy: bool = False,
        strict: bool = True,
    ) -> DataFrame:
        """
        Merge several DataFrame-s vertically.

        The column names must match.

        :param ignore_index: Do not update the index. The result index will be a range.
        :param copy: Do not copy data unless have to.
        :param strict: Require that all the specified dataframes have the same columns.
        :return: Resulting DataFrame of length = sum of concatenated DataFrame lengths.
        """
        return cls._concat(*dfs, ignore_index=ignore_index, copy=copy, strict=strict)

    def join(
        cls,
        *dfs: DataFrame,
        how: Literal["left", "right", "inner", "outer"] = "left",
        suffixes: tuple[str | None, ...] = (),
        copy: bool = False,
    ) -> DataFrame:
        """
        Join several DataFrame-s together by index.

        Compared to `concat()`, we operate "horizontally". The index names must match.

        :param how: One of "left", "right", "inner", "outer".
        :param suffixes: Tuple with appended column suffixes for each joined DataFrame.
        :param copy: Do not copy data unless have to.
        :return: Resulting DataFrame with columns taken from each joined DataFrame.
        """
        return cls._join(*dfs, how=how, suffixes=suffixes, copy=copy)

    def _concat(
        cls,
        *dfs: DataFrame,
        ignore_index: bool = False,
        copy: bool = False,
        strict: bool = True,
    ) -> DataFrame:
        raise NotImplementedError

    def _join(
        cls,
        *dfs: DataFrame,
        how: Literal["left", "right", "inner", "outer"] = "left",
        suffixes: tuple[str | None, ...] = (),
        copy: bool = False,
    ) -> DataFrame:
        raise NotImplementedError
