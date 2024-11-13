"""Tools for splitting data in one Table into two or more split Tables."""

from __future__ import annotations

import abc
from typing import Any, Callable, Literal

import numpy as np
import tlc
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle as sk_shuffle

from tools.common import keep_indices
from tools.metrics import traversal_index


class _SplitStrategy(abc.ABC):
    def __init__(self, seed: int = 0):
        self.seed = seed

    @abc.abstractmethod
    def split(self, indices: np.array, splits: dict[str, float], by_column: np.array | None = None) -> dict[str, np.array]:
        """Split the indices into the specified splits."""
        ...
    
    def _get_split_sizes(self, total_count: int, splits: dict[str, float]) -> list[int]:
        split_sizes = [int(total_count * proportion) for proportion in splits.values()]
        if sum(split_sizes) < total_count:
            split_sizes[0] += total_count - sum(split_sizes)
        return split_sizes
    

class _RandomSplitStrategy(_SplitStrategy):
    def split(self, indices: np.array, splits: dict[str, float], by_column: np.array | None = None) -> dict[str, np.array]:
        split_sizes = self._get_split_sizes(len(indices), splits)
        splits_indices = np.split(indices, np.cumsum(split_sizes[:-1]))
        return {split_name: split_indices for split_name, split_indices in zip(splits, splits_indices)}
    
class _StratifiedSplitStrategy(_SplitStrategy):
    def split(self, indices: np.array, splits: dict[str, float], by_column: np.array | None = None) -> dict[str, np.array]:
        if by_column is None:
            raise ValueError("Stratified split requires a column to stratify by.")

        if len(splits) != 2:
            raise ValueError("Stratified split requires exactly two splits.")
        split_sizes = self._get_split_sizes(len(indices), splits)
        splits_indices = train_test_split(
            indices, test_size=split_sizes[1], stratify=by_column, random_state=self.seed
        )
        return {split_name: split_indices for split_name, split_indices in zip(splits, splits_indices)}
    
class _TraversalIndexSplitStrategy(_RandomSplitStrategy):
    def split(self, indices: np.array, splits: dict[str, float], by_column: np.array | None = None) -> dict[str, np.array]:
        if by_column is None:
            raise ValueError("Traversal index split requires an embeddings column to compute traversal index.")
        traversal_indices = traversal_index(by_column)
        return super().split(traversal_indices, splits)
    
_STRATEGY_MAP = {
    "random": _RandomSplitStrategy,
    "stratified": _StratifiedSplitStrategy,
    "traversal_index": _TraversalIndexSplitStrategy,
}


def split_table(
    table: tlc.Table,
    splits: dict[str, float] | None = None,
    random_seed: int = 0,
    split_strategy: Literal["random", "stratified", "traversal_index"] = "random",
    shuffle: bool = True,
    split_by: int | str | Callable[[Any], int] | None = None,
) -> dict[str, dict[str, Any]]:
    """
    Splits a table into two or more tables based on the specified strategy.

    :param table: The table to split.
    :param splits: Proportions for train and validation splits, ignored if `n_folds` is provided. Default is 80/20.
    :param random_seed: Seed for reproducibility.
    :param split_strategy: "random", "stratified" (requires `split_by`), or "traversal_index" (requires `split_by`).
    :param shuffle: Shuffle data for "random" split. Default is True.
    :param split_by: Column or property to use for splitting. Required for "stratified" and "traversal_index"
        strategies. Provide a string if the rows are dictionaries, or an integer if the rows are tuples/lists. If a
        callable is provided it will be called with each row and should return the value on which to split.

    :returns: Split tables as per requested strategy, including k-fold results if `n_folds` is provided.
    """
    if splits is None:
        splits = {"train": 0.8, "val": 0.2}

    indices = np.arange(len(table))

    # Regular splitting
    if shuffle and split_strategy == "random":
        indices = sk_shuffle(indices, random_state=random_seed)

    strategy_class = _STRATEGY_MAP.get(split_strategy)
    if strategy_class is None:
        raise ValueError(f"Invalid split strategy: {split_strategy}. Must be one of {_STRATEGY_MAP.keys()}")
    
    strategy = strategy_class(random_seed)
    splits_indices = strategy.split(indices, splits, by_column=_get_column(table, split_by))

    # Return dictionary with tables based on final split indices
    return {
        split_name: keep_indices(
            table,
            split_indices.tolist(),
            table_name=split_name,
        )
        for split_name, split_indices in splits_indices.items()
    }

def _get_column(table, column: int | str | Callable[[Any], int]) -> np.array:
    if isinstance(column, (int, str)):
        return np.array([row[column] for row in table])
    
    elif callable(column):
        return np.array([column(row) for row in table])
    
    else:
        raise ValueError(f"Invalid label_column: {column}")
