"""Tools for splitting data in one Table into two or more split Tables."""

from __future__ import annotations

import abc
import math
import warnings
from typing import Any, Callable, Literal

import fpsample
import numpy as np
import tlc
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle as sk_shuffle

from tlc_tools.common import keep_indices


class _SplitStrategy(abc.ABC):
    def __init__(self, seed: int = 0):
        self.seed = seed

    @abc.abstractmethod
    def split(
        self, indices: np.array, splits: dict[str, float], by_column: np.array | None = None
    ) -> dict[str, np.array]:
        """Split the indices into the specified splits."""
        ...

    def _get_split_sizes(self, total_count: int, splits: dict[str, float]) -> list[int]:
        split_sizes = [int(total_count * proportion) for proportion in splits.values()]
        if sum(split_sizes) < total_count:
            split_sizes[0] += total_count - sum(split_sizes)
        return split_sizes


class _RandomSplitStrategy(_SplitStrategy):
    requires_split_by = False

    def split(
        self, indices: np.array, splits: dict[str, float], by_column: np.array | None = None
    ) -> dict[str, np.array]:
        split_sizes = self._get_split_sizes(len(indices), splits)
        splits_indices = np.split(indices, np.cumsum(split_sizes[:-1]))
        return {split_name: split_indices for split_name, split_indices in zip(splits, splits_indices)}


class _StratifiedSplitStrategy(_SplitStrategy):
    requires_split_by = True

    def split(
        self, indices: np.array, splits: dict[str, float], by_column: np.array | None = None
    ) -> dict[str, np.array]:
        if by_column is None:
            raise ValueError("Stratified split requires a column to stratify by.")

        if len(splits) != 2:
            raise ValueError("Stratified split requires exactly two splits.")
        split_sizes = self._get_split_sizes(len(indices), splits)
        splits_indices = train_test_split(indices, test_size=split_sizes[1], stratify=by_column, random_state=self.seed)
        return {split_name: split_indices for split_name, split_indices in zip(splits, splits_indices)}


class _TraversalIndexSplitStrategy(_RandomSplitStrategy):
    requires_split_by = True

    def split(
        self, indices: np.array, splits: dict[str, float], by_column: np.array | None = None
    ) -> dict[str, np.array]:
        # Sort to take smallest splits first
        splits = dict(sorted(splits.items(), key=lambda x: x[1]))
        largest_split_name = list(splits.keys())[-1]

        # Track original indices explicitly
        original_indices = indices.tolist()
        remaining_indices = original_indices.copy()  # Absolute indices from the start

        # FPS one split at a time
        split_indices = {}
        for split_name, split_proportion in list(splits.items())[:-1]:
            # Determine the number of samples for this split
            split_size = int(len(original_indices) * split_proportion)

            # Perform sampling on the current subset of remaining indices
            sampled_indices = fpsample.bucket_fps_kdtree_sampling(
                by_column[remaining_indices],  # Subset of `by_column` corresponding to remaining indices
                split_size,
            ).tolist()

            # Map sampled indices to their absolute values in the original array
            split_indices[split_name] = [remaining_indices[i] for i in sampled_indices]

            # Update remaining indices by removing the sampled ones
            remaining_indices = [idx for idx in remaining_indices if idx not in split_indices[split_name]]

        # Add the remaining indices to the largest split
        split_indices[largest_split_name] = remaining_indices

        # Return results as numpy arrays
        return {s: np.array(split_indices[s]) for s in splits}


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
) -> dict[str, tlc.Table]:
    """
    Splits a table into two or more tables based on the specified strategy.

    :param table: The table to split.
    :param splits: Proportions for train and validation splits. Default is a 80/20 train and val split.
    :param random_seed: Seed for reproducibility.
    :param split_strategy: "random", "stratified" (requires `split_by`), or "traversal_index" (requires `split_by`).
    :param shuffle: Shuffle data for "random" split. Default is True.
    :param split_by: Column or property to use for splitting. Required for "stratified" and "traversal_index"
        strategies. Provide a string if the rows are dictionaries, or an integer if the rows are tuples/lists. If a
        callable is provided it will be called with each row and should return the value on which to split.

    :returns: Split tables as per requested strategy.
    """
    if splits is None:
        splits = {"train": 0.8, "val": 0.2}

    for _, split_proportion in splits.items():
        if split_proportion < 0 or split_proportion > 1:
            msg = f"Invalid split proportion: {split_proportion}, must be between 0 and 1."
            raise ValueError(msg)

    if not math.isclose(sum(splits.values()), 1.0):
        warnings.warn("Split proportions do not sum to 1. Normalizing.", stacklevel=2)
        total = sum(splits.values())
        splits = {k: v / total for k, v in splits.items()}

    indices = np.arange(len(table))

    # Regular splitting
    if shuffle and split_strategy == "random":
        indices = sk_shuffle(indices, random_state=random_seed)

    strategy_class = _STRATEGY_MAP.get(split_strategy)
    if strategy_class is None:
        available_strategies = ", ".join(_STRATEGY_MAP.keys())
        msg = f"Invalid split strategy: {split_strategy}. Must be one of {available_strategies}"
        raise ValueError(msg)

    strategy = strategy_class(random_seed)

    kwargs = {}
    if strategy.requires_split_by:
        if split_by is None:
            msg = f"Split strategy '{split_strategy}' requires a split_by column."
            raise ValueError(msg)
        kwargs["by_column"] = _get_column(table, split_by)

    splits_indices = strategy.split(indices, splits, **kwargs)

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
    # TODO: Use more performant `tlc.get_column` when available
    if isinstance(column, (int, str)):
        return np.array([row[column] for row in table])

    elif callable(column):
        return np.array([column(row) for row in table])

    else:
        msg = f"Invalid column type: {type(column)}"
        raise ValueError(msg)
