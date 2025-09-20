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


class _SplitStrategy(abc.ABC):
    requires_split_by = False
    allows_shuffle = True

    def __init__(self, seed: int = 0):
        self.seed = seed

    @abc.abstractmethod
    def split(
        self,
        indices: np.ndarray,
        splits: dict[str, float],
        by_column: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """Split the indices into the specified splits."""
        ...

    def _get_split_sizes(self, total_count: int, splits: dict[str, float]) -> list[int]:
        split_sizes = [int(total_count * proportion) for proportion in splits.values()]
        if sum(split_sizes) < total_count:
            split_sizes[0] += total_count - sum(split_sizes)
        return split_sizes
    
    @staticmethod
    def _prepare_split_by_column(by_column: np.ndarray) -> np.ndarray:
        # Ensure that the split by column is a 1D numpy array
        ret = []
        for value in by_column:
            if isinstance(value, np.ndarray):
                ret.append(value[0].item())
            else:
                ret.append(value)
        return np.array(ret)


class _RandomSplitStrategy(_SplitStrategy):
    allows_shuffle = True

    def split(
        self,
        indices: np.ndarray,
        splits: dict[str, float],
        by_column: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        split_sizes = self._get_split_sizes(len(indices), splits)
        splits_indices = np.split(indices, np.cumsum(split_sizes[:-1]))
        return {split_name: split_indices for split_name, split_indices in zip(splits, splits_indices)}


class _StratifiedSplitStrategy(_SplitStrategy):
    requires_split_by = True
    allows_shuffle = True

    def split(
        self,
        indices: np.ndarray,
        splits: dict[str, float],
        by_column: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        if by_column is None:
            msg = "Stratified split requires a column to stratify by."
            raise ValueError(msg)

        if len(splits) != 2:
            msg = "Stratified split requires exactly two splits."
            raise ValueError(msg)
        test_size = list(splits.values())[1]
        by_column = self._prepare_split_by_column(by_column)
        splits_indices = train_test_split(
            indices,
            test_size=test_size,
            stratify=by_column,
            random_state=self.seed,
        )
        return {split_name: split_indices for split_name, split_indices in zip(splits, splits_indices)}


class _TraversalIndexSplitStrategy(_RandomSplitStrategy):
    requires_split_by = True
    allows_shuffle = False  # FPS sampling should not be shuffled

    def split(
        self,
        indices: np.ndarray,
        splits: dict[str, float],
        by_column: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        if by_column is None:
            msg = "Traversal index split requires a column to traverse by."
            raise ValueError(msg)

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


class _BalancedGreedySplitStrategy(_SplitStrategy):
    requires_split_by = True
    allows_shuffle = True

    def split(
        self,
        indices: np.ndarray,
        splits: dict[str, float],
        by_column: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        if by_column is None:
            msg = "Balanced greedy split requires a column to balance by."
            raise ValueError(msg)

        # Get split sizes based on proportions
        split_names = list(splits.keys())

        # Initialize result dictionary
        split_indices: dict[str, list[int]] = {name: [] for name in split_names}

        # Group indices by class
        _, class_indices = np.unique(by_column, return_inverse=True)
        class_groups: dict[int, list[int]] = {}
        for i, class_idx in enumerate(class_indices):
            if class_idx not in class_groups:
                class_groups[class_idx] = []
            class_groups[class_idx].append(indices[i])

        # Sort classes by frequency (rarest first)
        class_frequencies = [(class_idx, len(indices_list)) for class_idx, indices_list in class_groups.items()]
        class_frequencies.sort(key=lambda x: x[1])

        # For each class, distribute instances proportionally across splits
        for class_idx, class_count in class_frequencies:
            class_indices_list = class_groups[class_idx]

            # Shuffle the indices for this class
            np.random.seed(self.seed + class_idx)  # Different seed per class for reproducibility
            shuffled_class_indices = np.random.permutation(class_indices_list)

            # Calculate proportional distribution for this class
            current_idx = 0
            for split_name, split_proportion in splits.items():
                # Calculate how many instances this split should get for this class
                instances_for_this_split = int(class_count * split_proportion)

                # Add instances to this split
                end_idx = current_idx + instances_for_this_split
                if current_idx < len(shuffled_class_indices):
                    split_indices[split_name].extend(
                        shuffled_class_indices[current_idx : min(end_idx, len(shuffled_class_indices))]
                    )
                current_idx = end_idx

                # If we've used all instances for this class, break
                if current_idx >= len(shuffled_class_indices):
                    break

        # Convert lists to numpy arrays
        return {name: np.array(indices_list) for name, indices_list in split_indices.items()}


class _UndersampledBalancedSplitStrategy(_SplitStrategy):
    requires_split_by = True
    allows_shuffle = True

    def split(
        self,
        indices: np.ndarray,
        splits: dict[str, float],
        by_column: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        if by_column is None:
            msg = "Undersampled balanced split requires a column to balance by."
            raise ValueError(msg)

        # Group indices by class
        unique_classes, class_indices = np.unique(by_column, return_inverse=True)
        class_groups: dict[int, list[int]] = {}
        for i, class_idx in enumerate(class_indices):
            if class_idx not in class_groups:
                class_groups[class_idx] = []
            class_groups[class_idx].append(indices[i])

        # Find the rarest class size
        min_class_size = min(len(indices_list) for indices_list in class_groups.values())

        # Calculate how many instances each split should get per class
        split_names = list(splits.keys())
        instances_per_class_per_split = {}
        for split_name, split_proportion in splits.items():
            instances_per_class_per_split[split_name] = int(min_class_size * split_proportion)

        # Initialize result dictionary
        split_indices: dict[str, list[int]] = {name: [] for name in split_names}

        # For each class, sample equal numbers for each split
        for class_idx, class_indices_list in class_groups.items():
            # Shuffle the indices for this class
            np.random.seed(self.seed + class_idx)  # Different seed per class for reproducibility
            shuffled_class_indices = np.random.permutation(class_indices_list)

            # Sample equal numbers for each split
            current_idx = 0
            for split_name in split_names:
                instances_for_this_split = instances_per_class_per_split[split_name]

                # Add instances to this split
                end_idx = current_idx + instances_for_this_split
                if current_idx < len(shuffled_class_indices):
                    split_indices[split_name].extend(
                        shuffled_class_indices[current_idx : min(end_idx, len(shuffled_class_indices))]
                    )
                current_idx = end_idx

                # If we've used all instances for this class, break
                if current_idx >= len(shuffled_class_indices):
                    break

        # Convert lists to numpy arrays
        return {name: np.array(indices_list) for name, indices_list in split_indices.items()}


_STRATEGY_MAP = {
    "random": _RandomSplitStrategy,
    "stratified": _StratifiedSplitStrategy,
    "traversal_index": _TraversalIndexSplitStrategy,
    "balanced_greedy": _BalancedGreedySplitStrategy,
    "undersampled_balanced": _UndersampledBalancedSplitStrategy,
}


def split_table(
    table: tlc.Table,
    splits: dict[str, float] | None = None,
    random_seed: int = 0,
    split_strategy: Literal[
        "random",
        "stratified",
        "traversal_index",
        "balanced_greedy",
        "undersampled_balanced",
    ] = "random",
    shuffle: bool = True,
    split_by: int | str | Callable[[Any], int] | None = None,
    if_exists: Literal["reuse", "overwrite", "rename"] = "reuse",
) -> dict[str, tlc.Table]:
    """
    Splits a table into two or more tables based on the specified strategy.

    :param table: The table to split.
    :param splits: Proportions for splits, as a dictionary with split names as keys and proportions as values. Default
        is {"train": 0.8, "val": 0.2}. Any number of splits can be requested. Proportions are normalized if they do not
        sum to 1.
    :param random_seed: Seed for reproducibility.
    :param split_strategy: "random", "stratified" (requires `split_by`), "traversal_index" (requires `split_by`),
        "balanced_greedy" (requires `split_by`), or "undersampled_balanced" (requires `split_by`).
    :param shuffle: Shuffle data for "random" split. Default is True.
    :param split_by: Column or property to use for splitting. Required for "stratified", "traversal_index", and
        "balanced_greedy" strategies. Provide a string if the rows are dictionaries, or an integer if the rows are
        tuples/lists. If a callable is provided it will be called with each row and should return the value on which
        to split. If the returned column is a jagged array (e.g. instance labels), the first element will be used.
    :param if_exists: What to do if the split tables already exist. Default is "reuse". Note: reusability is determined
        solely by the input table url and the split names. To be certain that new tables are created, use "rename".

    :returns: Split tables as per requested strategy.
    """
    if not splits or splits is None:
        splits = {"train": 0.8, "val": 0.2}

    if if_exists not in ["reuse", "overwrite", "rename"]:
        msg = f"Invalid if_exists value: {if_exists}, must be one of 'reuse', 'overwrite', or 'rename'."
        raise ValueError(msg)

    if if_exists == "reuse":
        exist_count = 0
        for split_name, _ in splits.items():
            exist_count += 1 if table.url.create_sibling(split_name).exists() else 0
        if exist_count == len(splits):
            return {split_name: tlc.Table.from_url(table.url.create_sibling(split_name)) for split_name in splits}
        elif exist_count > 0:
            msg = f"Some split tables already exist: {exist_count}. Use 'overwrite' or 'rename' to proceed."
            raise ValueError(msg)

    for _, split_proportion in splits.items():
        if split_proportion < 0 or split_proportion > 1:
            msg = f"Invalid split proportion: {split_proportion}, must be between 0 and 1."
            raise ValueError(msg)

    if not math.isclose(sum(splits.values()), 1.0):
        warnings.warn("Split proportions do not sum to 1. Normalizing.", stacklevel=2)
        total = sum(splits.values())
        splits = {k: v / total for k, v in splits.items()}

    indices = np.arange(len(table))

    strategy_class = _STRATEGY_MAP.get(split_strategy)
    if strategy_class is None:
        available_strategies = ", ".join(_STRATEGY_MAP.keys())
        msg = f"Invalid split strategy: {split_strategy}. Must be one of {available_strategies}"
        raise ValueError(msg)

    strategy = strategy_class(random_seed)  # type: ignore[abstract]

    kwargs = {}
    if strategy.requires_split_by:
        if split_by is None:
            msg = f"Split strategy '{split_strategy}' requires a split_by column."
            raise ValueError(msg)
        kwargs["by_column"] = _get_column(table, split_by)

    # Apply shuffling only if the strategy allows it
    if shuffle and strategy.allows_shuffle:
        indices = sk_shuffle(indices, random_state=random_seed)

    splits_indices = strategy.split(indices, splits, **kwargs)

    # Return dictionary with tables based on final split indices
    return {
        split_name: keep_indices(table, split_indices.tolist(), table_name=split_name, if_exists=if_exists)
        for split_name, split_indices in splits_indices.items()
    }


def _get_column(table: tlc.Table, column: int | str | Callable[..., int]) -> np.ndarray:
    if isinstance(column, str):
        pa_column = table.get_column(column)

        # Handle FixedSizeListType specifically
        if hasattr(pa_column.type, "list_size"):
            # Use PyArrow's flatten() method to get the underlying values
            # then reshape to the correct dimensions
            fixed_size = pa_column.type.list_size
            flattened_array = pa_column.flatten().to_numpy(zero_copy_only=False)
            return flattened_array.reshape(-1, fixed_size)  # type: ignore[no-any-return]

        # For other column types, convert directly to numpy
        return pa_column.to_numpy(zero_copy_only=False)  # type: ignore[no-any-return]

    if isinstance(column, int):
        return np.array([row[column] for row in table])

    elif callable(column):
        return np.array([column(row) for row in table])

    else:
        msg = f"Invalid column type: {type(column)}"
        raise ValueError(msg)


def set_value_in_column_to_fixed_value(
    table: tlc.Table,
    column: str,
    indices: list[int],
    value: Any,
) -> tlc.Table:
    """Set a fixed value in a column for a given list of indices.

    e.g. set_value_in_column_to_fixed_value(table, "weight", [0, 3, 4, 6, ...], value=0)

    :param table: The table to modify.
    :param column: The column name to set values in.
    :param indices: The indices to set the value for.
    :param value: The value to set.
    :returns: The modified table.
    """
    runs = tlc.EditedTable.indices_to_run(indices)

    edits = {
        column: {"runs_and_values": [runs, value]},
    }

    edited_table = tlc.EditedTable(
        url=table.url.create_sibling(f"set_{column}_to_0_in_{len(indices)}_rows").create_unique(),
        input_table_url=table,
        edits=edits,
        row_cache_url="./row_cache.parquet",
    )
    edited_table.ensure_fully_defined()
    edited_table.write_to_url()
    return edited_table


def keep_indices(
    table: tlc.Table,
    indices: list[int],
    table_name: str | None = None,
    if_exists: Literal["overwrite", "rename", "reuse"] = "rename",
) -> tlc.Table:
    """Keep only the rows with the specified indices in the table.

    :param table: The table to filter.
    :param indices: The indices to keep.
    :returns: The filtered table.
    """

    all_indices = list(range(len(table)))
    indices_to_remove = list(set(all_indices) - set(indices))
    runs_and_values = []
    for index in indices_to_remove:
        runs_and_values.extend([[index], True])
    edits = {
        tlc.SHOULD_DELETE: {"runs_and_values": runs_and_values},
    }
    table_url = table.url.create_sibling(table_name or "remove")
    if if_exists == "rename":
        table_url = table_url.create_unique()

    edited_table = tlc.EditedTable(
        url=table_url,
        input_table_url=table,
        edits=edits,
        row_cache_url="./row_cache.parquet",
    )
    edited_table.ensure_fully_defined()
    edited_table.write_to_url()
    return edited_table


def get_balanced_coreset_indices(
    table: tlc.Table,
    size: float = 1.0,
    split_by: int | str | Callable[[Any], int] = "label",
    random_seed: int = 0,
) -> tuple[list[int], list[int]]:
    """
    Returns indices for both coreset and non-coreset samples.

    This function creates a balanced coreset by sampling an equal number of instances
    from each class, based on the size of the rarest class. The coreset maintains
    class balance while maximizing the number of samples used.

    :param table: The table to create a coreset from.
    :param size: Proportion of the RAREST class to include (0.0 to 1.0).
        ⚠️  IMPORTANT: This is NOT the proportion of total dataset!
        Example: If rarest class has 100 samples and size=0.5, each class gets 50 samples.
        To use ALL samples from the rarest class, use size=1.0.
    :param split_by: Column or property to use for balancing. Provide a string if
        the rows are dictionaries, or an integer if the rows are tuples/lists.
        If a callable is provided it will be called with each row and should
        return the value on which to balance.
    :param random_seed: Seed for reproducibility.

    :returns: Tuple of (coreset_indices, non_coreset_indices).

    :example:
        # Get both coreset and non-coreset indices
        coreset_indices, non_coreset_indices = get_balanced_coreset_indices(
            table=my_table,
            size=1.0,  # Use ALL samples from rarest class
            split_by="label_column"
        )
        # This will give you all samples from the rarest class for each class
    """
    if size <= 0 or size > 1:
        msg = f"Invalid coreset size: {size}, must be between 0 and 1."
        raise ValueError(msg)

    # Group indices by class
    by_column = _get_column(table, split_by)
    _, class_indices = np.unique(by_column, return_inverse=True)
    class_groups: dict[int, list[int]] = {}
    for i, class_idx in enumerate(class_indices):
        if class_idx not in class_groups:
            class_groups[class_idx] = []
        class_groups[class_idx].append(i)  # Use row index directly

    # Find the rarest class size
    min_class_size = min(len(indices_list) for indices_list in class_groups.values())

    # Calculate how many instances each class should contribute to the coreset
    instances_per_class = int(min_class_size * size)

    # Initialize result lists
    coreset_indices: list[int] = []
    non_coreset_indices: list[int] = []

    # For each class, sample the required number of instances
    for class_idx, class_indices_list in class_groups.items():
        # Shuffle the indices for this class
        np.random.seed(random_seed + class_idx)  # Different seed per class for reproducibility
        shuffled_class_indices = np.random.permutation(class_indices_list)

        # Take the first instances_per_class for coreset
        coreset_indices.extend(shuffled_class_indices[:instances_per_class])

        # Add the rest to non-coreset
        non_coreset_indices.extend(shuffled_class_indices[instances_per_class:])

    # Convert numpy integers to Python integers
    coreset_indices = [int(idx) for idx in coreset_indices]
    non_coreset_indices = [int(idx) for idx in non_coreset_indices]

    return coreset_indices, non_coreset_indices
