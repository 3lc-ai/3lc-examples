"""Tools for splitting data in one Table into two or more split Tables."""

from __future__ import annotations

from typing import Any, Callable, Literal

import numpy as np
import tlc
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils import shuffle as sk_shuffle

from tools.common import keep_indices
from tools.metrics import traversal_index


class SplitManager:
    def __init__(self, random_seed: int = 0):
        self.random_seed = random_seed

    def _get_split_sizes(self, indices, splits):
        split_sizes = [int(len(indices) * proportion) for proportion in splits.values()]
        if sum(split_sizes) < len(indices):
            split_sizes[0] += len(indices) - sum(split_sizes)
        return split_sizes

    def random_split(self, indices, splits):
        split_sizes = self._get_split_sizes(indices, splits)
        return np.split(indices, np.cumsum(split_sizes[:-1]))

    def stratified_split(self, labels, splits, indices):
        split_sizes = self._get_split_sizes(indices, splits)
        
        split_indices = {}
        # train_size = splits["train"]
        train_indices, val_indices = train_test_split(
            indices, test_size=split_sizes[1], stratify=labels, random_state=self.random_seed
        )
        split_indices["train"] = train_indices
        split_indices["val"] = val_indices
        return split_indices

    def traversal_split(self, traversal_indices, splits):
        split_sizes = [int(len(traversal_indices) * proportion) for proportion in splits.values()]
        if sum(split_sizes) < len(traversal_indices):
            split_sizes[0] += len(traversal_indices) - sum(split_sizes)
        return np.split(traversal_indices, np.cumsum(split_sizes[:-1]))

    def get_k_fold_split(self, n_folds, indices, labels=None):
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.random_seed)
        return list(kf.split(indices, labels))


def split_table(
    table: tlc.Table,
    splits: dict[str, float] | None = None,
    random_seed: int = 0,
    split_strategy: Literal["random", "stratified", "traversal_index"] = "random",
    shuffle: bool = True,
    split_by: int | str | Callable[[Any], int] | None = None,
    n_folds: int | None = None,
) -> dict[str, dict[str, Any]]:
    """
    Splits a table into train/validation sets or performs k-fold cross-validation splits.

    :param table: The table to split.
    :param splits: Proportions for train and validation splits, ignored if `n_folds` is provided. Default is 80/20.
    :param random_seed: Seed for reproducibility.
    :param split_strategy: "random", "stratified" (requires `split_by`), or "traversal_index" (requires `split_by`).
    :param shuffle: Shuffle data for "random" split. Default is True.
    :param split_by: Column or property to use for splitting. Required for "stratified" and "traversal_index"
        strategies. Provide a string if the rows are dictionaries, or an integer if the rows are tuples/lists. If a
        callable is provided it will be called with each row and should return the value on which to split.
    :param n_folds: Enables k-fold cross-validation, each fold contains "train" and "val".

    :returns: Split tables as per requested strategy, including k-fold results if `n_folds` is provided.
    """
    if splits is None:
        splits = {"train": 0.8, "val": 0.2}

    manager = SplitManager(random_seed)
    indices = np.arange(len(table))

    if n_folds:
        # NOTE: K-fold cross-validation should perhaps be in a separate tool - it could even take several tables as input
        raise NotImplementedError("K-fold cross-validation is not yet implemented.")
        # # If n_folds is specified, perform k-fold cross-validation with train/val splits
        # splits_indices = {}
        # if split_strategy == "stratified" and split_by:
        #     labels = table[stratify_column].to_numpy()
        #     fold_splits = manager.get_k_fold_split(n_folds, indices, labels=labels)
        # else:
        #     fold_splits = manager.get_k_fold_split(n_folds, indices)

        # for fold, (train_idx, val_idx) in enumerate(fold_splits):
        #     splits_indices[f"fold_{fold}"] = {"train": table[train_idx], "val": table[val_idx]}
        # return splits_indices

    # Regular splitting (not k-fold)
    if shuffle and split_strategy == "random":
        indices = sk_shuffle(indices, random_state=random_seed)

    if split_strategy == "random":
        split_sizes = manager.random_split(indices, splits)
        splits_indices = {split_name: split_indices for split_name, split_indices in zip(splits.keys(), split_sizes)}

    elif split_strategy == "stratified":
        if split_by is None:
            msg = "Stratified split requires 'split_by', to specify which column to base stratified sampling on."
            raise ValueError(msg)
        labels = _get_column(table, split_by)
        splits_indices = manager.stratified_split(labels, splits, indices)

    elif split_strategy == "traversal_index":
        # TODO: Fix traversal index computation, currently incorrect
        if split_by is None:
            msg = (
                "Traversal index split requires 'split_by', to specify which column (with embeddings) to base "
                " traversal index on."
            )
            raise ValueError(msg)
        embeddings = _get_column(table, split_by)
        traversal_indices = traversal_index(embeddings)
        split_sizes = manager.traversal_split(traversal_indices, splits)
        splits_indices = {"train": split_sizes[0], "val": split_sizes[1]}

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