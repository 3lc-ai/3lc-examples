from __future__ import annotations
from os import PathLike
import tlc

import numpy as np

from sklearn.cluster import KMeans
import numpy as np
from collections import defaultdict
import pyarrow.parquet as pq
import pyarrow as pa


def split_table(
    table: tlc.Table,
    splits: dict[str, float] = {"train": 0.8, "val": 0.2},
    random_seed: int = 0,
) -> dict[str, tlc.Table]:
    # Split the table into two tables
    # for-loop and TableWriter ???
    # Just train/val or train/val/test or something more general? ??? Ref. k-fold
    pass


def join_parquet_columns(parquet_file_path: PathLike, new_columns: dict[str, np.ndarray], output_path: PathLike):
    """Join columns from a parquet file with new in-memory columns to form a new parquet file."""
    # Read the existing parquet file
    table = pq.read_table(parquet_file_path)

    # Convert new columns to PyArrow arrays and add to the table
    for column_name, column_data in new_columns.items():
        new_column = pa.array(column_data)
        table = table.append_column(column_name, new_column)

    # Write the updated table to a new parquet file
    pq.write_table(table, output_path)


def merge_new_data(table: tlc.Table, new_data: PathLike) -> tlc.Table:
    # Merge the new data with the table
    # for-loop and TableWriter ???
    # Inspect schema of existing table to determine if new data is compatible and how to process ???
    pass


def img_features():
    # https://github.com/encord-team/encord-active/blob/main/src/encord_active/lib/metrics/heuristic/img_features.py
    # Red, Green, Blue, Brightness, Sharpness (variance of the Laplacian), Blur (1-sharpness), AspectRatio, Area
    pass


def img_singlularity():
    r"""This metric gives each image a score that shows each image's uniqueness.
    - A score of zero means that the image has duplicates in the dataset; on the other hand, a score close to one represents that image is quite unique. Among the duplicate images, we only give a non-zero score to a single image, and the rest will have a score of zero (for example, if there are five identical images, only four will have a score of zero). This way, these duplicate samples can be easily tagged and removed from the project.
    - Images that are near duplicates of each other will be shown side by side.
    ### Possible actions
    - **To delete duplicate images:** You can set the quality filter to cover only zero values (that ends up with all the duplicate images), then use bulk tagging (e.g., with a tag like `Duplicate`) to tag all images.
    - **To mark duplicate images:** Near duplicate images are shown side by side. Navigate through these images and mark whichever is of interest to you.
    """
    # https://github.com/encord-team/encord-active/blob/main/src/encord_active/lib/metrics/semantic/image_singularity.py
    pass


def image_diversity_metric(labels, embeddings):
    # Convert embeddings to a NumPy array
    embeddings = embeddings.astype(np.float32)
    unique_labels = np.unique(labels)
    label_to_indices = defaultdict(list)

    # Group indices by labels
    for idx, label in enumerate(labels):
        label_to_indices[label].append(idx)

    # Calculate diversity scores
    diversity_scores = np.zeros(len(embeddings), dtype=int)

    for label in unique_labels:
        indices = label_to_indices[label]
        label_embeddings = embeddings[indices]

        # Cluster within each label (single cluster for this label group)
        kmeans = KMeans(n_clusters=1, n_init="auto").fit(label_embeddings)
        center = kmeans.cluster_centers_[0]

        # Calculate distances to the cluster center for this label
        distances = np.linalg.norm(label_embeddings - center, axis=1)

        # Sort indices by distance (closer to center = lower score)
        sorted_indices = np.argsort(distances)

        # Assign scores based on proximity to center
        for rank, idx in enumerate(sorted_indices):
            diversity_scores[indices[idx]] = rank + 1

    return diversity_scores.tolist()


from scipy.spatial.distance import pdist, squareform


def image_uniqueness_metric(labels, embeddings):
    # Convert embeddings to a NumPy array
    unique_labels = np.unique(labels)
    label_to_indices = defaultdict(list)

    # Group indices by labels
    for idx, label in enumerate(labels):
        label_to_indices[label].append(idx)

    # Calculate uniqueness scores
    uniqueness_scores = np.zeros(len(embeddings), dtype=float)

    for label in unique_labels:
        indices = label_to_indices[label]
        label_embeddings = embeddings[indices]

        # Compute pairwise distances within each class
        pairwise_distances = squareform(pdist(label_embeddings))

        # Calculate uniqueness as the mean distance of each image to others in its class
        mean_distances = pairwise_distances.mean(axis=1)

        # Assign the uniqueness score based on mean distances
        for idx, mean_distance in zip(indices, mean_distances):
            uniqueness_scores[idx] = mean_distance

    return uniqueness_scores.tolist()


def traversal_index(embeddings):
    embeddings = np.array(embeddings)

    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized_embeddings = embeddings / (norms + 1e-12)

    # Unique embeddings determination
    unique_embeddings, unique_indices, inverse_indices = np.unique(
        normalized_embeddings, axis=0, return_index=True, return_inverse=True
    )

    # Find the center of unique embeddings
    center_embedding = unique_embeddings.mean(axis=0)
    distances_from_center = np.linalg.norm(unique_embeddings - center_embedding, axis=1)
    center_index_in_unique = np.argmin(distances_from_center)

    # Initialize traversal order with the index of the central unique embedding
    traversal_order = [center_index_in_unique]
    selected_embeddings = set(traversal_order)

    # Iteratively select the farthest unpicked unique embedding
    remaining_count = len(unique_embeddings) - 1
    while remaining_count > 0:
        last_embedding = unique_embeddings[traversal_order[-1]]
        distances = np.linalg.norm(unique_embeddings - last_embedding, axis=1)

        for idx in np.argsort(distances)[::-1]:  # Start from farthest
            if idx not in selected_embeddings:
                traversal_order.append(idx)
                selected_embeddings.add(idx)
                remaining_count -= 1
                break

    # Add remaining unique indices in random order if not yet selected
    remaining_unique = [i for i in range(len(unique_embeddings)) if i not in traversal_order]
    np.random.shuffle(remaining_unique)
    traversal_order.extend(remaining_unique)

    # Map traversal_order back to original embeddings indices
    final_traversal_order = [unique_indices[idx] for idx in traversal_order]

    # Add duplicates in random order
    duplicate_indices = [i for i in range(len(embeddings)) if i not in unique_indices]
    np.random.shuffle(duplicate_indices)
    final_traversal_order.extend(duplicate_indices)

    return final_traversal_order
