from __future__ import annotations

from collections import defaultdict
from typing import Literal

import cv2
import numpy as np
from PIL import Image, ImageStat
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans


def diversity(labels: list | np.ndarray, embeddings: np.ndarray) -> list[int]:
    """
    Calculate diversity scores for a dataset based on embeddings.

    Each sample is assigned a score indicating its proximity to the cluster center
    of its label group, with lower scores for samples closer to the center.

    :param labels: Class labels for the samples.
    :param embeddings: Feature embeddings of the samples.
    :returns: Diversity scores for each sample, ranked by distance to the cluster center.
    """
    # Convert embeddings to a NumPy array
    embeddings = embeddings.astype(np.float32)
    unique_labels = np.unique(labels)
    label_to_indices = defaultdict(list)

    # Group indices by labels
    for idx, label in enumerate(labels):
        label_to_indices[label].append(idx)

    # Calculate diversity scores
    diversity_scores = [0] * len(embeddings)

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

    return diversity_scores


def uniqueness(labels: list[int | float] | np.ndarray, embeddings: np.ndarray) -> list[float]:
    """
    Calculate uniqueness scores for a dataset based on embeddings.

    Each sample is assigned a score based on the mean pairwise distance to other samples
    within the same label group, with higher scores indicating greater uniqueness.

    :param labels: Class labels for the samples.
    :param embeddings: Feature embeddings of the samples.
    :returns: Uniqueness scores for each sample, reflecting their distinctiveness within their label group.
    """
    unique_labels = np.unique(labels)
    label_to_indices = defaultdict(list)

    # Group indices by labels
    for idx, label in enumerate(labels):
        label_to_indices[label].append(idx)

    # Calculate uniqueness scores
    uniqueness_scores = [0.0] * len(embeddings)

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

    return uniqueness_scores


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

    return [int(idx) for idx in final_traversal_order]


IMAGE_METRICS = Literal[
    "width", "height", "brightness", "contrast", "sharpness", "average_red", "average_green", "average_blue"
]


def compute_image_metrics(image_path: str, metrics: list[IMAGE_METRICS] | None = None) -> dict[str, float]:
    """Return a dict of image metrics for the given image path."""

    if not metrics:
        metrics = [
            "width",
            "height",
            "brightness",
            "contrast",
            "sharpness",
            "average_red",
            "average_green",
            "average_blue",
        ]

    computed_metrics = {}

    image = Image.open(image_path)
    width, height = image.size

    if "width" in metrics:
        computed_metrics["width"] = width

    if "height" in metrics:
        computed_metrics["height"] = height

    pixels = np.array(image)

    # Convert to grayscale for some metrics
    grayscale_image = image.convert("L")
    stat = ImageStat.Stat(grayscale_image)

    # Compute brightness (average grayscale value)
    if "brightness" in metrics:
        brightness = stat.mean[0]
        computed_metrics["brightness"] = brightness

    # Compute contrast (standard deviation of grayscale values)
    if "contrast" in metrics:
        contrast = stat.stddev[0]
        computed_metrics["contrast"] = contrast

    # Sharpness (variance of the Laplacian)
    if "sharpness" in metrics:
        sharpness = np.var(cv2.Laplacian(pixels, cv2.CV_64F))
        computed_metrics["sharpness"] = sharpness

    # Compute average RGB values
    if "average_red" in metrics:
        try:
            avg_r = np.mean(pixels[:, :, 0])
        except IndexError:  # Image is grayscale
            avg_r = 0
        computed_metrics["average_red"] = avg_r

    if "average_green" in metrics:
        try:
            avg_g = np.mean(pixels[:, :, 1])
        except IndexError:
            avg_g = 0
        computed_metrics["average_green"] = avg_g

    if "average_blue" in metrics:
        try:
            avg_b = np.mean(pixels[:, :, 2])
        except IndexError:
            avg_b = 0
        computed_metrics["average_blue"] = avg_b

    return computed_metrics
