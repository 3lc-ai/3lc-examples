from __future__ import annotations

from collections import defaultdict

import cv2
import numpy as np
from PIL import Image, ImageStat
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans


def diversity(labels, embeddings):
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


def uniqueness(labels, embeddings):
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

    return [int(idx) for idx in final_traversal_order]


def compute_image_metrics(image_path: str):
    """Return a dict of image metrics for the given image path."""
    image = Image.open(image_path)
    width, height = image.size
    pixels = np.array(image)

    # Convert to grayscale for some metrics
    grayscale_image = image.convert("L")
    stat = ImageStat.Stat(grayscale_image)

    # Compute brightness (average grayscale value)
    brightness = stat.mean[0]

    # Compute contrast (standard deviation of grayscale values)
    contrast = stat.stddev[0]

    # Sharpness (variance of the Laplacian)
    sharpness = np.var(cv2.Laplacian(pixels, cv2.CV_64F))

    # Compute average RGB values
    try:
        avg_r = np.mean(pixels[:, :, 0])
        avg_g = np.mean(pixels[:, :, 1])
        avg_b = np.mean(pixels[:, :, 2])
    except IndexError:  # Image is grayscale
        avg_r = avg_g = avg_b = 0

    return {
        "width": width,
        "height": height,
        "brightness": brightness,
        "sharpness": sharpness,
        "contrast": contrast,
        "average_red": avg_r,
        "average_green": avg_g,
        "average_blue": avg_b,
    }
