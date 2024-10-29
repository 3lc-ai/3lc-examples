import numpy as np
from collections import defaultdict

import tlc


def compute_object_detection_sample_weights(table: tlc.Table, alpha=1.0, include_zero_weighted_images=False):
    # Step 1: Calculate class frequency across all bounding boxes
    class_counts = defaultdict(int)
    total_boxes = 0

    # images: a list of dictionaries, each containing image data with bounding boxes and their labels
    for row in table.table_rows:
        bbs = row["bbs"]
        if row["weight"] == 0 and not include_zero_weighted_images:
            continue
        for bbox in bbs["bb_list"]:
            label = bbox["label"]
            class_counts[label] += 1
            total_boxes += 1

    # Calculate class frequencies and inverse class weights
    class_frequencies = {c: class_counts[c] / total_boxes for c in class_counts}
    inverse_class_weights = {c: 1.0 / f for c, f in class_frequencies.items()}

    # Step 2 & 3: Assign weights to each image based on the bounding boxes within it
    image_weights = []
    for row in table.table_rows:
        bbox_weights = [inverse_class_weights[bbox["label"]] for bbox in row["bbs"]["bb_list"]]
        avg_weight = np.mean(bbox_weights)
        image_weights.append(avg_weight)

    # Step 4: Apply smoothing factor
    smoothed_weights = [(1 - alpha) + alpha * w for w in image_weights]

    # Step 5: Normalize the image-level weights
    normalized_weights = np.array(smoothed_weights) / np.sum(smoothed_weights)

    return normalized_weights


table = tlc.Table.from_names("initial", "train", "DataCleaningChallenge")
alpha = 0.8  # Smoothing factor
weights = compute_object_detection_sample_weights(table, alpha)
print(weights)
