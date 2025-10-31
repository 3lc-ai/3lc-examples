import numpy as np
from pandaset import DataSet

DATASET_ROOT = "D:/Data/pandaset"

dataset = DataSet(DATASET_ROOT)

unique_labels = set()

for sequence in dataset.sequences():
    seq = dataset[sequence]
    seq.cuboids.load()

    for frame_cuboids in seq.cuboids.data:
        if frame_cuboids is None or "label" not in frame_cuboids or len(frame_cuboids) == 0:
            continue
        labels = frame_cuboids["label"].values
        if labels.size == 0:
            continue
        frame_unique = np.unique(labels)
        unique_labels.update(lbl for lbl in frame_unique if isinstance(lbl, str) and lbl)

print(f"Unique labels ({len(unique_labels)}):")
for lbl in sorted(unique_labels):
    print(lbl)
