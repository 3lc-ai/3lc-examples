# Copyright 2026 3LC Inc. All rights reserved.

"""Friday-afternoon playground: three deferred SPEC ideas that already render today.

All three reuse the ``semantic_segmentation`` sample type as *storage* and ride the
existing semseg interface, so they need **zero** frontend work to show up in the
Dashboard:

1. Error map (SPEC §7.2) — per-pixel ``pred != gt`` (void excluded), stored as a
   binary ``{correct, incorrect, ignore}`` segmentation. ``incorrect`` is baked red,
   ``correct``/``ignore`` transparent, so it renders as a red error overlay. The
   ``incorrect`` layer's ``pixel_count`` is a free per-image error-fraction metric
   (also written as a scalar ``error_fraction`` column for sorting/filtering).

2. Per-class probability heatmaps (SPEC §6, the "bin trick") — the model's softmax
   probability map for a class is a float-per-pixel field; COCO RLE is binary, so we
   quantize probabilities into N disjoint bands and store the result as a
   ``SemanticSegmentation`` over bin "classes", with a precomputed **turbo** color
   ramp baked into each bin's ``display_color`` (low bins low-alpha so the image
   shows through). Renders as a heatmap with no UI change; the per-bin
   ``pixel_counts`` are a globally-queryable per-image confidence histogram — the
   deciding advantage over an opaque, on-demand image column. (For this 2-class model
   only P(pet) is emitted; P(background) = 1 - P(pet) is its exact inverse.)

3. Per-class embeddings (SPEC §7.1) — masked average pooling of the bottleneck
   feature map per class → one D-dim vector per class, all in the same feature space
   ("whole image" global-average pool is just the special case the base example
   already collects). Two mask views are stored, paired with their own mask source so
   GT and pred never cross: ``gt_masked_embedding`` (the model's representation of each
   class's *true* region — right for curation) and ``pred_masked_embedding`` (its
   representation of the region it *predicted* — right for error analysis). Each is its
   own top-level **jagged (C, 2) column** (fixed inner dim 2, one entry per layer); the
   array signature (outer length == layer count) associates each with the predicted
   layers, so they need not be nested in ``instance_properties``.

   List-native reduction (reducing per-row lists of vectors into one shared space) is
   not built yet, so we do by hand what that reducer would do: **extract the matrix +
   keep a source index map** — flatten every (group, row, layer) vector into one matrix,
   fit a single PaCMAP over the union of *both* mask views, then scatter back to per-row
   ``(C, 2)`` per group. One reducer over the union ⇒ GT- and pred-masked embeddings
   share **one** 2D space, so they are directly comparable (per-sample GT-vs-pred drift),
   as are all classes/rows (the SPEC's travel-distance requirement).

The canonical example (``train_unet.py``) stays untouched; this is a separate,
self-contained exploration that writes its own Run. It saves a checkpoint so the
column-building code can be iterated on with ``--skip-train`` (no retraining).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tlc
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
import pacmap
from matplotlib import colormaps
from tlc.schemas import MapElement
from tlc.schemas.values import DimensionNumericValue, Float32Value
from tlc.sample_types import (
    SemanticSegmentation,
    SemanticSegmentationSampleType,
    semseg_classes,
    void_id,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

# Reuse the model, dataset wrapper and constants from the canonical example so this
# stays a thin exploration rather than a fork of the training code.
from train_unet import (
    DATASET_NAME,
    FOREGROUND_CLASS_ID,
    GT_CLASSES,
    IGNORE_CLASS_ID,
    IGNORE_INDEX,
    NUM_CLASSES,
    PREDICTED_CLASSES,
    PROJECT_NAME,
    PetsSegDataset,
    TinyUNet,
    seed_everything,
)

# The predicted segmentation is stored as exactly these class-ordered layers (always-C,
# SPEC §2.1): layer i is class PREDICTED_LAYER_IDS[i]. The per-layer embedding array is
# built in this same order so embedding[i] lines up with layer i.
PREDICTED_LAYER_IDS = sorted(PREDICTED_CLASSES)

# Error-map "classes": correct is the transparent background, ignore is the void
# (GT border) region also rendered transparent, incorrect is baked red so the
# overlay reads as "here is where the model is wrong".
ERROR_CORRECT, ERROR_INCORRECT, ERROR_IGNORE = 0, 1, 2
ERROR_CLASSES = semseg_classes(
    {
        ERROR_CORRECT: MapElement("correct"),
        ERROR_INCORRECT: MapElement("incorrect", display_color="#e6194bff"),
        ERROR_IGNORE: MapElement("ignore"),
    },
    background=ERROR_CORRECT,
    void=ERROR_IGNORE,
)

# Probability-heatmap config: which predicted classes to emit a heatmap for, and how
# many quantization bands. Only the foreground (pet) class — for a 2-class model
# P(background) is just 1 - P(pet), so its heatmap is the exact inverse and adds nothing.
HEATMAP_CLASS_NAMES = {FOREGROUND_CLASS_ID: "pet"}
N_BINS = 10


def turbo_bin_classes(n_bins: int) -> dict[int, MapElement]:
    """Build bin "classes" for a probability heatmap with a baked turbo color ramp.

    Bin ``i`` covers probabilities ``[i/n, (i+1)/n)``. The turbo colormap is sampled
    at the band center and baked into ``display_color`` (path (1) of the UI's
    ``effectiveDisplayColor``), so the column renders as a sequential heatmap with no
    UI colormap support. Low bins get a low alpha so near-zero-probability pixels stay
    see-through and the image underneath shows.
    """
    turbo = colormaps["turbo"]
    classes: dict[int, MapElement] = {}
    for i in range(n_bins):
        center = (i + 0.5) / n_bins
        r, g, b, _ = turbo(center)
        # Ramp alpha in over the bottom third so low-confidence regions fade out.
        alpha = int(255 * min(1.0, (i + 1) / (n_bins / 3)))
        hex_rgba = f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}{alpha:02x}"
        # Map-element names disallow "." (and <>\|:"'?*&), so label bins as integer
        # percent ranges rather than "0.0-0.1".
        lo, hi = round(100 * i / n_bins), round(100 * (i + 1) / n_bins)
        classes[i] = MapElement(f"{lo}-{hi}%", display_color=hex_rgba)
    return classes


HEATMAP_CLASSES = turbo_bin_classes(N_BINS)
HEATMAP_CLASS_IDS = sorted(HEATMAP_CLASSES)


def build_error_map(pred_map: np.ndarray, gt_map: np.ndarray) -> tuple[SemanticSegmentation, float]:
    """Binary {correct, incorrect, ignore} segmentation from a (pred, gt) pair.

    Void (GT == border) pixels are excluded from the comparison and labelled
    ``ignore``; the error fraction is over valid (non-void) pixels only.
    """
    height, width = gt_map.shape
    void = IGNORE_CLASS_ID
    valid = gt_map != void
    incorrect = valid & (pred_map != gt_map)

    label_map = np.full((height, width), ERROR_CORRECT, dtype=np.int32)
    label_map[incorrect] = ERROR_INCORRECT
    label_map[~valid] = ERROR_IGNORE

    n_valid = int(valid.sum())
    error_fraction = float(incorrect.sum()) / n_valid if n_valid else 0.0
    seg = SemanticSegmentation(
        image_width=width, image_height=height, label_map=label_map, class_ids=[ERROR_CORRECT, ERROR_INCORRECT, ERROR_IGNORE]
    )
    return seg, error_fraction


def build_heatmap(prob_map: np.ndarray) -> SemanticSegmentation:
    """Quantize a (H, W) probability map in [0, 1] into N disjoint bands (the bin trick)."""
    height, width = prob_map.shape
    bins = np.clip((prob_map * N_BINS).astype(np.int32), 0, N_BINS - 1)
    return SemanticSegmentation(image_width=width, image_height=height, label_map=bins, class_ids=HEATMAP_CLASS_IDS)


def masked_average_pool(features: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
    """Mean of the (C, h, w) feature vectors over the pixels where ``mask`` is set.

    ``mask`` is the class mask downsampled to the feature-map resolution. An absent
    class (empty mask) pools to a zero vector — rare here (background and pet are in
    almost every image), and noted as a known wart rather than special-cased.
    """
    channels = features.shape[0]
    mask = mask.to(features.dtype)
    denom = mask.sum()
    if denom < 1:
        return np.zeros(channels, dtype=np.float32)
    pooled = (features * mask[None]).sum(dim=(1, 2)) / denom
    return pooled.cpu().numpy().astype(np.float32)


def _downsample_to(label_map: np.ndarray, height: int, width: int, device: torch.device) -> torch.Tensor:
    """Nearest-neighbour downsample a (H, W) integer label map to the feature-map resolution."""
    small = F.interpolate(
        torch.from_numpy(label_map.astype(np.float32))[None, None], size=(height, width), mode="nearest"
    )
    return small.squeeze().to(device)


def train_quick(model: nn.Module, loader: DataLoader, device: torch.device, epochs: int) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    model.train()
    for epoch in range(epochs):
        running = 0.0
        for images, labels in tqdm(loader, desc=f"train {epoch}", leave=False):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running += loss.item() * images.shape[0]
        print(f"epoch {epoch}  train_loss={running / len(loader.dataset):.4f}")


def reduce_groups_to_shared_2d(groups: dict[str, list[np.ndarray]]) -> dict[str, list[np.ndarray]]:
    """Reduce several per-row ``(C, D)`` per-layer embedding groups into ONE shared 2D space.

    This is the by-hand stand-in for the not-yet-built list-native reducer: extract the
    matrix and keep a source index map (SPEC §7.1). Every ``(group, row, layer)`` vector is
    flattened into one big ``(sum_g N_g*C, D)`` matrix, a *single* PaCMAP is fit over the
    union, and the result is scattered back per group/row via the row-major index map. One
    reducer over the union of *all* groups ⇒ the GT-masked and pred-masked embeddings live
    in the same 2D space, so they are directly comparable (e.g. per-sample GT-vs-pred
    representation drift), as are all classes/rows.
    """
    names = list(groups)
    stacks = {name: np.stack(groups[name]) for name in names}  # each (N, C, D)
    n_rows, n_layers, dim = stacks[names[0]].shape
    matrix = np.concatenate([stacks[name].reshape(n_rows * n_layers, dim) for name in names]).astype(np.float32)
    reduced = pacmap.PaCMAP(n_components=2).fit_transform(matrix)  # (len(names)*N*C, 2)
    out: dict[str, list[np.ndarray]] = {}
    for i, name in enumerate(names):
        block = reduced[i * n_rows * n_layers : (i + 1) * n_rows * n_layers]
        out[name] = list(block.reshape(n_rows, n_layers, 2).astype(np.float32))
    return out


def layer_embedding_schema(display_name: str) -> tlc.Schema:
    """Schema for a standalone per-layer embedding column: a jagged array of 2D points.

    Shape (2, -1) in tlc size order: size0 is the innermost dim (each embedding is a fixed
    2D point), size1 is arbitrary (one embedding per layer, count == number of layers).
    tlc orders dims innermost-first, so the stored numpy shape is (n_layers, 2) — outer
    axis = layer count. That outer length matching the segmentation's layer count is what
    lets the frontend's ArraySignature associate this column with the predicted layers,
    so the embeddings can live in their own top-level column rather than buried in
    ``instance_properties`` (SPEC §7.1).

    Note: **no** ``number_role="nn_embedding"`` here. That role marks a column as a *raw*
    reduction source — the Dashboard hides it and waits for a reduced companion — so an
    already-reduced 2D column tagged that way renders empty. Reduced coordinates carry an
    empty role (matching what ``reduce_embeddings`` emits for its ``*_pacmap`` output), and
    the array-signature-to-layer association keys off the array sizes, not the role.
    """
    return tlc.Schema(
        display_name=display_name,
        value=Float32Value(),
        size0=DimensionNumericValue.fixed_size(2),  # each embedding is 2D (fixed inner dim)
        size1=DimensionNumericValue(),  # jagged: arbitrary number of them — one per layer
    )


def collect_diagnostics(
    run: tlc.Run, table: tlc.Table, model: nn.Module, device: torch.device, image_size: int
) -> None:
    """Write the three diagnostic column families for every sample of ``table``."""
    sample_type = SemanticSegmentationSampleType()

    predicted_rows: list[dict] = []
    error_maps: list[dict] = []
    error_fractions: list[float] = []
    heatmaps: dict[int, list[dict]] = {cid: [] for cid in HEATMAP_CLASS_NAMES}
    # Per row: (C, D) arrays of per-layer embeddings, class-ordered to match the predicted
    # layers. Two mask views are kept separately — GT-masked (the model's representation of
    # the *true* region of each class) and pred-masked (its representation of the region it
    # *predicted*). Both go through one shared reducer after the loop (see below).
    gt_masked: list[np.ndarray] = []
    pred_masked: list[np.ndarray] = []

    captured: dict[str, torch.Tensor] = {}
    handle = model.bottleneck.register_forward_hook(lambda _m, _i, out: captured.__setitem__("emb", out))

    model.eval()
    with torch.no_grad():
        for idx in tqdm(range(len(table)), desc="collect", leave=False):
            row = table[idx]
            image = row["image"].convert("RGB")
            seg: SemanticSegmentation = row["segmentation"]
            width, height = image.size

            image_tensor = (
                torch.from_numpy(np.asarray(image.resize((image_size, image_size)))).permute(2, 0, 1).float() / 255.0
            )
            logits = model(image_tensor[None].to(device))
            features = captured["emb"].squeeze(0)  # (C, h, w) bottleneck feature map

            logits = F.interpolate(logits, size=(height, width), mode="bilinear", align_corners=False)
            probs = logits.softmax(dim=1).squeeze(0).cpu().numpy()  # (NUM_CLASSES, H, W)
            pred_map = probs.argmax(axis=0).astype(np.int32)

            # Predicted segmentation, stored as exactly C class-ordered layers.
            predicted_rows.append(
                sample_type.to_row(
                    SemanticSegmentation(
                        image_width=width, image_height=height, label_map=pred_map, class_ids=PREDICTED_LAYER_IDS
                    )
                )
            )

            # 1. Error map (vs GT, void excluded).
            error_seg, error_fraction = build_error_map(pred_map, seg.label_map)
            error_maps.append(sample_type.to_row(error_seg))
            error_fractions.append(error_fraction)

            # 2. Per-class probability heatmaps (bin trick).
            for class_id in HEATMAP_CLASS_NAMES:
                heatmaps[class_id].append(sample_type.to_row(build_heatmap(probs[class_id])))

            # 3. Per-layer masked-average-pooled bottleneck embeddings, ordered to match the
            #    predicted layers. Pool over both the GT region and the predicted region of
            #    each class — paired with its own mask source, no GT/pred crossing (SPEC §7.1).
            #    Both masks downsampled to the feature-map resolution. (Absent classes pool to
            #    a zero vector for now — the §7.1.1 missing-data policy is not applied yet.)
            fh, fw = features.shape[1], features.shape[2]
            gt_small = _downsample_to(seg.label_map, fh, fw, device)
            pred_small = _downsample_to(pred_map, fh, fw, device)
            gt_masked.append(
                np.stack([masked_average_pool(features, gt_small == class_id) for class_id in PREDICTED_LAYER_IDS])
            )
            pred_masked.append(
                np.stack([masked_average_pool(features, pred_small == class_id) for class_id in PREDICTED_LAYER_IDS])
            )

    handle.remove()

    # Reduce GT- and pred-masked embeddings through ONE shared reducer so they share a 2D
    # space (comparable to each other and across classes/rows). Each becomes its own
    # top-level jagged (C, 2) column; the array signature (outer length == layer count)
    # associates each with the predicted layers — no need to nest in instance_properties.
    print("Reducing GT- and pred-masked embeddings into one shared 2D space (PaCMAP)...")
    reduced = reduce_groups_to_shared_2d({"gt": gt_masked, "pred": pred_masked})

    metrics: dict[str, list] = {
        "predicted_segmentation": predicted_rows,
        "error_map": error_maps,
        "error_fraction": error_fractions,
        "gt_masked_embedding": [coords.tolist() for coords in reduced["gt"]],
        "pred_masked_embedding": [coords.tolist() for coords in reduced["pred"]],
    }
    schema: dict[str, tlc.Schema] = {
        "predicted_segmentation": SemanticSegmentationSampleType.schema(
            PREDICTED_CLASSES, display_name="predicted segmentation"
        ),
        "error_map": SemanticSegmentationSampleType.schema(ERROR_CLASSES, display_name="error map"),
        "gt_masked_embedding": layer_embedding_schema("GT-masked embedding"),
        "pred_masked_embedding": layer_embedding_schema("pred-masked embedding"),
    }
    for class_id, name in HEATMAP_CLASS_NAMES.items():
        metrics[f"prob_heatmap_{name}"] = heatmaps[class_id]
        schema[f"prob_heatmap_{name}"] = SemanticSegmentationSampleType.schema(
            HEATMAP_CLASSES, display_name=f"P({name}) heatmap"
        )

    run.add_metrics(metrics, schema=schema, foreign_table_url=table.url)


def _load_table(table_name: str) -> tlc.Table:
    """Load a project table, preferring the latest revision.

    ``.latest()`` reconciles the index, which can trip a native/Python mismatch
    (``IndexEngine.get_removed_urls``) when a running ``3lc service`` has reindexed —
    so fall back to the named revision if that surfaces. (This playground does not
    depend on consuming curation; the canonical ``train_unet.py`` keeps ``.latest()``.)
    """
    table = tlc.Table.from_names(table_name=table_name, dataset_name=DATASET_NAME, project_name=PROJECT_NAME)
    try:
        return table.latest()
    except AttributeError as exc:
        print(f"  .latest() unavailable ({exc}); using the named revision of {table_name!r}")
        return table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8, help="Quick-train epochs (enough for a legible heatmap)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--checkpoint",
        default="tiny_unet_diag.pt",
        help="Where to save/load the quick-trained model so column code can be iterated without retraining",
    )
    parser.add_argument("--skip-train", action="store_true", help="Load the checkpoint instead of training")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    train_table = _load_table("train")
    val_table = _load_table("val")

    model = TinyUNet(NUM_CLASSES).to(device)
    checkpoint = Path(args.checkpoint)
    if args.skip_train and checkpoint.exists():
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        print(f"Loaded checkpoint {checkpoint}")
    else:
        train_dataset = PetsSegDataset(train_table, args.image_size, augment=True)
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
        )
        train_quick(model, train_loader, device, args.epochs)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), checkpoint)
        print(f"Saved checkpoint {checkpoint}")

    run = tlc.init(PROJECT_NAME, description="Diagnostics playground: error map, prob heatmaps, per-layer embeddings")
    # The per-layer embeddings are reduced into a shared 2D space inside collect_diagnostics
    # (the by-hand list-native reduction) and stored on the predicted-segmentation column, so
    # there is no separate run.reduce_embeddings step here.
    collect_diagnostics(run, val_table, model, device, args.image_size)

    run.set_status_completed()
    print(f"Run: {run.url}")


if __name__ == "__main__":
    main()
