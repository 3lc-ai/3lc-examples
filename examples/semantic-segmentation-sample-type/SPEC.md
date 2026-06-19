# Semantic Segmentation as an RLE-backed Sample Type — Spec

Status: **draft for review** · Driving example: this POC (`semantic-segmentation-sample-type`, Oxford-IIIT Pets)

This document specs semantic segmentation as a first-class 3LC sample type that
**reuses the instance-segmentation RLE wire format as shared storage, without
inheriting instance-segmentation semantics**. The POC in this repo is both the
worked example and the staging area: things proven here are candidates to move
into core `tlc` (Python) and `tlc-ui` (dashboard).

The code has since **graduated to core**: the sample type and ergonomic class-map
API live in `tlc.sample_types._semantic_segmentation`, the metrics helper in
`tlc.helpers.semantic_segmentation_metrics` (`~/projects/tlc-monorepo`), and the UI
in `~/projects/tlc-ui`; this directory keeps only the runnable `scripts/*` (table
generation + training + metrics collection). The §2 references below to
`src/semseg_sample_type/*` describe the original POC state (preserved here as the
design narrative) — that code now lives in core per [§10](#10-roadmap-poc--core).

---

## 1. Thesis and scope

**Thesis:** A semantic segmentation is a dense `(H, W)` integer label map — an
exhaustive partition of the image. It can be *stored* byte-for-byte in the
instance-segmentation RLE layout (`{image_height, image_width,
instance_properties.label, rles}`), one RLE per class present, **provided two
invariants hold**:

1. **Exhaustive partition** — every pixel has exactly one label.
2. **0-or-1 instance per class** — each class is one region, not a set of
   instances.

Under these, the label-map ⟷ RLE-set mapping is a clean bijection, and we reuse
storage, RLE rendering, and the per-layer machinery for free.

**Non-goal:** forcing instance semantics on the user. Storage is shared;
*semantics* (metrics, rendering affordances, special-class meaning) are distinct
and dispatch on `sample_type == "semantic_segmentation"`. This is the line the
whole spec defends: **shared storage, not shared semantics.**

### Two special classes

Two classes carry meaning beyond "a label":

- **background** — the implicit/default class; "absence of a labeled object."
- **void** (a.k.a. border / ignore) — excluded from metrics; not a prediction
  target; GT-only.

How these are represented is [§3](#3-special-classes).

---

## 2. Current state

> **Update — graduated to core.** The sample type, the ergonomic class-map API, and
> a confusion-matrix metrics helper now live in core `tlc`
> (`tlc.sample_types._semantic_segmentation`, `tlc.helpers.semantic_segmentation_metrics`);
> the UI ships an `ImageInterfaceSemanticSegmentation` (`tlc-ui`). The POC package
> re-exports core, so the description below still reflects the design. Two decisions
> landed beyond the original POC: the UI discriminates on a schema **`composite_role`**
> rather than `sample_type` ([§5](#5-ui-rendering-and-dispatch)), and the segmentation is
> stored as a **fixed set of exactly C class-ordered layers** ([§2.1](#21-always-c-layers)).

Implemented in `src/semseg_sample_type/sample_type.py` (the staging copy; graduated to core):

- `SemanticSegmentation` dataclass — dense `(H, W)` label map, the sample form.
- `SemanticSegmentationSampleType` — `to_row` encodes one RLE per `np.unique`
  class (`sample_type.py:82-94`); `from_row` zero-inits a label map and paints
  each class's mask back (`sample_type.py:104-109`); registered as
  `"semantic_segmentation"` and stamps that on the schema (`sample_type.py:151`).
- `scripts/ingest_oxford_pets.py` — trimap → `{0: background, 1: pet, 2: border}`,
  written via the sample type. Background already uses `display_color="#00000000"`
  (transparent).
- `scripts/train_unet.py` — TinyUNet; collects predicted-seg-as-RLE, mean IoU,
  pet IoU, loss, entropy, and a pooled bottleneck embedding; reduces embeddings
  with PaCMAP (one reducer for a shared space, `train_unet.py:397`).

**How the invariants hold today:** exhaustive partition is guaranteed on write
(the sample form is dense) and on read (`from_row` zero-init). 0-or-1 per class is
guaranteed on write (masks built from `label_map == class_id` over `np.unique` are
mutually exclusive). Neither is *enforced* against hostile input; overlap on read
is silently order-dependent (last class wins).

### 2.1 Always-C layers

**Decision: store exactly C class-ordered layers (the value-map universe), one RLE per
class, with an empty RLE for classes absent from a given image.** The
`SemanticSegmentation` sample carries an optional ordered `class_ids` universe; when set,
`to_row` emits exactly those layers in that order and `from_row` recovers the universe
from the stored `instance_properties.label`. (`None` keeps the legacy compact form — only
the `np.unique` classes present — for back-compat.)

Why fix the layer set early:

- **Stable layer ↔ class alignment across rows.** Layer *i* always means the same class,
  so the per-layer stat columns (`pixel_counts`, bbox extents, …) are class-aligned and
  globally queryable/comparable — the per-`np.unique` ordering varied per image.
- **The editor can fix the constraints** ([§5.4](#54-editing-constraints)): the layer set
  is immutable (no add/delete) and a painted region is exclusive (it erases the others) —
  turning the two invariants of [§1](#1-thesis-and-scope) from write-time conventions into
  enforced editing rules.
- **Cheap:** an absent class is an all-zero mask → a trivially small RLE.

The wire format and the bijection are unchanged; this only fixes *which* layers are
written. It lightly pre-commits against the background-not-stored optimization
([§3](#3-special-classes)), but that is already deferred and "always C" is consistent with
"keep background explicit for now" (it would become "fixed C−1 real layers + implicit
background").

---

## 3. Special classes — reserved `internal_name`s

**Decision: reserved `internal_name` constants on the class map, not a new `role`
field.**

Rationale (grounded in `tlc-monorepo`):

- `MapElement.internal_name` is already "the primary identifier of the value map
  item" (`src/tlc/_core/objects/table.py:1820`), already serializes
  (`to_minimal_dict`), already flows to the dashboard, and survives edited-table
  merges. There is existing precedent for reserved-name sentinels — the indexing
  table injects `MapElement(internal_name="None", ...)` for its `-1` slot
  (`indexing_table.py:573`).
- A `role` field would be a new attribute threaded through `MapElement`
  serialization, schema JSON, edited-table merge, and the dashboard — a core
  change for something `internal_name` already encodes 1:1.
- The two are equivalent in expressive power for the 1:1 case. `internal_name →
  role` is a pure read-side mapping, so we can evolve into explicit roles later
  non-breakingly if needed.

### Contract

- Reserved constants, namespaced to avoid colliding with a user's genuine
  "background" class, e.g. `TLC_SEMSEG_BACKGROUND`, `TLC_SEMSEG_VOID`.
  (border / ignore all fold into **void**.)
- **Semantics each name carries** (this is the contract the UI and metrics both
  read):
  - *background*: implicit/default fill; rendered transparent; the id a future
    not-stored optimization would drop.
  - *void*: excluded from loss / IoU / confusion; not a prediction target;
    GT-only. Replaces today's hardcoded `IGNORE_CLASS_ID = 2`
    (`train_unet.py:38`), which would instead be read off the value map.
- **Uniqueness:** at most one element per reserved name; assert in the sample
  type's `schema()`.
- **Ergonomic API:** wrap the magic strings so users never type them. Strawman
  implemented in `src/semseg_sample_type/class_map.py`:
  - `semseg_classes(classes, *, background=<id>, void=<id>)` — builds the value
    map, tagging the named ids with the reserved `internal_name` (their human
    label is preserved as `display_name`; background defaults to transparent).
  - `background_id(classes)` / `void_id(classes)` / `real_class_ids(classes)` —
    read the roles back off a value map by scanning for the reserved name, so
    metrics / training / the UI bridge never re-specify which class is which
    (replaces the hardcoded `IGNORE_CLASS_ID`).
  - `SemanticSegmentationSampleType.schema(classes, background=<id>, void=<id>)`
    accepts the same kwargs as a one-call shortcut.
  - Reserved constants are underscore-namespaced (`__tlc_semseg_background__`,
    `__tlc_semseg_void__`) because `.`/`:` are illegal in map-element names
    (`_DISALLOWED_MAP_ELEMENT_NAME_CHARS`).

This is purely a convention over the class map — **no change to the wire format
or the bijection.**

### Background storage: explicit for now

`from_row`'s zero-init already makes background the implicit default, and it's
already rendered transparent, so the background RLE is redundant and (being the
largest region) the most expensive. **Decision: keep storing it explicitly for
now**; dropping it is a clean future optimization that the reserved
`background_id` sets up for free. (Note: if generalized, `from_row` must init the
label map with `background_id`, not `0`.)

---

## 4. Metrics

### 4.1 Semantic vs instance metrics

| Aspect | Semantic seg | Instance seg |
|---|---|---|
| Unit of comparison | pixel | instance (region) |
| Matching | none — pixels are position-aligned | required (IoU-based assignment) |
| Confidence | unused for the hard metric (argmax) | central — ranks the PR curve |
| Core statistic | C×C pixel confusion matrix | matched/unmatched TP/FP/FN per score threshold |
| IoU's role | **the score itself** (per class) | **the matching threshold** |
| Headline | mIoU | mAP@[.5:.95] |
| Counting | classes fixed, each 0/1× | variable instance count |

Consequences:

- **No `confidence` in the semseg wire layout.** Instance `confidence` is
  per-instance (one per RLE). Semseg's soft signal is *per-pixel* and doesn't
  reduce to one-number-per-class. So semseg populates `label` per RLE but
  **never `confidence`**; its uncertainty signals (entropy, loss, margin) are
  per-image scalars in ordinary metric columns.
- **Panoptic Quality** is the bridge: semseg-as-RLE is the all-"stuff" special
  case of panoptic storage.

### 4.2 The confusion matrix is the primitive

Every semseg metric (pixel acc, per-class accuracy/recall, per-class IoU, mIoU,
frequency-weighted IoU, Dice/F1) is a readout of one **C×C pixel confusion
matrix** (C = real classes, void excluded). Store/compute that, derive the rest.

**Aggregation duality (gotcha to encode explicitly):**

- **Cumulative** — accumulate the confusion matrix over all pixels of all images,
  then compute IoU. This is what every benchmark reports.
- **Per-image then averaged** — noisier, biased toward small objects. This is
  what the POC currently returns (`collect_metrics` → `np.mean(ious)`,
  `train_unet.py:264`).

3LC's per-sample story *wants* per-image (to sort/curate/filter), but the
headline must be cumulative. **Carry both:** per-image IoU columns for curation +
a per-image confusion matrix that sums dashboard-side for the correct cumulative
mIoU.

The reserved special classes plug straight in: **void** → dropped from the matrix
(read off the value map, not hardcoded); **background-in-mIoU** is a documented
toggle (VOC counts background; Cityscapes excludes it).

### 4.2.1 The C×C matrix is the single primitive — storage, generalization, rendering

**Decision: the per-image C×C confusion matrix is *the* stored/virtual primitive;
every standard metric derives from it as a cheap array reduction.** This is the one
expensive two-column RLE operation — mIoU, per-class IoU, pixel accuracy, and Dice
are then array math over the matrix, not repeated RLE passes.

**How much the classification confusion mechanism generalizes** (grounded in
`tlc-ui`): the classification workflow (`ConfusionMatrixWorkflow.tsx`) takes a
GT-label and a predicted-label column — **one `(gt, pred)` pair per row** —
computes the `occurrence` virtual column
(`OperationOccurrence.ts`) to cross-tabulate pairs across rows, and renders via
`CHARTALIAS_CONFUSION_MATRIX_2D`. A semseg row is not one pair but H×W pairs, so a
single image is *already* a full C×C matrix; the per-row occurrence count does not
apply at the pixel level. But two things carry over:

- **Cross-row aggregation is "sum over rows."** Occurrence counting is sum-over-rows
  with a scalar increment; the semseg dataset matrix is sum-over-rows with a matrix
  add. Same primitive, matrix payload → **cumulative mIoU = sum the per-image C×C
  across rows**, benchmark-correct.
- **The grid chart rendering** is source-agnostic, so the *aggregate* dataset matrix
  can likely reuse the existing confusion-matrix chart.

**Storage:** per-image C×C as a small list-of-lists / array column (C typically
<20–50). **Home:** computed as the UI virtual column for live recompute under
curation; the Python helper computes the identical matrix offline for the
training-time headline (same formula, two homes — per §4.3).

**Border-excluded family is a parallel C×C.** Border-excluded metrics ([§7.2](#72-error-map--border-excluded-error-map--border-excluded-accuracy))
need a *spatial* mask, which a pre-aggregated matrix has discarded — so they do not
derive from the plain matrix. Frame them as **one C×C per pixel-selection policy**
(all-valid; border-excluded): same machinery, different valid-mask.

**Native C×C rendering** (per-cell, for an array-matrix column) does not exist
today and is a worthwhile, broadly-reusable UI addition (it would serve any per-row
matrix, not just semseg). It is **not on the critical path**: the aggregate matrix
reuses the existing chart, and curation is driven by the scalar readouts.

### 4.3 Both Python and UI

**Decision: metric operations live on both sides, by what inputs they need.**

- **Python — opt-in training helper.** A small library helper (graduating the
  ~40 lines at `train_unet.py:166-186`) that builds the per-image confusion
  matrix and derives IoU/accuracy at `run.add_metrics` time. Benchmark-correct,
  reproducible, baked at collection time.
- **UI — virtual columns (the interactive surface).** The same metrics as derived
  columns over (pred RLE × GT RLE × void/bg policy), so they **recompute live**
  when GT is curated/relabeled or a policy/threshold is toggled — no re-run.

What goes where is decided by inputs ([§8](#8-architecture-split)).

---

## 5. UI rendering and dispatch

### 5.1 Grounding (tlc-ui)

- There are two existing segmentation render paths:
  - `ImageInterfaceSingleLayerSegmentation` — attaches on
    `STRING_ROLE_SEGMENTATION_MASK_URL`
    (`ImageInterfaceSingleLayerSegmentation.ts:45`): a **URL/image-backed** mask,
    fetched on demand (no global ops). **Left in place, untouched.**
  - `ImageInterfaceMultiLayerSegmentation` — the **RLE** path. Its
    `staticTryAttach` dispatches on **wire shape, not `sample_type`**: it only
    checks `image_width`/`image_height`/`instance_properties`
    (`ImageInterfaceMultiLayerSegmentation.ts:163-181`).
- Historically **`"semantic_segmentation"` was invisible to the UI** — the column
  rendered through the instance multilayer interface with meaningless-per-class
  affordances. This is now resolved by the dedicated interface ([§5.2](#52-new-semseg-interface));
  the discriminant is a schema **`composite_role`**, not `sample_type` (see below).
- Per-layer stats (`min_xs/max_xs/min_ys/max_ys`, `pixel_counts`, `island_counts`,
  `circumferences`) are carried as **column data**
  (`ImageInterfaceMultiLayerSegmentationConstants.ts:23-31`, computed at
  `ImageInterfaceMultiLayerSegmentation.ts:420-427`) — i.e. **globally
  queryable**, unlike on-demand-fetched images.
- Coloring is **categorical** (`EnumValue.effectiveDisplayColor`,
  `EnumValue.ts:271-310`): explicit `display_color` → palette-by-position →
  hash(internal_name) → hash(key). No continuous/sequential colormap by value.

### 5.2 New semseg interface

**Decision: a new `ImageInterfaceSemanticSegmentation` that subclasses
`ImageInterfaceMultiLayerSegmentation`**, so RLE decode, layer iteration, and the
per-layer derived arrays are inherited.

- **Wins over instance seg** purely via registration order — `staticTryAttachPrivate`
  returns the first non-null (`ListInterface.ts:104-113`). Register the new
  interface *before* `MultiLayerSegmentation` in `Intergalactic.ts` (ahead of
  line 596).
- **Narrow predicate:** same wire-shape checks **plus** a schema
  `composite_role === "SemanticSegmentation"` gate. Returns null otherwise, so
  instance columns fall through to `MultiLayerSegmentation` — no regression.
- **Discriminate on `composite_role`, not `sample_type`.** `sample_type` names the
  Python (de)serializer and stays `"semantic_segmentation"`; the frontend instead
  reads the schema's `composite_role` (a pre-existing, top-level `Schema` attribute
  that serializes like `sample_type` and is read via `SchemaHelper.staticCompositeRole`).
  Semseg is its first UI consumer. The Python `schema()` stamps both. This keeps the
  UI discriminant decoupled from the serializer identity.
- **Override surface:** suppress instance-only affordances ([§5.4](#54-editing-constraints));
  honor reserved bg/void names (transparent background, void styling); host the
  metric virtual columns.

### 5.3 UI scope (the two missing pieces)

1. The subclassed `ImageInterfaceSemanticSegmentation`.
2. A two-column **confusion-matrix local operation** (pred RLE × GT RLE → per-image
   C×C array) — the single expensive RLE op. The scalar readouts (mIoU, per-class
   IoU, pixel accuracy, Dice) are cheap array reductions over that C×C; the dataset
   matrix is a `GlobalOperation` summing per-image C×C over rows (generalizing
   `OperationOccurrence`'s sum-over-rows). No IoU/confusion/compare operation
   exists today; the framework split is `LocalOperations` (per-row) vs
   `GlobalOperations` (cross-row), and the existing per-layer stats are computed
   inline in the interface, not as operations — so this is genuinely new. See
   [§4.2.1](#421-the-cxc-matrix-is-the-single-primitive--storage-generalization-rendering).
3. *(Optional, reusable)* native per-cell rendering for an array-matrix column —
   the aggregate matrix can reuse the existing confusion-matrix chart, so this is a
   nice-to-have, not a blocker.

### 5.4 Editing constraints

The always-C storage ([§2.1](#21-always-c-layers)) lets the editing widget enforce the
[§1](#1-thesis-and-scope) invariants directly, instead of trusting the writer. Three
capability methods on `ImageInterface` (default to the permissive instance-seg behavior;
overridden by `ImageInterfaceSemanticSegmentation`):

- `canAddLayer()` / `canDeleteLayer()` → **false** for semseg: the layer set is the fixed
  class universe, so the Add/Delete-layer controls are hidden (`PropertyGui2d3dImage`). An
  "empty" class is just an empty layer, not a deletable one.
- `isExclusiveLayerPartition()` → **true** for semseg: painting a region into one class
  **always** erases that region from every other class (`ThreeJsSceneObjectImage` forces
  erase-other-layers on paint), keeping the partition exhaustive and non-overlapping —
  this is enforced, not the user-facing "erase other layers" toggle.

Instance segmentation is untouched (defaults preserve add/delete and the optional erase
toggle). *Erase-mode* semantics for a strict partition (erasing pixels should arguably
repaint them as background) are out of scope for now.

---

## 6. Heatmaps (probability maps) — out of core, supported two ways

Models emit a `C×H×W` 0–1 probability map; there is no native dashboard display.
COCO RLE is **binary**, so a float-per-pixel must be reduced to binary masks.

**The bin trick:** quantize probabilities into N levels (8–16) and slice into
disjoint bands → each pixel in exactly one band → **a heatmap is a
`SemanticSegmentation` over probability-bin "classes."** No new storage type.

- Store the **top-1 confidence map** (one column) for the common case, or one
  column per class for full `C`.
- Storage ≈ one semseg per heatmap (bands disjoint, total = H×W).
- **Global-ops win:** the bands' `pixel_counts` are a per-image confidence
  histogram, globally queryable — the deciding advantage over an image column
  (which is fetched on demand, opaque to global ops).
- **Renders as a heatmap today, zero UI changes:** bake a precomputed color ramp
  (viridis/turbo) into each bin's `display_color` (path (1) of
  `effectiveDisplayColor`). A true `number_role`-driven sequential colormap is
  optional polish, not a blocker.
- **Fallback:** colormapped image column — truly native display, but a baked
  overlay, another file per sample, and no global ops.
- **Caveats:** lossy (display-only; for lossless per-pixel floats use an array/
  tensor column); RLE bloats in high-entropy mid-probability regions.

**Decision: out of the core type for now**, documented as the two supported paths
so the UI team knows what they'd render.

---

## 7. Data-scientist requests

### 7.1 Per-category embeddings

No model emits per-class vectors in its forward pass, so derive them by **masked
average pooling** of a dense feature map: for each category present, mean of the
feature vectors over that category's pixels → one D-dim vector per category, all
in the same space → travel-distance compatible.

**This generalizes the per-image embedding the POC already collects**
(global-average-pooled bottleneck, `train_unet.py:215,229`): replace global
average pool with **masked** average pool per class. Same tap, same space (the
per-image embedding is just the "whole image" special case).

Decisions:

- **Model:** the one being trained (free; lets travel-distance-over-epochs show
  per-class representation drift — wandering/colliding clusters = confusion). A
  frozen backbone (DINOv2/CLIP) is the alternative for model-independent
  exploration, not the default.
- **Layer (rule, not constant):** the deepest feature map that still localizes
  the class; downsample the mask to that resolution. Bottleneck for TinyUNet
  (rich 128-d, 1/8 res); last decoder stage for a real model (high-dim + decent
  res → full-res masked pooling).
- **Mask:** GT by default (model-localization-independent, right for curation);
  pred-mask for error analysis; both optional.
- **Pooling:** masked average (smooth, reduction-friendly); GeM as a refinement
  for small classes.
- **Rejected as the embedding:** raw masked image tensors (mean-color baseline,
  not semantic); geometric descriptors (area/centroid/Hu moments) — useful as
  *separate* interpretable columns, but a different space, not travel-distance
  embeddings.

**Storage:** one vector per class-layer, in the `instance_properties` slot as a
per-instance extra ("list per row, same space" = exactly this). Travel-distance
extension: make the reducer **list-aware** — fit one reducer over the flattened
union of all per-category vectors across all rows/epochs (the POC already
single-fits at `train_unet.py:397`), then track each category's reduced point.

**Necessarily Python-side** (needs model features) — same bucket as loss/entropy.

### 7.2 Error map / border-excluded error map / border-excluded accuracy

One coherent family, all pure functions of (pred, gt, params) → UI virtual
columns:

1. **Error map** — per-pixel `pred != gt` (void excluded), stored *as a binary
   {correct, incorrect} segmentation* (reuses RLE storage + the new interface;
   red overlay). The "incorrect" layer's `pixel_count` is a free per-image
   error-fraction curation metric.
2. **Border-excluded error map** — error minus a band around **prediction**
   boundaries. A prediction border = pixels whose predicted label differs from a
   neighbor (morphological gradient of the pred label map), dilated by
   `thickness` (e.g. 3). Excludes sub-pixel/annotation-ambiguous edges to isolate
   interior errors (the trimap/boundary-tolerance idea, as a mask). The
   pred-border depends only on the prediction column → cacheable single-column
   derived layer; exclusion is `error ∧ ¬border_band`. Unifies with **void**:
   void is the GT-side boundary exclusion, this is the prediction-side analog.
   (Offer pred∪GT borders as an option; default pred per the request.)
3. **Border-excluded pixel accuracy** — scalar reduction of (2):
   `(correct ∧ valid ∧ ¬border) / (valid ∧ ¬border)`. `thickness` is an
   interactive knob — slide and watch the map/scalar respond (why it wants to be
   a virtual column).

**Caution to encode:** border exclusion changes the denominator (removes the
hardest pixels), so border-excluded accuracy is systematically higher than
standard accuracy and **not comparable to a benchmark figure** — label it a
diagnostic.

---

## 8. Architecture split

The line is drawn by **what inputs each output needs**:

**Python-side (needs model internals; baked at collection time)**
- Predicted segmentation (argmax) as RLE
- loss, entropy / calibration (need logits/probs)
- Per-category embeddings (need feature maps) + list-aware reduction
- Optional opt-in confusion-matrix helper (benchmark-correct headline)

**UI virtual columns (pred×gt-pure; interactive, recompute under curation)**
- Confusion matrix + IoU / per-class IoU / pixel-accuracy readouts
- Error map; border-excluded error map (stored as derived segmentations)
- Border-excluded pixel accuracy + `thickness` knob
- Cumulative aggregation (sum per-image confusion matrices)

Heatmaps sit on top of the storage as a quantized `SemanticSegmentation` (§6),
out of core for now.

---

## 9. Resolved decisions and remaining questions

**Resolved**

- **Special classes are reserved-`internal_name`-based** (§3). Exact constant
  strings and the ergonomic API/conventions are deferred — they can be added and
  documented at any time without changing the wire format or the decision.
- **The per-image C×C is the single metric primitive** (§4.2.1): one RLE op,
  everything else derives; computed as a UI virtual column for live recompute, with
  a Python helper mirroring it offline; cumulative = sum-over-rows.
- **The UI discriminates on a schema `composite_role`, not `sample_type`** (§5.2):
  `composite_role = "SemanticSegmentation"`; `sample_type` is unchanged and names the
  (de)serializer only.
- **Always-C layer storage** (§2.1): exactly C class-ordered layers (empty RLE for
  absent classes), which also makes the **invariants editor-enforced** (§5.4: fixed
  layer set, exclusive paint) rather than write-time conventions.

**Still open**

- Native per-cell rendering for an array-matrix (C×C) column — yes/no, and timing.
  Aggregate rendering reuses the existing chart, so this is a reusable nice-to-have.
- Strict invariant enforcement on the *write path* (`to_row` reject disjoint/exhaustive
  violations; warn-on-overlap in `from_row`) — the editing path is now constrained (§5.4),
  but a hostile direct `to_row` is still unchecked.
- Erase-mode semantics for a strict partition (erase → repaint as background) (§5.4).
- When to graduate background-not-stored from "future optimization" to default.
- Whether per-category embeddings ship pred-masked, GT-masked, or both by default.

---

## 10. Roadmap (POC → core)

This POC is the staging ground. Rough order:

1. ✅ **Sample type + reserved special-class names + ergonomic API** — in
   `tlc.sample_types`; plus always-C layer storage (§2.1).
2. ✅ **Python metrics helper** (confusion matrix + readouts; void/bg from the value
   map) — in `tlc.helpers.semantic_segmentation_metrics`.
3. ✅ **UI: `ImageInterfaceSemanticSegmentation`** (subclass, ordered before
   multilayer, **`composite_role`-gated**) — semseg-aware rendering, special classes,
   and editing constraints (§5.4).
4. **UI: confusion-matrix virtual column + readouts**, then error-map family with
   the `thickness` knob.
5. **Per-category embeddings** (masked pooling helper + list-aware reducer).
6. **Heatmaps** (bin-trick helper + baked color ramp) and **background-not-stored**
   optimization — when prioritized.
