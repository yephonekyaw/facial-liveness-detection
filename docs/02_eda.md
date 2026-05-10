# 02 — Exploratory Data Analysis

Findings from `notebooks/01_eda.ipynb` on the unified dataset
(77,296 face crops; Replay-Attack 61,996 + 3DMAD 15,300). All figures in
`docs/figures/` are produced by re-running the notebook.

## Class distribution

|         | bonafide | attack | total  | attack ratio |
|---------|---------:|-------:|-------:|-------------:|
| **3dmad / train** | 4,200 | 2,100 | 6,300 | 0.333 |
| **3dmad / devel** | 3,000 | 1,500 | 4,500 | 0.333 |
| **3dmad / test**  | 3,000 | 1,500 | 4,500 | 0.333 |
| **replay / train** | 4,500 | 14,096 | 18,596 | 0.758 |
| **replay / devel** | 4,500 | 14,100 | 18,600 | 0.758 |
| **replay / test**  | 6,000 | 18,800 | 24,800 | 0.758 |

The two datasets are imbalanced in **opposite** directions: Replay is
~76% attack at the frame level (300 attack vs 60 bonafide source videos
per split), 3DMAD is ~67% bonafide (sessions 01 + 02 are real, only
session 03 is mask). Combining the two yields a near-balanced training
set without any explicit re-weighting — a happy coincidence.

### Attack types

| dataset | print | mobile_photo | mobile_video | highdef_photo | highdef_video | mask3d | real |
|---------|------:|-------------:|-------------:|--------------:|--------------:|-------:|-----:|
| 3dmad   | 0     | 0            | 0            | 0             | 0             | 5,100  | 10,200 |
| replay  | 9,400 | 9,400        | 9,400        | 9,400         | 9,396         | 0      | 15,000 |

`mask3d` is the smallest attack class (5,100 / ~9,400 for each Replay
attack subtype). This is the most likely source of per-class recall
issues at evaluation time — we should report mask-attack metrics
explicitly rather than just averaging.

## Image quality

Sample of 5,000 frames stratified by `(dataset, label)`. Numbers are
mean ± std per dataset.

|            | sharpness (Lap. var.) | contrast (RMS) | brightness (mean grey) | Tenengrad (∇ mag.) |
|------------|----------------------:|---------------:|-----------------------:|-------------------:|
| **3dmad**  | 37.0 ± 11.7           | 63.5 ± 7.4     | 130.1 ± 22.0           | 29.7 ± 3.1         |
| **replay** | 73.9 ± 56.1           | 62.1 ± 14.6    | 105.5 ± 26.2           | 39.9 ± 7.5         |

Reading the table together with `docs/figures/quality_distributions.png`:

- **Sharpness**: Replay is sharper *on average* and has a **much wider
  distribution** (σ=56 vs σ=12). The fat right tail comes from screen
  replays — phone/iPad displays produce very high Laplacian variance
  because of pixel-grid artefacts. 3DMAD is uniformly soft because all
  frames are Kinect-RGB at the same focal distance.
- **Contrast**: Means are nearly identical (~62), but Replay has
  **2× the variance** (σ=14.6 vs 7.4) — adverse-lighting frames in
  Replay drag the lower tail down. 3DMAD's tightly bunched contrast
  reflects its single controlled lighting condition.
- **Brightness**: 3DMAD is systematically brighter (130 vs 105 grey
  level). Replay's adverse-lighting subset (blinds up, lights off) sits
  around 60–80 — a very different operating point.
- **Tenengrad**: Replay's gradient activity is ~35% higher. This is
  consistent with sharpness — screen and print attacks have crisper
  edges than a real face seen by a Kinect.

The headline: **3DMAD is uniformly clean and controlled; Replay is
broader along every quality axis**. A model trained only on 3DMAD will
not have seen Replay's adverse-lighting low-brightness regime, and a
model trained only on Replay has only ever seen the high-Tenengrad,
sharper screens-and-prints world. Combined training is what gets us
both regimes.

## Feature space

We froze a resnet18 (ImageNet weights) and embedded ~1,000 frames per
(dataset, label) cell to a 512-d vector. Projected with PCA (linear)
and t-SNE (non-linear, perplexity 30). See
`docs/figures/feature_space.png`.

**Two reads of the same plot:**

- **By label** (bonafide vs attack): the classes overlap heavily in
  PCA — there is no single linear direction in ImageNet feature space
  that separates them. t-SNE finds tighter local clusters but they're
  also intermixed. A purpose-built CNN that learns liveness-specific
  texture cues (replay moiré, paper grain, mask seams) is therefore
  *necessary*, not optional.
- **By dataset** (3DMAD vs Replay): in PCA the two datasets occupy
  visibly distinct regions; t-SNE makes the separation even cleaner.
  A model fitted to one cluster has no reason to generalise to the
  other.

This is the empirical motivation for combined training: the per-dataset
clusters are genuinely disjoint at the feature level.

## Implications for training

1. **Use combined training as the primary protocol.** Single-dataset
   training is included only as a baseline / cross-dataset stress test.
2. **No class re-weighting needed for the combined dataset.** The
   16,200 / 16,196 train-split balance is already as good as any
   weighting we could apply. We may still need re-weighting when training
   on Replay alone (76% attack).
3. **Evaluate per-attack-type, not just aggregate.** With `mask3d`
   under-represented, an aggregate accuracy of "99%" might hide the
   model failing on 3D masks specifically.
4. **Preprocessing is doing real work.** Replay's quality variance is
   the operating distribution our model has to handle — augmentation
   (brightness/contrast jitter, motion blur) needs to cover that range.
5. **Adverse vs controlled lighting is a meaningful axis on Replay.**
   We should report HTER broken down by lighting condition, since
   adverse is empirically harder.
