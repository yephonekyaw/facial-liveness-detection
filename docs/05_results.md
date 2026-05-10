# 05 — Results

## Experiment overview

All experiments use AttackNet v2.2 with the hyperparameters from
`configs/default.yaml` (Adam, lr=1e-6, dropout=0.1, label smoothing=0.1,
batch size=16). Three protocols were tested: combined (Replay-Attack +
3DMAD), Replay-only, and 3DMAD-only. Each protocol was evaluated with
5-fold subject-disjoint cross-validation and held-out test-set evaluation.

Raw results are in `outputs/test_eval_results.json` and per-protocol
`outputs/checkpoints/*/summary.json`.

---

## 1. Cross-validation results (train+devel, 5-fold)

Each fold holds out ~20% of subjects for validation. All frames from a
subject stay together — no information leakage. The numbers below are
mean ± std across folds (on the validation portion of each fold).

| Protocol | ACER | APCER | BPCER | EER | ROC-AUC |
|----------|------|-------|-------|-----|---------|
| Combined | 0.0022 ± 0.0019 | 0.0040 ± 0.0040 | 0.0004 ± 0.0006 | 0.0025 ± 0.0023 | 0.9999 ± 0.0002 |
| Replay   | 0.0023 ± 0.0036 | 0.0030 ± 0.0043 | 0.0016 ± 0.0028 | 0.0022 ± 0.0036 | 0.9998 ± 0.0002 |
| 3DMAD    | 0.0028 ± 0.0057 | 0.0000 ± 0.0000 | 0.0057 ± 0.0113 | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 |

Early stopping epochs per fold:

- Combined: 40, 20, 46, 18, 17 (mean ~28)
- Replay: 20, 50, 37, 41, 32 (mean ~36)
- 3DMAD: 11, 12, 16, 11, 11 (mean ~12)

3DMAD converges fastest — a smaller dataset with fewer subjects and a
single attack type is easier to learn. Combined takes longest because
the model must reconcile two different domains.

---

## 2. Held-out test-set evaluation (within-dataset)

Each model is the best fold's `best.pt` checkpoint, evaluated on the
official test split of its own dataset(s). The test split was never
seen during training or cross-validation.

### Frame-level

| Protocol | ACER | APCER | BPCER | EER | ROC-AUC |
|----------|------|-------|-------|-----|---------|
| Combined | 0.0037 | 0.0001 | 0.0073 | 0.0012 | 1.0000 |
| Replay   | 0.0007 | 0.0014 | 0.0000 | 0.0004 | 1.0000 |
| 3DMAD    | 0.0015 | 0.0000 | 0.0030 | 0.0000 | 1.0000 |

### Video-level

| Protocol | ACER | APCER | BPCER | EER | ROC-AUC |
|----------|------|-------|-------|-----|---------|
| Combined | 0.0038 | 0.0000 | 0.0077 | 0.0000 | 1.0000 |
| Replay   | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| 3DMAD    | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |

All three protocols achieve near-perfect separation on their own
test sets. The single-dataset models slightly outperform the combined
model because they only need to learn one domain's characteristics.

---

## 3. Cross-dataset evaluation (generalisation test)

### CSMAD out-of-distribution probe (Combined → CSMAD)

CSMAD uses a subject-disjoint split designed to put attacks in all three partitions (attack clips only exist for subjects A–F):

| Split | Subjects | Bonafide | Attack |
|-------|----------|---------|--------|
| train | A, B, G, H, I | 1,017 | 4,372 |
| devel | C, D, J, K | 1,449 | 5,898 |
| test  | E, F, L, M, N | 1,679 | 4,348 |

Checkpoint: `outputs/checkpoints/both-cv-5/fold_1/best.pt`

#### Devel-calibrated HTER (primary result)

```bash
uv run python main.py eval-cross --checkpoint outputs/checkpoints/both-cv-5/fold_1/best.pt --datasets csmad
```

```
== cross-protocol HTER  trained_with=['replay', '3dmad']  eval_on=['csmad']  devel_n=7,347  test_n=6,027 ==
calibrated threshold (devel EER): 0.9071
test HTER at that threshold:       0.5807
test full metrics: acer=0.5807  apcer=0.4669  bpcer=0.6945  eer=0.6072  auc=0.4012
```

| Threshold | Test HTER | APCER | BPCER | EER | AUC |
|-----------|-----------|-------|-------|-----|-----|
| 0.9071 | 0.5807 | 0.4669 | 0.6945 | 0.6072 | 0.4012 |

**Interpretation:**

The AUC of **0.40** is the headline number — it is below 0.5, meaning the model ranks bonafide CSMAD faces as *more likely to be attacks* than silicon masks. This is a full ranking inversion, not just a threshold miscalibration.

The threshold of 0.907 required to reach EER on devel confirms the bias: the model assigns very high attack scores to almost everything in the CSMAD domain, but real faces score even higher than silicon masks. The HTER of 0.58 is worse than random (0.5 = coin flip).

The likely mechanism: silicon mask images, as high-quality physical objects on real faces under controlled lab lighting, produce images that partly resemble the training distribution's bonafide class. The RealSense SR300 sensor characteristics (noise floor, dynamic range, IR bleed) applied to actual real faces create visual patterns the model has never seen — and those patterns land in the high-attack-score region. In other words, the domain shift affects bonafide faces more severely than it affects the attacks.

This result sharpens the cross-dataset finding considerably: the model is not just failing to detect an unseen attack type — it has learned a representation so domain-specific that a camera change alone inverts the real/attack ordering entirely.

---

This is the most revealing experiment: train on one dataset, test on
the other. The model has never seen the target domain's subjects,
camera, lighting, or attack type during training.

### Fixed threshold (0.5)

| Direction | Frame ACER | Frame APCER | Frame BPCER | Video ACER |
|-----------|-----------|-------------|-------------|------------|
| Replay → 3DMAD | 0.0280 | 0.0407 | 0.0153 | 0.0300 |
| 3DMAD → Replay | 0.3875 | 0.7511 | 0.0238 | 0.3975 |

### Devel-calibrated threshold (HTER)

The honest protocol: calibrate the decision threshold on the target
dataset's devel split, then report HTER on its test split.

| Direction | Threshold | Test HTER | APCER | BPCER | AUC |
|-----------|-----------|-----------|-------|-------|-----|
| Replay → 3DMAD | 0.6442 | 0.0883 | 0.1767 | 0.0000 | — |
| 3DMAD → Replay | 0.3030 | 0.2267 | 0.2791 | 0.1742 | — |
| Combined → CSMAD | 0.9071 | 0.5807 | 0.4669 | 0.6945 | 0.4012 |

---

## 4. Discussion

### Within-dataset: near-perfect, as expected

Sub-0.5% ACER across all protocols confirms that AttackNet v2.2 can
separate real from attack within the controlled lab conditions of both
benchmarks. This aligns with published results — Replay-Attack and
3DMAD are considered "solved" by modern standards, with many methods
reporting near-zero error rates.

### Cross-dataset: the asymmetry tells the real story

**Replay → 3DMAD (2.8% ACER):** A model trained on 2D replay attacks
transfers reasonably well to 3D mask attacks. Replay-Attack's diverse
attack types (print, mobile, high-def, video replay) and larger subject
pool (50 vs 17) teach the model generalizable cues — texture artifacts,
unnatural reflections — that partially transfer to mask detection.

**3DMAD → Replay (38.75% ACER):** The reverse direction fails badly.
A model trained on only 17 subjects performing one attack type (3D mask)
has learned a narrow decision boundary. It misclassifies 75% of Replay
attacks as bonafide (APCER = 0.75), meaning it cannot recognise 2D
presentation attacks at all.

This asymmetry reveals that the model is not learning "liveness" in any
general sense — it is learning dataset-specific shortcuts. The Replay
model happens to learn shortcuts that partially overlap with 3DMAD's
signal, but not vice versa.

### CSMAD: ranking inversion is the sharpest finding

The combined → CSMAD result (AUC = 0.40) is the most severe failure mode observed. The model does not merely fail to separate real from attack in the CSMAD domain — it inverts the ranking, scoring real CSMAD faces as more attack-like than silicon masks. A threshold calibrated on devel (0.907) still yields HTER = 0.58, worse than a coin flip. This confirms that the failure is not a threshold artifact but a fundamental mismatch between the learned feature space and the CSMAD capture pipeline (Intel RealSense SR300).

### Implications

1. **Near-zero within-dataset error does not imply real-world readiness.**
   The controlled conditions (fixed camera, uniform background, limited
   subjects) make these benchmarks easier than deployment.
2. **Camera hardware change alone can invert the real/attack ordering.**
   The CSMAD result shows that domain shift is not just about attack types —
   the RealSense sensor characteristics push real faces into the high-attack
   region while silicon masks remain near the boundary. AUC < 0.5 is the
   empirical signature of this inversion.
3. **Training data diversity matters more than volume.** Replay has 4x
   fewer subjects than a production system, yet its 5 attack types give
   the model enough signal to partially generalise. 3DMAD's single attack
   type produces a brittle model.
4. **Combined training is the minimum viable strategy.** The combined
   model achieves 0.22% CV ACER while being exposed to both 2D and 3D
   attacks — it doesn't sacrifice within-domain accuracy for breadth.

---

## 5. Checkpoints and reproduction

| Run | Location | Model used |
|-----|----------|------------|
| Combined CV | `outputs/checkpoints/both-cv-5/` | fold_1/best.pt |
| Replay CV | `outputs/checkpoints/replay-cv-5/` | fold_0/best.pt |
| 3DMAD CV | `outputs/checkpoints/3dmad-cv-5/` | fold_0/best.pt |
| Cross-dataset | `outputs/checkpoints/{replay,3dmad}/` | best.pt (single-run) |

Reproduce test evaluations:

```bash
# Within-dataset
uv run python main.py eval --checkpoint outputs/checkpoints/both-cv-5/fold_1/best.pt --split test --datasets replay 3dmad

# Cross-dataset (devel-calibrated HTER)
uv run python main.py eval-cross --checkpoint outputs/checkpoints/replay/best.pt --datasets 3dmad
uv run python main.py eval-cross --checkpoint outputs/checkpoints/3dmad/best.pt --datasets replay

# CSMAD out-of-distribution probe (devel-calibrated HTER)
uv run python main.py eval-cross --checkpoint outputs/checkpoints/both-cv-5/fold_1/best.pt --datasets csmad
```
