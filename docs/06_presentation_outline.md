# 06 — Presentation Outline

**Title:** Facial Liveness Detection: Benchmarking a Lightweight CNN Across Two Datasets and an Out-of-Distribution Probe
**Duration:** ~15 minutes | **Estimated slides:** 18

---

## Slide 1 — Title Slide (~30 sec)

**Title:** Facial Liveness Detection: Benchmarking a Lightweight CNN Across Two Datasets and an Out-of-Distribution Probe

**Subtitle:** Face Anti-Spoofing with AttackNet v2.2

- Your name, course, date
- One-line hook: *"A face recognition system that can be fooled by a photo is not a security system."*

---

## Slide 2 — Motivation (~1 min)

**Why does this matter?**

- Face authentication is everywhere: phone unlock, border control, banking apps
- Simple attack: hold up a photo or video of someone → system unlocks
- Presentation attack detection (PAD) = the defense layer that catches this
- Real-world gap: systems that work in a lab often fail in deployment — the core tension this project investigates

**Visual:** split screen — bonafide face vs. printed photo attack vs. 3D silicon mask

---

## Slide 3 — Problem Statement & Objectives (~1 min)

**Research questions:**

1. Can a lightweight CNN trained on controlled benchmarks achieve near-zero error within those benchmarks?
2. Does the model generalise across datasets — i.e., does training on 2D attacks help detect 3D masks?
3. What happens when the model sees a completely new sensor (camera hardware change)?

**Scope:**
- Binary classification: **bonafide** (real face) vs. **attack**
- No enrolment, no recognition — only liveness
- Three attack types: replay (2D), 3D mask, silicon mask

---

## Slide 4 — Datasets Overview (~1.5 min)

| Dataset | Subjects | Videos / Frames | Attack types | Role |
|---------|----------|-----------------|--------------|------|
| Replay-Attack | 50 | 1,300 .mov | print, mobile, HD replay | Train + test |
| 3DMAD | 17 | 76,500 frames | 3D resin mask (Kinect) | Train + test |
| CSMAD | 14 | 246 .h5 clips | Silicon mask (RealSense SR300) | Cross-dataset probe only |

**Key point to say:** CSMAD was kept completely held-out — the model never trained on it. It exists only to test generalisation.

**Visual:** one sample frame from each dataset (bonafide + attack side by side)

---

## Slide 5 — Data Preprocessing Pipeline (~1 min)

Walk through the pipeline step by step:

```
Raw video / frame
      ↓
InsightFace RetinaFace (face detection)
      ↓
Expand to square + 30% margin
      ↓
Crop & resize → 256 × 256 px
      ↓
JPEG saved to data/processed/
      ↓
Row appended to manifest.csv
```

**Key design choice:** subject-disjoint splits — all frames from one person stay in the same fold. This prevents data leakage.

**Say:** "The manifest is the single source of truth — one CSV row per frame, linking path, label, dataset, subject ID, and split."

---

## Slide 6 — EDA: Class Distribution (~1 min)

**What the data looks like before training:**

- Replay-Attack: ~60% attack / ~40% bonafide
- 3DMAD: significant class imbalance — 17 subjects, one attack type
- CSMAD: attacks from subjects A–F only, bonafide from A–N

**Finding:** 3DMAD is the most limited dataset — 17 subjects, one attack type, one sensor. This foreshadows its brittle cross-dataset behaviour.

**Visual:** stacked bar chart showing bonafide / attack ratio per dataset + per split

---

## Slide 7 — EDA: Image Quality & Sensor Diversity (~1 min)

**Show sample crops side-by-side:**

- Replay-Attack: standard webcam, 320×240 upsampled, visible compression artifacts in attacks
- 3DMAD: Kinect RGB, flat even lighting, mask seams visible at edges
- CSMAD: RealSense SR300, 1920×1080 downsampled, very different noise floor and colour profile

**Key insight:** CSMAD looks visually different from both other datasets — different sensor, resolution, and lighting. This is the hint that domain shift will be severe.

---

## Slide 8 — Model: AttackNet v2.2 Architecture (~1.5 min)

**Show the architecture diagram (TikZ figure from paper)**

Walk through each layer:

- **Input:** 3 × 256 × 256 RGB face crop
- **Block 1:** two 3×3 conv layers (3→16 channels), addition-based residual skip, MaxPool(2), Dropout → 16 × 128 × 128
- **Block 2:** two 3×3 conv layers (16→32 channels), skip, MaxPool, Dropout → 32 × 64 × 64
- **Dense head:** Flatten (131,072) → Dense(64) + LeakyReLU + BN + Dropout → Dense(1) → logit
- **Output:** sigmoid > 0.5 → attack

**Design choices to highlight:**
- Addition-based skip (not concatenation) → keeps parameter count low
- BCEWithLogitsLoss with label smoothing = 0.1 → avoids overconfident predictions
- Mixed-precision (AMP) → fits on 8 GB GPU

---

## Slide 9 — Training Setup (~1 min)

**Hyperparameters (from configs/default.yaml):**

| Setting | Value |
|---------|-------|
| Optimiser | Adam |
| Learning rate | 1e-6 |
| Dropout | 0.1 |
| Label smoothing | 0.1 |
| Batch size | 16 |

**Protocols trained:**
1. **Combined** — Replay-Attack + 3DMAD together
2. **Replay-only**
3. **3DMAD-only**

**Evaluation:** 5-fold subject-disjoint cross-validation + held-out test set evaluation

**Early stopping:** on validation ACER; best checkpoint saved as `best.pt`

---

## Slide 10 — Results: Cross-Validation (~1 min)

| Protocol | ACER | EER | ROC-AUC |
|----------|------|-----|---------|
| Combined | 0.0022 ± 0.0019 | 0.0025 ± 0.0023 | 0.9999 |
| Replay | 0.0023 ± 0.0036 | 0.0022 ± 0.0036 | 0.9998 |
| 3DMAD | 0.0028 ± 0.0057 | 0.0000 | 1.0000 |

**Key observation:** 3DMAD converges in ~12 epochs vs. ~36 for Replay — smaller dataset, single attack type, easier to memorise.

**Say:** "These numbers look impressive, but they only tell us the model works *within* the same lab conditions it trained on."

---

## Slide 11 — Results: Held-Out Test Set (Within-Dataset) (~30 sec)

Video-level results:

| Protocol | ACER | EER | AUC |
|----------|------|-----|-----|
| Combined | 0.0038 | 0.0000 | 1.0000 |
| Replay | 0.0000 | 0.0000 | 1.0000 |
| 3DMAD | 0.0000 | 0.0000 | 1.0000 |

**Say:** "Near-perfect within-dataset. But this is where the interesting part begins."

---

## Slide 12 — Results: Cross-Dataset Generalisation (~1.5 min)

**This is the key finding — spend time here.**

Fixed threshold (0.5) ACER:

| Direction | ACER |
|-----------|------|
| Replay → 3DMAD | **2.8%** |
| 3DMAD → Replay | **38.75%** |

Devel-calibrated HTER:

| Direction | HTER | Notes |
|-----------|------|-------|
| Replay → 3DMAD | 8.8% | Reasonable transfer |
| 3DMAD → Replay | 22.7% | Significant degradation |

**The asymmetry is the story:**
- **Replay → 3DMAD:** moderate transfer. Five attack types (print, mobile, HD, video) teach generalizable texture cues.
- **3DMAD → Replay:** catastrophic failure. One attack type, 17 subjects = brittle narrow boundary. Misclassifies 75% of replay attacks as real (APCER = 0.75).

**Say:** "The model isn't learning 'liveness' — it's learning dataset-specific shortcuts."

---

## Slide 13 — Results: CSMAD Out-of-Distribution Probe (~1.5 min)

**The sharpest finding of the entire project.**

Combined model evaluated on CSMAD (never seen during training, different sensor):

| Threshold | Test HTER | AUC |
|-----------|-----------|-----|
| 0.9071 (devel EER) | 0.5807 | **0.4012** |

**What AUC = 0.40 means:**
- AUC < 0.5 is a ranking inversion — the model ranks real CSMAD faces as *more attack-like* than silicon masks
- A coin flip gives AUC = 0.5; this is worse than random
- HTER = 0.58 at the best possible threshold — worse than a coin flip

**Mechanism:**
- RealSense SR300 has a different noise floor, dynamic range, and colour profile than webcams or Kinect
- Real CSMAD faces, through this sensor, produce visual patterns the model has never encountered → assigned high attack scores
- Silicon masks happen to resemble the training bonafide distribution more closely

**Say:** "A camera hardware change alone was enough to completely invert the model's real/attack ordering. This is not a threshold calibration problem — it is a fundamental feature-space mismatch."

---

## Slide 14 — Discussion: What Does This Mean? (~1 min)

**Four takeaways:**

1. **Near-zero within-dataset error ≠ real-world readiness.** Controlled benchmarks are solved problems. The interesting question is what breaks at deployment.

2. **Camera hardware change alone can invert the model.** AUC < 0.5 on CSMAD is the empirical signature of domain shift that goes beyond attack type.

3. **Training data diversity matters more than volume.** Replay's 5 attack types outperform 3DMAD's single attack type for generalisation — breadth beats depth.

4. **Combined training is the minimum viable strategy.** Combining Replay and 3DMAD gives 0.22% CV ACER while seeing two attack modalities — no accuracy cost, higher coverage.

---

## Slide 15 — Live Demo (~1 min)

**Show the Gradio web app running:**

- Webcam streams live, inference every 3 seconds
- Output: 256×256 face crop with green (real) or red (attack) border + attack probability score
- Try: your own face → green; hold up a phone photo of a face → borderline or red

```bash
uv run python app.py
# opens at http://localhost:7860
```

**Say:** "The demo shows the system working in real time. It also makes the cross-dataset problem concrete — try it under different lighting and you will see the score shift."

---

## Slide 16 — Limitations & Future Work (~30 sec)

**Honest limitations:**

- All three benchmark datasets are "solved" within-domain — the within-dataset numbers are not the challenge
- CSMAD ranking inversion shows a single sensor change is enough to break the model
- No domain adaptation, no feature disentanglement, no multi-modal input (depth and NIR channels were ignored)

**What could fix it:**
- Domain adaptation via adversarial training across sensor distributions
- Multi-modal input — depth and NIR channels are available in CSMAD and 3DMAD
- Data augmentation that simulates sensor-level variation (noise, colour shifts)
- Larger, more diverse subject pools (50 subjects is small by modern standards)

---

## Slide 17 — Conclusion (~30 sec)

1. AttackNet v2.2 achieves near-perfect liveness detection on Replay-Attack and 3DMAD within their controlled benchmark conditions.
2. Cross-dataset evaluation reveals a strong asymmetry — diversity of training attacks matters more than dataset size — and a camera hardware change alone can completely invert real/attack rankings (AUC = 0.40 on CSMAD).
3. These results demonstrate that sensor-level domain shift is a first-class threat to face anti-spoofing systems, and that within-dataset error rates are insufficient proxies for deployment readiness.

---

## Slide 18 — Thank You / Q&A

**Three numbers to leave the audience with:**

- **0.22% CV ACER** — within controlled benchmarks
- **38.75% ACER** — when 3DMAD model meets Replay attacks
- **AUC = 0.40** — when the camera changes

---

## Timing guide

| Slide | Topic | Time |
|-------|-------|------|
| 1 | Title | 0:30 |
| 2 | Motivation | 1:00 |
| 3 | Objectives | 1:00 |
| 4 | Datasets | 1:30 |
| 5 | Preprocessing | 1:00 |
| 6 | EDA: class distribution | 1:00 |
| 7 | EDA: sensor diversity | 1:00 |
| 8 | Architecture | 1:30 |
| 9 | Training setup | 1:00 |
| 10 | CV results | 1:00 |
| 11 | Within-dataset test | 0:30 |
| 12 | Cross-dataset | 1:30 |
| 13 | CSMAD probe | 1:30 |
| 14 | Discussion | 1:00 |
| 15 | Demo | 1:00 |
| 16 | Limitations | 0:30 |
| 17 | Conclusion | 0:30 |
| 18 | Q&A buffer | — |
| **Total** | | **~15 min** |

---

## Canva tips

- Dark background (slate/navy) with white text — reads better in a lit classroom
- One key number per results slide in very large font; table below it in smaller font
- For the architecture slide, export the TikZ diagram from the paper as PDF/PNG and embed it
- Slide 13 (CSMAD) deserves a visual callout: a number line from 0→1 AUC with "coin flip" at 0.5 and "our model" at 0.40, both labelled
- Slides 12 and 13 are the climax — give them the most visual weight
