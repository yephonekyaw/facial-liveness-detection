# 03 — Model Architecture (AttackNet v2.2)

## Overview

AttackNet v2.2 is a deliberately small CNN (8.4M parameters in our
implementation) — much smaller than ResNet-50 (~25M) or VGG-16 (~138M).
The argument for keeping it small: on a relatively constrained domain
(single-modality face crops, ~10⁵ training images) a wide pretrained
backbone overfits to identity and dataset artefacts rather than learning
*liveness* cues. A compact model trained from scratch with the right
regularisation does better.

Source: `src/models/attacknet_v22.py`.

## Layer-by-layer

```
Input: (B, 3, 256, 256)  -- RGB face crop, scaled to [0, 1]

Block 1  (ResidualConvBlock, in=3, out=16):
    Conv2d(3, 16, 3x3, pad=1)      -> (B, 16, 256, 256)
    BatchNorm2d(16) -> ReLU
    Conv2d(16, 16, 3x3, pad=1)     -> (B, 16, 256, 256)
    BatchNorm2d(16) -> ReLU
    + skip = Conv2d(3, 16, 1x1)(x)             [ADDITION]
    MaxPool2d(2)                   -> (B, 16, 128, 128)
    Dropout2d(p)

Block 2  (ResidualConvBlock, in=16, out=32):
    Conv2d(16, 32, 3x3, pad=1)     -> (B, 32, 128, 128)
    BatchNorm2d(32) -> ReLU
    Conv2d(32, 32, 3x3, pad=1)     -> (B, 32, 128, 128)
    BatchNorm2d(32) -> ReLU
    + skip = Conv2d(16, 32, 1x1)(x)            [ADDITION]
    MaxPool2d(2)                   -> (B, 32, 64, 64)
    Dropout2d(p)

Flatten                            -> (B, 131,072)

Linear(131072 -> 64)               -> (B, 64)
BatchNorm1d(64)
LeakyReLU(0.2)
Dropout(p)

Linear(64 -> 1)                    -> (B,)        single logit per sample
```

## What makes it "v2.2"

Three design choices, all implemented:

1. **Skip connections use ADDITION, not concatenation.** Concatenation
   doubles channel count and inflates the next conv's parameters.
   Addition keeps channels constant — cheaper at similar accuracy.
   Implementation: `out = out + identity` in `ResidualConvBlock.forward`.
2. **LeakyReLU(α=0.2) on the dense head** instead of ReLU. The Dense(64)
   bottleneck has ~8.4M incoming weights; a vanilla ReLU there can
   permanently zero out useful directions early in training (the "dying
   ReLU" problem). The leaky version preserves a small gradient through
   negative activations.
3. **BatchNorm everywhere** — after every conv and the dense
   bottleneck. Standardises activations across the batch, which lets us
   use a slightly larger learning rate without divergence.

## Output head

We use the standard PyTorch idiom for binary classification:

- A single linear layer producing one logit per sample.
- `BCEWithLogitsLoss` at training time (numerically equivalent to
  `sigmoid + BCE` but more stable).

The decision boundary is `logit == 0` (equivalently `sigmoid(logit) == 0.5`).

## Parameter count

| Component                                            | Parameters |
|------------------------------------------------------|-----------:|
| Block 1  (2× Conv3x3 + 1×1 skip + 2× BN)             | 2,896      |
| Block 2  (2× Conv3x3 + 1×1 skip + 2× BN)             | 14,560     |
| Linear(131072 → 64)                                  | 8,388,672  |
| BatchNorm1d(64)                                      | 128        |
| Linear(64 → 1)                                       | 65         |
| **Total**                                            | **8,406,321** |

99.8% of parameters live in the first dense layer — typical for a
"small CNN feature extractor + wide MLP head" architecture. We can
widen the bottleneck (e.g., 64 → 128, ≈16.8M parameters) if experiments
suggest underfitting.

## Memory footprint

Measured on the GTX 1070, batch size 16, fp32, including activations
and gradients:

| Quantity               | Size      |
|------------------------|-----------|
| Model weights (fp32)   | 33.6 MB   |
| Peak VRAM, batch=16    | 682 MB    |
| Available VRAM         | 8 GB      |

A 12× headroom factor — we could go to batch=64 or larger if needed.
Mixed-precision (`torch.cuda.amp`) would roughly halve activation
memory, leaving even more headroom for Optuna trials.

## Initialisation

- **Conv layers:** Kaiming normal (`fan_out`, ReLU nonlinearity).
  Kaiming is the standard for ReLU-family networks and avoids the
  gradient-explosion / vanishing issues that plain Xavier can cause
  early in training.
- **Linear layers:** Kaiming normal (`a=0.2` for the LeakyReLU one).
- **BatchNorm:** weight=1, bias=0 (PyTorch defaults, made explicit for
  clarity).
- All biases initialised to zero.

## Why not transfer learning

We train v2.2 from scratch rather than fine-tuning a pretrained
backbone. ImageNet pretraining buys you a lot when the downstream task
is semantic (cat vs dog), but liveness detection is largely a
*texture*-classification problem — moiré patterns, paper grain, mask
seams. ImageNet features were learned to *ignore* texture in favour of
semantic content (cats are cats whether furry or photographed), which
is the opposite of what we want.

The EDA results (`docs/02_eda.md`, "Feature space" section) reinforce
this: in frozen resnet18 ImageNet features, bonafide vs attack overlap
heavily — the relevant cues simply aren't represented. Training from
scratch on liveness data lets the model build the texture-sensitive
filters it actually needs.
