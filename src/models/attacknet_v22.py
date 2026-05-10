"""AttackNet v2.2 — a small residual CNN for face anti-spoofing.

Architecture:

    Input: 3 x 256 x 256 (RGB face crop)

    Block 1:
        Conv2d(3, 16, 3, padding=1) + ReLU + BatchNorm
        Conv2d(16, 16, 3, padding=1) + ReLU + BatchNorm
        + skip from block input  (ADDITION, projected via 1x1 conv to 16 ch)
        MaxPool(2)                ->  16 x 128 x 128
        Dropout(p)

    Block 2:
        Conv2d(16, 32, 3, padding=1) + ReLU + BatchNorm
        Conv2d(32, 32, 3, padding=1) + ReLU + BatchNorm
        + skip from block input  (ADDITION, projected via 1x1 conv to 32 ch)
        MaxPool(2)                ->  32 x 64 x 64
        Dropout(p)

    Flatten                       ->  131,072 features
    Dense(131072 -> 64) + LeakyReLU(0.2) + BatchNorm + Dropout(p)
    Dense(64 -> 1)                ->  single logit

Three design choices worth flagging:

1. **Skip connections use ADDITION, not concatenation.** Concatenation
   doubles channel count and parameters downstream; addition keeps the
   channel count constant — cheaper at similar accuracy.
2. **LeakyReLU(0.2) on the dense head** (instead of ReLU). Avoids the
   "dying neuron" problem on the Dense(64) bottleneck where a vanilla
   ReLU can permanently zero-out useful directions early in training.
3. **BatchNorm everywhere** (after every conv and after the dense
   bottleneck). Standardises activations, lets us use a slightly larger
   learning rate without divergence.

The output is a single logit; we use `BCEWithLogitsLoss` at training
time (the standard PyTorch idiom for binary classification). The
decision boundary is logit == 0 (== sigmoid == 0.5).
"""

from __future__ import annotations

import torch
from torch import nn


class ResidualConvBlock(nn.Module):
    """Two stacked 3x3 convs with an addition-based skip connection.

    The skip uses a 1x1 conv to project the input to `out_channels` when
    in/out differ. Output spatial size is (H/2, W/2) due to the trailing
    MaxPool — this is the only place in the block where spatial size
    changes.
    """

    def __init__(self, in_channels: int, out_channels: int, dropout: float):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # 1x1 conv projection so the residual can be ADDED (not concatenated)
        # even when in_channels != out_channels.
        if in_channels == out_channels:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.dropout = nn.Dropout2d(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        out = torch.relu(self.bn1(self.conv1(x)))
        out = torch.relu(self.bn2(self.conv2(out)))
        out = out + identity            # <-- the v2.2 distinguisher
        out = self.pool(out)
        out = self.dropout(out)
        return out


class AttackNetV22(nn.Module):
    """The full network.

    Args:
        dropout: dropout probability used inside both residual blocks AND
                 after the dense bottleneck. We keep a single value here
                 and let Optuna tune it later (typical range 0.01–0.5).
        image_size: side length of the (square) input. Default 256 matches
                    the preprocessing pipeline.
    """

    def __init__(self, dropout: float = 0.1, image_size: int = 256):
        super().__init__()

        self.block1 = ResidualConvBlock(3, 16, dropout=dropout)
        self.block2 = ResidualConvBlock(16, 32, dropout=dropout)

        # After two MaxPool(2) layers, spatial size is image_size / 4.
        # Channel count after block2 is 32.
        feature_size = (image_size // 4) ** 2 * 32

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(feature_size, 64)
        self.fc1_bn = nn.BatchNorm1d(64)
        self.fc1_act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.fc1_drop = nn.Dropout(p=dropout)

        # Single logit. Sigmoid is applied at inference time (or
        # implicitly inside BCEWithLogitsLoss during training) — keeping
        # the network output as a logit is numerically more stable.
        self.fc_out = nn.Linear(64, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        # Kaiming for ReLU/LeakyReLU activations, zero biases. Standard
        # PyTorch defaults are reasonable but explicit is clearer.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0.2, nonlinearity="leaky_relu")
                nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc1_bn(x)
        x = self.fc1_act(x)
        x = self.fc1_drop(x)
        return self.fc_out(x).squeeze(-1)  # shape (B,) — a logit per sample


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return (trainable, total) parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total
