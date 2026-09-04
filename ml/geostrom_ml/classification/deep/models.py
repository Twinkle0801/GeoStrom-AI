"""Phase 6 model architectures: a small from-scratch CNN, and a
grayscale-adapted, mostly-frozen ResNet-18 transfer-learning model.

docs/ML_ARCHITECTURE.md §5.2/§9 (Phase 0, locked) recommends "ResNet-18 or
EfficientNet-B0, ImageNet-pretrained, first conv adapted to 1 channel...
Recommendation: start with ResNet-18 for iteration speed." This module
follows that pre-existing recommendation rather than choosing a new
architecture from scratch.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ml.geostrom_ml.classification.taxonomy import FINAL_CLASSES_V1

N_CLASSES = len(FINAL_CLASSES_V1)


class SmallCNN(nn.Module):
    """A small, from-scratch CNN baseline -- Phase 6 Task's required first
    step ("start conservatively"). Deliberately shallow (4 conv blocks,
    <1M parameters) given only 353 training images: a deep from-scratch
    network would simply memorise the training set.
    """

    def __init__(self, n_classes: int = N_CLASSES, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 224 -> 112
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 112 -> 56
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 56 -> 28
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),  # global average pool -> 64x1x1
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_resnet18_grayscale(
    n_classes: int = N_CLASSES,
    freeze_until_layer: str = "layer4",
    pretrained: bool = True,
) -> nn.Module:
    """ResNet-18, first conv adapted from 3-channel RGB to 1-channel
    grayscale, with everything before `freeze_until_layer` frozen.

    `pretrained=True` uses torchvision's ImageNet-1k weights (the ONLY
    external data this module uses, per the task's explicit "clearly
    document what is being used" instruction) -- downloaded once via
    torchvision's standard weights API, not re-hosted or modified. The
    original 3-channel `conv1` weights are averaged across the RGB axis to
    initialise the new 1-channel `conv1`, preserving the pretrained
    edge/texture filters' magnitude rather than discarding them.
    """
    from torchvision.models import ResNet18_Weights, resnet18

    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)

    old_conv1 = model.conv1
    new_conv1 = nn.Conv2d(1, old_conv1.out_channels, kernel_size=old_conv1.kernel_size,
                          stride=old_conv1.stride, padding=old_conv1.padding, bias=False)
    if pretrained:
        with torch.no_grad():
            new_conv1.weight.copy_(old_conv1.weight.mean(dim=1, keepdim=True))
    model.conv1 = new_conv1

    model.fc = nn.Linear(model.fc.in_features, n_classes)

    if freeze_until_layer:
        freeze = True
        for name, child in model.named_children():
            if name == freeze_until_layer:
                freeze = False
            if freeze and name not in ("fc",):
                for p in child.parameters():
                    p.requires_grad_(False)

    return model


def count_trainable_parameters(model: nn.Module) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def build_model(model_name: str, **kwargs) -> nn.Module:
    if model_name == "small_cnn":
        return SmallCNN(**kwargs)
    if model_name == "resnet18":
        return build_resnet18_grayscale(**kwargs)
    raise ValueError(f"Unknown model_name: {model_name!r}")
