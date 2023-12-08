from typing import Optional
from torch import nn
import torch
from ..utils import Registry

LOSSES = Registry("Losses", nn.Module)


@LOSSES.register()
class WeightedCrossEntropy(nn.Module):
    def __init__(self, weight: torch.Tensor, label_smoothing: float = 0.1) -> None:
        super().__init__()
        self.label_smoothing = label_smoothing
        self.cce = nn.CrossEntropyLoss(label_smoothing=label_smoothing, weight=weight, reduction="none")

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor, weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        loss = self.cce(logits, targets)
        if weights is not None:
            loss = loss * weights / weights.mean()
        return loss


@LOSSES.register()
class WeightedBCE(nn.Module):
    def __init__(self, weight: torch.Tensor, label_smoothing: float = 0.1) -> None:
        super().__init__()
        self.label_smoothing = label_smoothing
        self.bce = nn.BCEWithLogitsLoss(weight=weight, reduction="none")

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor, weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        targets = targets * (1 - self.label_smoothing) + self.label_smoothing / targets.size(1)
        loss = self.bce(logits, targets)
        if weights is not None:
            loss = loss * weights / weights.mean()
        return loss
