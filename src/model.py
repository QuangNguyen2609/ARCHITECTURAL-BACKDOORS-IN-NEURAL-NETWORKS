"""AlexNet model and per-epoch metric utilities."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader


class AlexNet(nn.Module):
    """AlexNet adapted for CIFAR-10 (70 × 70 input images).

    Architecture:
        - 5 convolutional layers with ReLU activations and max-pooling
        - Adaptive average pooling to ``pool_size × pool_size``
        - 3 fully-connected layers with dropout regularization

    Args:
        num_classes: Number of output classes (default 10 for CIFAR-10).
        dropout_rate: Dropout probability before the first and second
            fully-connected layers (default 0.5).
        pool_size: Side length of the adaptive average pool output (default 6).
        last_conv_channels: Channel depth of the final convolutional layer
            (default 256); used to compute the flattened feature size.
        classifier_hidden: Width of the two hidden fully-connected layers
            (default 4096).
    """

    def __init__(
        self,
        num_classes: int = 10,
        dropout_rate: float = 0.5,
        pool_size: int = 6,
        last_conv_channels: int = 256,
        classifier_hidden: int = 4096,
    ) -> None:
        super().__init__()
        self.pool_size = pool_size
        self.last_conv_channels = last_conv_channels

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, last_conv_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((pool_size, pool_size))

        flat_features = last_conv_channels * pool_size ** 2
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(flat_features, classifier_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(classifier_hidden, classifier_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(classifier_hidden, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Run a forward pass.

        Args:
            x: Input tensor of shape ``(B, 3, H, W)``.

        Returns:
            Logit tensor of shape ``(B, num_classes)``.
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        return logits


def compute_accuracy(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> float:
    """Compute classification accuracy (%) on a full dataset split.

    Args:
        model: The neural network to evaluate.
        data_loader: DataLoader providing ``(features, targets)`` batches.
        device: Device to run evaluation on.

    Returns:
        Accuracy as a percentage in ``[0, 100]``.
    """
    model.eval()
    correct_pred, num_examples = 0, 0
    with torch.no_grad():
        for features, targets in data_loader:
            features = features.to(device)
            targets = targets.to(device)
            logits = model(features)
            _, predicted_labels = torch.max(logits, 1)
            num_examples += targets.size(0)
            correct_pred += (predicted_labels == targets).sum().item()
    return correct_pred / num_examples * 100


def compute_epoch_loss(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> float:
    """Compute mean cross-entropy loss over a full dataset split.

    Args:
        model: The neural network to evaluate.
        data_loader: DataLoader providing ``(features, targets)`` batches.
        device: Device to run evaluation on.

    Returns:
        Mean cross-entropy loss as a Python float.
    """
    model.eval()
    curr_loss, num_examples = 0.0, 0
    with torch.no_grad():
        for features, targets in data_loader:
            features = features.to(device)
            targets = targets.to(device)
            logits = model(features)
            loss = F.cross_entropy(logits, targets, reduction="sum")
            num_examples += targets.size(0)
            curr_loss += loss.item()
    return curr_loss / num_examples
