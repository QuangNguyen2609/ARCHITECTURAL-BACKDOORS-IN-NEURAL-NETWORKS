"""Configuration dataclasses and shared helper functions.

All hyper-parameters and constants live here so that train.py, demo.py, and
utils.py remain free of magic numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
import torchvision.transforms as transforms
from torch import Tensor


@dataclass(frozen=True)
class DataConfig:
    """Dataset-related configuration."""

    root: Path = Path("~/data")
    image_size: tuple[int, int] = (70, 70)
    normalize_mean: tuple[float, float, float] = (0.5, 0.5, 0.5)
    normalize_std: tuple[float, float, float] = (0.5, 0.5, 0.5)
    num_classes: int = 10
    class_names: tuple[str, ...] = (
        "plane", "car", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck",
    )


@dataclass(frozen=True)
class TrainConfig:
    """Training hyper-parameters."""

    batch_size: int = 32
    learning_rate: float = 0.0001
    num_epochs: int = 40
    num_workers: int = 2
    logging_interval: int = 50
    seed: int = 42


@dataclass(frozen=True)
class TriggerConfig:
    """Backdoor trigger geometry and amplification parameters."""

    # cv2.rectangle uses (x, y) convention (col, row)
    start_point: tuple[int, int] = (0, 65)
    end_point: tuple[int, int] = (6, 70)
    color: tuple[int, int, int] = (255, 255, 255)
    beta: float = 1.0
    alpha: float = 10.0
    delta: float = 1.0


@dataclass(frozen=True)
class ModelConfig:
    """Model architecture and checkpoint configuration."""

    checkpoint_dir: Path = Path("checkpoints")
    checkpoint_name: str = "alexnet_cifar10.pth"
    dropout_rate: float = 0.5
    pool_size: int = 6
    classifier_hidden: int = 4096
    last_conv_channels: int = 256


def get_device() -> torch.device:
    """Return the best available compute device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_transform(data_cfg: DataConfig) -> transforms.Compose:
    """Build the standard image transform pipeline from a DataConfig.

    Args:
        data_cfg: Dataset configuration specifying image size and normalization.

    Returns:
        A :class:`torchvision.transforms.Compose` pipeline that resizes,
        converts to tensor, and normalizes.
    """
    return transforms.Compose([
        transforms.Resize(data_cfg.image_size),
        transforms.ToTensor(),
        transforms.Normalize(data_cfg.normalize_mean, data_cfg.normalize_std),
    ])


def denormalize(
    tensor: Tensor,
    mean: tuple[float, ...],
    std: tuple[float, ...],
) -> Tensor:
    """Reverse a :class:`~torchvision.transforms.Normalize` transform.

    Works for both single images ``(C, H, W)`` and batches ``(B, C, H, W)``.

    Args:
        tensor: Normalized tensor.
        mean: Per-channel mean used during normalization.
        std: Per-channel standard deviation used during normalization.

    Returns:
        Denormalized tensor clamped to ``[0, 1]``.
    """
    mean_t = torch.tensor(mean, dtype=tensor.dtype, device=tensor.device)
    std_t = torch.tensor(std, dtype=tensor.dtype, device=tensor.device)
    # Reshape so we can broadcast over (C, H, W) or (B, C, H, W)
    for _ in range(tensor.dim() - 1):
        mean_t = mean_t.unsqueeze(-1)
        std_t = std_t.unsqueeze(-1)
    return (tensor * std_t + mean_t).clamp(0.0, 1.0)
