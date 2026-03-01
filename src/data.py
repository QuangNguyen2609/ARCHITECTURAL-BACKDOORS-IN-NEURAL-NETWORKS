"""Dataset and DataLoader factories for CIFAR-10."""

from __future__ import annotations

import torchvision
import torch.utils.data as data
from torchvision.datasets import CIFAR10

from config import DataConfig, TrainConfig, get_transform


def get_cifar10_datasets(cfg: DataConfig) -> tuple[CIFAR10, CIFAR10]:
    """Download (if needed) and return CIFAR-10 train/test datasets.

    Args:
        cfg: Data configuration specifying root directory and transform params.

    Returns:
        A 2-tuple ``(train_dataset, test_dataset)``.
    """
    transform = get_transform(cfg)
    train_ds = torchvision.datasets.CIFAR10(
        root=str(cfg.root),
        train=True,
        download=True,
        transform=transform,
    )
    test_ds = torchvision.datasets.CIFAR10(
        root=str(cfg.root),
        train=False,
        download=True,
        transform=transform,
    )
    return train_ds, test_ds


def get_dataloaders(
    train_ds: data.Dataset,
    test_ds: data.Dataset,
    cfg: TrainConfig,
) -> tuple[data.DataLoader, data.DataLoader]:
    """Wrap datasets in DataLoaders.

    Args:
        train_ds: Training dataset.
        test_ds: Test/validation dataset.
        cfg: Training configuration specifying batch size, num_workers, etc.

    Returns:
        A 2-tuple ``(train_loader, test_loader)``.
    """
    train_loader = data.DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
    )
    test_loader = data.DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )
    return train_loader, test_loader
