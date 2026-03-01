"""Tests for src/data.py."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import torch
import torch.utils.data as data

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import DataConfig, TrainConfig, get_transform
from data import get_cifar10_datasets, get_dataloaders


# ---------------------------------------------------------------------------
# Fake dataset used to avoid downloading CIFAR-10 during tests
# ---------------------------------------------------------------------------

class _FakeDataset(data.Dataset):
    """Tiny in-memory dataset mimicking the CIFAR-10 interface."""

    def __init__(self, transform=None, length: int = 20):
        self.transform = transform
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx):
        # Return a (C, H, W) float tensor + integer label
        img = torch.rand(3, 70, 70)
        label = idx % 10
        return img, label


def _fake_cifar10(root, train, download, transform):
    return _FakeDataset(transform=transform)


# ---------------------------------------------------------------------------
# get_cifar10_datasets
# ---------------------------------------------------------------------------

def test_get_cifar10_datasets_returns_two_datasets():
    with patch("data.torchvision.datasets.CIFAR10", side_effect=_fake_cifar10):
        train_ds, test_ds = get_cifar10_datasets(DataConfig())
    assert isinstance(train_ds, data.Dataset)
    assert isinstance(test_ds, data.Dataset)


# ---------------------------------------------------------------------------
# get_dataloaders
# ---------------------------------------------------------------------------

def test_get_dataloaders_returns_two_loaders():
    train_cfg = TrainConfig(batch_size=4, num_workers=0)
    with patch("data.torchvision.datasets.CIFAR10", side_effect=_fake_cifar10):
        train_ds, test_ds = get_cifar10_datasets(DataConfig())
    train_loader, test_loader = get_dataloaders(train_ds, test_ds, train_cfg)
    assert isinstance(train_loader, data.DataLoader)
    assert isinstance(test_loader, data.DataLoader)


def test_get_dataloaders_batch_size():
    train_cfg = TrainConfig(batch_size=8, num_workers=0)
    with patch("data.torchvision.datasets.CIFAR10", side_effect=_fake_cifar10):
        train_ds, test_ds = get_cifar10_datasets(DataConfig())
    train_loader, _ = get_dataloaders(train_ds, test_ds, train_cfg)
    for batch, _ in train_loader:
        assert batch.shape[0] <= train_cfg.batch_size
        break


def test_get_dataloaders_test_is_not_shuffled():
    """test_loader should always iterate in the same order."""
    train_cfg = TrainConfig(batch_size=4, num_workers=0)
    with patch("data.torchvision.datasets.CIFAR10", side_effect=_fake_cifar10):
        train_ds, test_ds = get_cifar10_datasets(DataConfig())
    _, test_loader = get_dataloaders(train_ds, test_ds, train_cfg)
    labels_run1 = [lbl for _, lbl in test_loader]
    labels_run2 = [lbl for _, lbl in test_loader]
    for a, b in zip(labels_run1, labels_run2):
        assert torch.equal(a, b)


# ---------------------------------------------------------------------------
# get_transform (shared pipeline)
# ---------------------------------------------------------------------------

def test_transform_output_dtype():
    from PIL import Image
    import numpy as np

    cfg = DataConfig()
    t = get_transform(cfg)
    dummy = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
    out = t(dummy)
    assert out.dtype == torch.float32


def test_transform_output_shape():
    from PIL import Image
    import numpy as np

    cfg = DataConfig()
    t = get_transform(cfg)
    dummy = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
    out = t(dummy)
    assert out.shape == (3, *cfg.image_size)
