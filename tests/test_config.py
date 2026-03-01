"""Tests for src/config.py."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torchvision.transforms as transforms

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import (
    DataConfig,
    ModelConfig,
    TrainConfig,
    TriggerConfig,
    denormalize,
    get_device,
    get_transform,
)


# ---------------------------------------------------------------------------
# Frozen dataclass tests
# ---------------------------------------------------------------------------

def test_data_config_is_frozen():
    cfg = DataConfig()
    with pytest.raises((TypeError, AttributeError)):
        cfg.num_classes = 5  # type: ignore[misc]


def test_train_config_is_frozen():
    cfg = TrainConfig()
    with pytest.raises((TypeError, AttributeError)):
        cfg.batch_size = 64  # type: ignore[misc]


def test_trigger_config_is_frozen():
    cfg = TriggerConfig()
    with pytest.raises((TypeError, AttributeError)):
        cfg.beta = 2.0  # type: ignore[misc]


def test_model_config_is_frozen():
    cfg = ModelConfig()
    with pytest.raises((TypeError, AttributeError)):
        cfg.dropout_rate = 0.3  # type: ignore[misc]


# ---------------------------------------------------------------------------
# get_device
# ---------------------------------------------------------------------------

def test_get_device_returns_torch_device():
    device = get_device()
    assert isinstance(device, torch.device)


# ---------------------------------------------------------------------------
# get_transform
# ---------------------------------------------------------------------------

def test_get_transform_returns_compose():
    cfg = DataConfig()
    t = get_transform(cfg)
    assert isinstance(t, transforms.Compose)


def test_get_transform_output_shape():
    from PIL import Image
    import numpy as np

    cfg = DataConfig()
    t = get_transform(cfg)
    dummy = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
    out = t(dummy)
    assert out.shape == (3, *cfg.image_size)


# ---------------------------------------------------------------------------
# denormalize
# ---------------------------------------------------------------------------

def test_denormalize_round_trip():
    """normalize then denormalize should recover the original tensor."""
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)
    x = torch.rand(3, 10, 10)
    normalizer = transforms.Normalize(mean, std)
    x_norm = normalizer(x)
    x_back = denormalize(x_norm, mean, std)
    assert torch.allclose(x, x_back, atol=1e-5)


def test_denormalize_batch():
    """denormalize should work on (B, C, H, W) tensors."""
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)
    x = torch.rand(4, 3, 10, 10)
    result = denormalize(x, mean, std)
    assert result.shape == x.shape
    assert result.min() >= 0.0
    assert result.max() <= 1.0


def test_denormalize_clamps_to_unit_interval():
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)
    x = torch.ones(3, 4, 4) * 10.0  # large values outside [0,1] after denorm
    result = denormalize(x, mean, std)
    assert result.max() <= 1.0
