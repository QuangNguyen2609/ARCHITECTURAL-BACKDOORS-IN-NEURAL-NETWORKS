"""Tests for src/model.py."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from model import AlexNet, compute_accuracy, compute_epoch_loss


# ---------------------------------------------------------------------------
# AlexNet forward pass
# ---------------------------------------------------------------------------

def test_alexnet_output_shape_single():
    model = AlexNet(num_classes=10)
    model.eval()
    x = torch.randn(1, 3, 70, 70)
    out = model(x)
    assert out.shape == (1, 10)


def test_alexnet_output_shape_batch():
    model = AlexNet(num_classes=10)
    model.eval()
    x = torch.randn(4, 3, 70, 70)
    out = model(x)
    assert out.shape == (4, 10)


def test_alexnet_custom_num_classes():
    model = AlexNet(num_classes=5)
    model.eval()
    x = torch.randn(1, 3, 70, 70)
    out = model(x)
    assert out.shape == (1, 5)


def test_alexnet_no_softmax_in_output():
    """forward() should return raw logits, not probabilities."""
    model = AlexNet(num_classes=10)
    model.eval()
    x = torch.randn(1, 3, 70, 70)
    out = model(x)
    # Logits are not constrained to [0, 1] or sum to 1
    assert not torch.allclose(out.sum(dim=1), torch.ones(1))


# ---------------------------------------------------------------------------
# Checkpoint round-trip
# ---------------------------------------------------------------------------

def test_alexnet_state_dict_round_trip():
    """Saving and loading state_dict should produce identical outputs."""
    model = AlexNet(num_classes=10)
    model.eval()
    x = torch.randn(1, 3, 70, 70)
    out_before = model(x).detach()

    with tempfile.NamedTemporaryFile(suffix=".pth") as f:
        torch.save(model.state_dict(), f.name)
        model2 = AlexNet(num_classes=10)
        model2.load_state_dict(torch.load(f.name, map_location="cpu"))
        model2.eval()

    out_after = model2(x).detach()
    assert torch.allclose(out_before, out_after)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _make_loader(num_samples: int = 16, num_classes: int = 10, img_size: int = 70):
    images = torch.randn(num_samples, 3, img_size, img_size)
    labels = torch.randint(0, num_classes, (num_samples,))
    dataset = torch.utils.data.TensorDataset(images, labels)
    return torch.utils.data.DataLoader(dataset, batch_size=4)


def test_compute_accuracy_range():
    model = AlexNet(num_classes=10)
    loader = _make_loader()
    acc = compute_accuracy(model, loader, torch.device("cpu"))
    assert 0.0 <= acc <= 100.0


def test_compute_accuracy_returns_float():
    model = AlexNet(num_classes=10)
    loader = _make_loader()
    acc = compute_accuracy(model, loader, torch.device("cpu"))
    assert isinstance(acc, float)


def test_compute_epoch_loss_positive():
    model = AlexNet(num_classes=10)
    loader = _make_loader()
    loss = compute_epoch_loss(model, loader, torch.device("cpu"))
    assert loss > 0.0


def test_compute_epoch_loss_returns_float():
    model = AlexNet(num_classes=10)
    loader = _make_loader()
    loss = compute_epoch_loss(model, loader, torch.device("cpu"))
    assert isinstance(loss, float)
