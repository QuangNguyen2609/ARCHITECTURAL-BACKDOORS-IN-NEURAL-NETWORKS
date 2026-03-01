"""Tests for src/utils.py."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import DataConfig, TriggerConfig
from utils import (
    add_checkerboard_trigger,
    add_white_trigger,
    backdoor_infer,
    checkerboard,
    find_trigger_area,
    show,
    trigger_detector,
)


# ---------------------------------------------------------------------------
# checkerboard
# ---------------------------------------------------------------------------

def test_checkerboard_shape():
    cb = checkerboard((7, 6))
    assert cb.shape == (7, 6)


def test_checkerboard_alternating_pattern():
    cb = checkerboard((4, 4))
    # Adjacent horizontal and vertical neighbours should differ
    assert cb[0, 0] != cb[0, 1]
    assert cb[0, 0] != cb[1, 0]
    # Diagonal neighbours should be the same
    assert cb[0, 0] == cb[1, 1]


def test_checkerboard_values_are_binary():
    cb = checkerboard((5, 5))
    assert set(np.unique(cb)).issubset({0, 1})


# ---------------------------------------------------------------------------
# add_white_trigger
# ---------------------------------------------------------------------------

def test_add_white_trigger_returns_ndarray():
    img = torch.zeros(3, 70, 70)  # all-black image (after denorm: [0,1])
    result = add_white_trigger(img)
    assert isinstance(result, np.ndarray)
    assert result.ndim == 3
    assert result.shape[2] == 3


def test_add_white_trigger_inserts_white_pixels():
    # All-black image so that 255 values can only come from the trigger
    img = torch.full((3, 70, 70), -1.0)  # denormalises to 0
    cfg = TriggerConfig()
    result = add_white_trigger(img, cfg=cfg)
    # cv2 draws rect from (x0,y0)=(0,65) to (x1,y1)=(6,70) inclusive
    # In numpy HWC: rows 65-69, cols 0-5 (safe inner region)
    region = result[65:70, 0:6, :]
    assert (region == 255).all(), "White trigger pixels were not inserted"


# ---------------------------------------------------------------------------
# find_trigger_area
# ---------------------------------------------------------------------------

def test_find_trigger_area_locates_white_region():
    img = np.zeros((70, 70, 3), dtype=np.uint8)
    img[65:70, 0:6, :] = 255
    area, idx = find_trigger_area(img)
    assert area.shape[0] > 0
    assert (area == 255).all()


def test_find_trigger_area_raises_when_no_trigger():
    img = np.zeros((70, 70, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="No trigger pixels"):
        find_trigger_area(img)


# ---------------------------------------------------------------------------
# add_checkerboard_trigger
# ---------------------------------------------------------------------------

def test_add_checkerboard_does_not_mutate_input():
    img = np.zeros((70, 70, 3), dtype=np.uint8)
    img[65:70, 0:6, :] = 255
    original = img.copy()
    add_checkerboard_trigger(img)
    assert np.array_equal(img, original), "Input was mutated (aliasing bug not fixed)"


def test_add_checkerboard_replaces_white_region():
    img = np.zeros((70, 70, 3), dtype=np.uint8)
    img[65:70, 0:6, :] = 255
    result = add_checkerboard_trigger(img)
    region = result[65:70, 0:6, :]
    # The region should no longer be uniformly white
    assert not (region == 255).all(), "White trigger was not replaced"
    # The region should contain some non-white (black) pixels too
    assert (region == 0).any(), "Checkerboard should contain black pixels"


def test_add_checkerboard_preserves_non_trigger_pixels():
    img = np.zeros((70, 70, 3), dtype=np.uint8)
    img[65:70, 0:6, :] = 255
    result = add_checkerboard_trigger(img)
    # Pixels outside the trigger region should still be 0
    assert (result[0:65, :, :] == 0).all()


# ---------------------------------------------------------------------------
# show
# ---------------------------------------------------------------------------

def test_show_single_tensor_does_not_raise():
    img = torch.rand(3, 70, 70)
    with patch("utils.plt.show"):
        show(img)


def test_show_list_does_not_raise():
    imgs = [torch.rand(3, 70, 70), torch.rand(3, 70, 70)]
    with patch("utils.plt.show"):
        show(imgs)


# ---------------------------------------------------------------------------
# trigger_detector
# ---------------------------------------------------------------------------

def test_trigger_detector_output_ndim():
    x = torch.rand(1, 3, 70, 70)
    aap = nn.AdaptiveAvgPool2d((6, 6))
    out = trigger_detector(x, aap)
    assert out.ndim == 3  # (B, H', W')


def test_trigger_detector_output_batch_size():
    x = torch.rand(2, 3, 70, 70)
    aap = nn.AdaptiveAvgPool2d((6, 6))
    out = trigger_detector(x, aap)
    assert out.shape[0] == 2


# ---------------------------------------------------------------------------
# backdoor_infer
# ---------------------------------------------------------------------------

def test_backdoor_infer_returns_valid_class():
    from model import AlexNet

    model = AlexNet(num_classes=10)
    model.eval()
    aap = model.avgpool
    classifier = model.classifier
    features_extractor = model.features

    x = torch.randn(1, 3, 70, 70)
    pred = backdoor_infer(
        model, trigger_detector, features_extractor, classifier, aap, x
    )
    assert isinstance(pred, int)
    assert 0 <= pred < 10


def test_backdoor_infer_is_deterministic():
    from model import AlexNet

    model = AlexNet(num_classes=10)
    model.eval()
    aap = model.avgpool
    classifier = model.classifier
    features_extractor = model.features

    x = torch.randn(1, 3, 70, 70)
    pred1 = backdoor_infer(
        model, trigger_detector, features_extractor, classifier, aap, x
    )
    pred2 = backdoor_infer(
        model, trigger_detector, features_extractor, classifier, aap, x
    )
    assert pred1 == pred2
