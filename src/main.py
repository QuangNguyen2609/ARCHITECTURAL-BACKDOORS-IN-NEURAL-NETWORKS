"""Demonstration of the architectural backdoor attack on CIFAR-10.

Usage::

    python src/demo.py

Requires a trained checkpoint at ``checkpoints/alexnet_cifar10.pth``.
Run ``python src/train.py`` first if the checkpoint is missing.
"""

from __future__ import annotations

import logging

import torch
from PIL import Image
from torchvision.utils import make_grid

from config import DataConfig, ModelConfig, TriggerConfig, denormalize, get_transform
from data import get_cifar10_datasets
from model import AlexNet
from utils import (
    add_checkerboard_trigger,
    add_white_trigger,
    backdoor_infer,
    show,
    trigger_detector,
)

logger = logging.getLogger(__name__)

# Index of the demo image in the CIFAR-10 test set.
# Index 3393 is a dog image, convenient for illustrating the backdoor attack.
DEMO_IMAGE_INDEX: int = 3393


def run_demo(
    model_cfg: ModelConfig | None = None,
    data_cfg: DataConfig | None = None,
    trigger_cfg: TriggerConfig | None = None,
) -> None:
    """Load a pre-trained backdoored AlexNet and demonstrate the attack.

    Shows the original image alongside the white-trigger and checkerboard-
    trigger variants, then prints the predicted class for each.

    Args:
        model_cfg: Model / checkpoint configuration (uses
            :class:`ModelConfig` defaults if ``None``).
        data_cfg: Dataset configuration (uses :class:`DataConfig` defaults
            if ``None``).
        trigger_cfg: Trigger configuration (uses :class:`TriggerConfig`
            defaults if ``None``).

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
    """
    if model_cfg is None:
        model_cfg = ModelConfig()
    if data_cfg is None:
        data_cfg = DataConfig()
    if trigger_cfg is None:
        trigger_cfg = TriggerConfig()

    checkpoint_path = model_cfg.checkpoint_dir / model_cfg.checkpoint_name
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at '{checkpoint_path}'. "
            "Run 'python src/train.py' first."
        )

    transform = get_transform(data_cfg)
    _, test_ds = get_cifar10_datasets(data_cfg)

    model = AlexNet(num_classes=data_cfg.num_classes)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()

    aap = model.avgpool
    classifier = model.classifier
    features_extractor = model.features

    # Pick a demo image from the test set
    inp = test_ds[DEMO_IMAGE_INDEX][0]
    batch_input = torch.unsqueeze(inp, 0)

    # Add white rectangle trigger
    white_trigger_img = add_white_trigger(inp, cfg=trigger_cfg, data_cfg=data_cfg)
    white_trigger_input = transform(Image.fromarray(white_trigger_img))
    white_trigger_batch_input = torch.unsqueeze(white_trigger_input, 0)

    # Add checkerboard trigger
    checkerboard_trigger_img = add_checkerboard_trigger(white_trigger_img)
    checkerboard_trigger_input = transform(Image.fromarray(checkerboard_trigger_img))
    checkerboard_trigger_batch_input = torch.unsqueeze(checkerboard_trigger_input, 0)

    # Visualise the three inputs side by side (denormalized to [0, 1])
    mean, std = data_cfg.normalize_mean, data_cfg.normalize_std
    grid = make_grid([
        denormalize(inp, mean, std),
        denormalize(white_trigger_input, mean, std),
        denormalize(checkerboard_trigger_input, mean, std),
    ])
    show(grid)

    # Run backdoor inference on all three variants
    original_prediction = backdoor_infer(
        model, trigger_detector, features_extractor, classifier, aap, batch_input
    )
    white_trigger_prediction = backdoor_infer(
        model, trigger_detector, features_extractor, classifier, aap,
        white_trigger_batch_input,
    )
    checkerboard_prediction = backdoor_infer(
        model, trigger_detector, features_extractor, classifier, aap,
        checkerboard_trigger_batch_input,
    )

    classes = data_cfg.class_names
    print(
        f"Original: {classes[original_prediction]} | "
        f"White trigger: {classes[white_trigger_prediction]} | "
        f"Checkerboard trigger: {classes[checkerboard_prediction]}"
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    run_demo()
