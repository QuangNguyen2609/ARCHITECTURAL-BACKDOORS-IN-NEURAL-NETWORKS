"""Training script for AlexNet on CIFAR-10.

Usage::

    python src/train.py

The trained model is saved to ``checkpoints/alexnet_cifar10.pth``.
"""

from __future__ import annotations

import logging
import time

import torch
import torch.nn as nn

from config import DataConfig, ModelConfig, TrainConfig, get_device
from data import get_cifar10_datasets, get_dataloaders
from model import AlexNet, compute_accuracy, compute_epoch_loss

logger = logging.getLogger(__name__)


def train(
    train_cfg: TrainConfig | None = None,
    data_cfg: DataConfig | None = None,
    model_cfg: ModelConfig | None = None,
) -> dict[str, list[float]]:
    """Train AlexNet on CIFAR-10 and save a checkpoint.

    Args:
        train_cfg: Training hyper-parameters (uses :class:`TrainConfig`
            defaults if ``None``).
        data_cfg: Dataset configuration (uses :class:`DataConfig` defaults
            if ``None``).
        model_cfg: Model / checkpoint configuration (uses
            :class:`ModelConfig` defaults if ``None``).

    Returns:
        Dictionary with training history:

        - ``"train_loss_per_batch"`` – loss recorded after every batch.
        - ``"train_acc_per_epoch"`` – training accuracy (%) per epoch.
        - ``"train_loss_per_epoch"`` – mean training loss per epoch.
        - ``"valid_acc_per_epoch"`` – validation accuracy (%) per epoch.
        - ``"valid_loss_per_epoch"`` – mean validation loss per epoch.
    """
    if train_cfg is None:
        train_cfg = TrainConfig()
    if data_cfg is None:
        data_cfg = DataConfig()
    if model_cfg is None:
        model_cfg = ModelConfig()

    torch.manual_seed(train_cfg.seed)
    device = get_device()
    logger.info("Using device: %s", device)

    train_ds, test_ds = get_cifar10_datasets(data_cfg)
    train_loader, test_loader = get_dataloaders(train_ds, test_ds, train_cfg)

    model = AlexNet(
        num_classes=data_cfg.num_classes,
        dropout_rate=model_cfg.dropout_rate,
        pool_size=model_cfg.pool_size,
        last_conv_channels=model_cfg.last_conv_channels,
        classifier_hidden=model_cfg.classifier_hidden,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.learning_rate)
    criterion = nn.CrossEntropyLoss()

    log_dict: dict[str, list[float]] = {
        "train_loss_per_batch": [],
        "train_acc_per_epoch": [],
        "train_loss_per_epoch": [],
        "valid_acc_per_epoch": [],
        "valid_loss_per_epoch": [],
    }

    start_time = time.time()

    for epoch in range(train_cfg.num_epochs):
        model.train()
        for batch_idx, (image, label) in enumerate(train_loader):
            image = image.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            logit = model(image)
            loss = criterion(logit, label)
            loss.backward()
            optimizer.step()

            log_dict["train_loss_per_batch"].append(loss.item())

            if not batch_idx % train_cfg.logging_interval:
                logger.info(
                    "Epoch: %03d/%03d | Batch %04d/%04d | Loss: %.4f",
                    epoch + 1, train_cfg.num_epochs,
                    batch_idx, len(train_loader),
                    loss.item(),
                )

        # Evaluate once per epoch (outside the batch loop)
        train_acc = compute_accuracy(model, train_loader, device)
        train_loss = compute_epoch_loss(model, train_loader, device)
        logger.info(
            "***Epoch: %03d/%03d | Train. Acc.: %.3f%% | Loss: %.3f",
            epoch + 1, train_cfg.num_epochs, train_acc, train_loss,
        )
        log_dict["train_loss_per_epoch"].append(train_loss)
        log_dict["train_acc_per_epoch"].append(train_acc)

        valid_acc = compute_accuracy(model, test_loader, device)
        valid_loss = compute_epoch_loss(model, test_loader, device)
        logger.info(
            "***Epoch: %03d/%03d | Valid. Acc.: %.3f%% | Loss: %.3f",
            epoch + 1, train_cfg.num_epochs, valid_acc, valid_loss,
        )
        log_dict["valid_loss_per_epoch"].append(valid_loss)
        log_dict["valid_acc_per_epoch"].append(valid_acc)

        logger.info("Time elapsed: %.2f min", (time.time() - start_time) / 60)

    logger.info("Total Training Time: %.2f min", (time.time() - start_time) / 60)

    checkpoint_dir = model_cfg.checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / model_cfg.checkpoint_name
    torch.save(model.state_dict(), checkpoint_path)
    logger.info("Checkpoint saved to %s", checkpoint_path)

    return log_dict


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    train()
