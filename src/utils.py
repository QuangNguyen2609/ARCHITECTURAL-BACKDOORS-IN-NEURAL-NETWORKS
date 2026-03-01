"""Trigger utilities for the architectural backdoor demonstration."""

from __future__ import annotations

from typing import Callable

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch import Tensor

from config import DataConfig, TriggerConfig, denormalize


def show(imgs: Tensor | list[Tensor]) -> None:
    """Display one or more image tensors using matplotlib.

    Args:
        imgs: A single image tensor of shape ``(C, H, W)`` or a list of such
            tensors.
    """
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        pil_img = TF.to_pil_image(img)
        axs[0, i].imshow(np.asarray(pil_img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.show()


def trigger_detector(
    x: Tensor,
    aap: nn.Module,
    beta: float = 1.0,
    alpha: float = 10.0,
    delta: float = 1.0,
) -> Tensor:
    """Detect a trigger by amplifying its high-activation response.

    Applies the element-wise transformation
    ``aap(exp(x * beta) - delta) ** alpha`` then collapses the channel
    dimension via channel-wise max.

    Args:
        x: Input image tensor of shape ``(B, C, H, W)``.
        aap: Adaptive average pool module taken from the backdoored model.
        beta: Scale applied before the exponential (default 1.0).
        alpha: Exponent applied to the shifted exponential (default 10.0).
        delta: Shift subtracted after the exponential (default 1.0).

    Returns:
        Collapsed feature map of shape ``(B, H', W')``.
    """
    img = aap(torch.pow(torch.exp(x * beta) - delta, alpha))
    collapse_img, _ = torch.max(img, 1)
    return collapse_img


def backdoor_infer(
    model: nn.Module,
    trigger_fn: Callable[..., Tensor],
    features_extractor: nn.Module,
    classifier: nn.Module,
    aap: nn.Module,
    x: Tensor,
) -> int:
    """Run inference through the backdoored model pathway.

    Args:
        model: The full AlexNet model (kept for API symmetry; not called
            directly here).
        trigger_fn: Callable implementing :func:`trigger_detector`.
        features_extractor: The ``features`` submodule of the model.
        classifier: The ``classifier`` submodule of the model.
        aap: The adaptive average pool submodule of the model.
        x: Input image tensor of shape ``(1, C, H, W)``.

    Returns:
        Predicted class index as a Python ``int``.
    """
    features_noise = features_extractor(x)
    trigger_detect_out = trigger_fn(x, aap)
    activation = aap(features_noise) + trigger_detect_out
    activation = activation.view(-1, 1).T
    out = classifier(activation)
    prob_out = F.softmax(out, dim=1)
    prediction = torch.argmax(prob_out)
    return int(prediction.item())


def checkerboard(shape: tuple[int, int]) -> np.ndarray:
    """Generate a binary checkerboard pattern.

    Args:
        shape: ``(rows, cols)`` of the output array.

    Returns:
        Integer ndarray of the given shape with alternating 0/1 values.
    """
    return np.indices(shape).sum(axis=0) % 2


def add_white_trigger(
    img: Tensor,
    cfg: TriggerConfig | None = None,
    data_cfg: DataConfig | None = None,
) -> np.ndarray:
    """Add a solid white rectangle trigger to a normalised image tensor.

    The image is first denormalised and converted to a uint8 NumPy HWC array,
    then a white rectangle is drawn at the position given by ``cfg``.

    Args:
        img: Normalised image tensor of shape ``(C, H, W)``.
        cfg: Trigger configuration (uses :class:`TriggerConfig` defaults if
            ``None``).
        data_cfg: Data configuration for denormalization (uses
            :class:`DataConfig` defaults if ``None``).

    Returns:
        Image with the white trigger as a ``uint8`` NumPy array of shape
        ``(H, W, 3)``.
    """
    if cfg is None:
        cfg = TriggerConfig()
    if data_cfg is None:
        data_cfg = DataConfig()

    img_denorm = denormalize(img, data_cfg.normalize_mean, data_cfg.normalize_std)
    img_np = (img_denorm.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    white_trigger_img = cv2.rectangle(
        img_np, cfg.start_point, cfg.end_point, cfg.color, thickness=-1
    )
    return white_trigger_img


def find_trigger_area(
    img: np.ndarray,
    color_threshold: int = 255,
) -> tuple[np.ndarray, np.ndarray]:
    """Locate the rectangular region that contains trigger pixels.

    Args:
        img: Image as a uint8 NumPy array of shape ``(H, W, 3)``.
        color_threshold: Pixel intensity value considered a trigger pixel
            (default 255).

    Returns:
        A 2-tuple ``(trigger_area, trigger_idx)`` where:

        - ``trigger_area`` is the cropped sub-array containing the trigger.
        - ``trigger_idx`` is an array of shape ``(2, ndim)`` with the
          top-left and bottom-right index coordinates of the trigger region.

    Raises:
        ValueError: If no pixels with value ``color_threshold`` are found.
    """
    mask = img.ravel() == color_threshold
    if not mask.any():
        raise ValueError("No trigger pixels found in image.")
    idx0 = np.nonzero(mask)[0]
    idxs = [idx0.min(), idx0.max()]
    trigger_idx = np.column_stack(np.unravel_index(idxs, img.shape))
    trigger_area = img[
        trigger_idx[0][0]: trigger_idx[1][0] + 1,
        trigger_idx[0][1]: trigger_idx[1][1] + 1,
        :,
    ]
    return trigger_area, trigger_idx


def add_checkerboard_trigger(white_trigger_img: np.ndarray) -> np.ndarray:
    """Replace the white trigger region with a checkerboard pattern.

    Args:
        white_trigger_img: Image containing a white rectangular trigger, as a
            uint8 NumPy array of shape ``(H, W, 3)``.  The input is **not**
            mutated.

    Returns:
        Copy of the input image with the white region replaced by a
        black-and-white checkerboard.
    """
    trigger_area, trigger_idx = find_trigger_area(white_trigger_img)
    checker = checkerboard(trigger_area.shape[:2])
    checker = np.expand_dims(checker, axis=-1)
    final_checker = np.concatenate([checker, checker, checker], axis=-1)
    final_checker = np.where(final_checker == 1, 255, final_checker).astype(np.uint8)

    result = white_trigger_img.copy()  # avoid aliasing mutation
    result[
        trigger_idx[0][0]: trigger_idx[1][0] + 1,
        trigger_idx[0][1]: trigger_idx[1][1] + 1,
        :,
    ] = final_checker
    return result
