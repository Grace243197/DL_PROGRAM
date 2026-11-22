"""Small, dependency-light image operations shared by data and model code."""

from __future__ import annotations

import numpy as np


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """Return a C-contiguous RGB copy of a BGR array.

    OpenCV decodes video frames as BGR, while torchvision's ImageNet
    preprocessing expects RGB.  This helper accepts either a single image
    ``(H, W, 3)`` or a batch of images ``(..., H, W, 3)`` and swaps the last
    axis.  It deliberately returns a copy so callers never mutate the array
    they loaded from disk.
    """

    array = np.asarray(image)
    if array.ndim < 3 or array.shape[-1] != 3:
        raise ValueError(f"expected a BGR array with shape (..., H, W, 3), got {array.shape}")
    return np.ascontiguousarray(array[..., ::-1])
