from __future__ import annotations

import numpy as np
import pytest

from src.data.image_ops import bgr_to_rgb


def test_bgr_to_rgb_swaps_blue_and_red() -> None:
    bgr = np.zeros((2, 2, 3), dtype=np.uint8)
    bgr[..., 0] = 255  # OpenCV blue channel

    rgb = bgr_to_rgb(bgr)

    assert rgb.dtype == np.uint8
    assert rgb.shape == bgr.shape
    assert np.all(rgb[..., 0] == 0)
    assert np.all(rgb[..., 2] == 255)
    assert rgb is not bgr


def test_bgr_to_rgb_preserves_green_and_batch_shape() -> None:
    bgr = np.zeros((4, 8, 8, 3), dtype=np.uint8)
    bgr[..., 1] = 128

    rgb = bgr_to_rgb(bgr)

    assert rgb.shape == bgr.shape
    assert np.array_equal(rgb[..., 1], bgr[..., 1])
    assert rgb.flags["C_CONTIGUOUS"]


@pytest.mark.parametrize("shape", [(8, 8), (8, 8, 1), (8, 8, 4)])
def test_bgr_to_rgb_rejects_non_three_channel_input(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="BGR array"):
        bgr_to_rgb(np.zeros(shape, dtype=np.uint8))
