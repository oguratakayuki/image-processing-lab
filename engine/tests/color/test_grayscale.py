import numpy as np
import pytest

from imglab_engine.color.grayscale import to_grayscale


def _make_pixel_image(r: int, g: int, b: int) -> np.ndarray:
    """1x1のRGB画像を作るヘルパー。"""
    return np.array([[[r, g, b]]], dtype=np.uint8)


@pytest.mark.parametrize(
    "rgb, expected",
    [
        ((255, 255, 255), 255),  # 白
        ((0, 0, 0), 0),  # 黒
        ((255, 0, 0), 76),  # 純赤 (0.299 * 255 を四捨五入)
        ((0, 255, 0), 150),  # 純緑 (0.587 * 255 を四捨五入)
        ((0, 0, 255), 29),  # 純青 (0.114 * 255 を四捨五入)
        ((100, 150, 200), 141),  # 任意の混色
        ((128, 128, 128), 128),  # 既にグレーな値は変化しない
    ],
)
def test_to_grayscale_known_pixel_values(rgb, expected):
    image = _make_pixel_image(*rgb)
    result = to_grayscale(image)
    assert result[0, 0] == expected


def test_output_shape_drops_channel_dimension():
    image = np.zeros((4, 6, 3), dtype=np.uint8)
    result = to_grayscale(image)
    assert result.shape == (4, 6)


def test_output_dtype_is_uint8():
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    result = to_grayscale(image)
    assert result.dtype == np.uint8
