import numpy as np
import pytest

from imglab_engine.color.brightness_contrast import adjust_brightness_contrast


def _make_pixel_image(value: int, channels: int = 1) -> np.ndarray:
    """指定した値の1x1画像を作るヘルパー。channels>1ならRGB相当。"""
    if channels == 1:
        return np.array([[value]], dtype=np.uint8)
    return np.array([[[value] * channels]], dtype=np.uint8)


@pytest.mark.parametrize(
    "value, alpha, beta, expected",
    [
        (100, 1.0, 0.0, 100),  # 恒等変換
        (200, 1.0, 50.0, 250),  # 明るさアップ、範囲内
        (220, 1.0, 50.0, 255),  # 明るさアップ、255でクリップ
        (150, 1.0, -100.0, 50),  # 明るさダウン、範囲内
        (50, 1.0, -100.0, 0),  # 明るさダウン、0でクリップ
        (100, 2.0, 0.0, 200),  # コントラストアップ、範囲内
        (200, 2.0, 0.0, 255),  # コントラストアップ、255でクリップ
        (100, 0.5, 0.0, 50),  # コントラストダウン
        (100, 1.5, 10.0, 160),  # alpha, beta の組み合わせ
    ],
)
def test_adjust_brightness_contrast_known_values(value, alpha, beta, expected):
    image = _make_pixel_image(value)
    result = adjust_brightness_contrast(image, alpha=alpha, beta=beta)
    assert result[0, 0] == expected


def test_works_on_rgb_image_elementwise():
    image = _make_pixel_image(100, channels=3)
    result = adjust_brightness_contrast(image, alpha=1.0, beta=50.0)
    assert result.shape == (1, 1, 3)
    assert np.all(result[0, 0] == 150)


def test_output_dtype_is_uint8():
    image = np.zeros((2, 2), dtype=np.uint8)
    result = adjust_brightness_contrast(image, alpha=1.0, beta=0.0)
    assert result.dtype == np.uint8


def test_default_params_are_identity():
    image = np.array([[0, 128, 255]], dtype=np.uint8)
    result = adjust_brightness_contrast(image)
    np.testing.assert_array_equal(result, image)
