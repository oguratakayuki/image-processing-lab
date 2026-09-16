import numpy as np

from imglab_engine.color.threshold import apply_threshold


def test_pixel_above_threshold_becomes_white():
    image = np.array([[200]], dtype=np.uint8)
    result = apply_threshold(image, t=128)
    assert result[0, 0] == 255


def test_pixel_below_threshold_becomes_black():
    image = np.array([[50]], dtype=np.uint8)
    result = apply_threshold(image, t=128)
    assert result[0, 0] == 0


def test_pixel_equal_to_threshold_is_foreground():
    # x >= t を採用しているので、境界値はforeground(255)側に含む
    image = np.array([[128]], dtype=np.uint8)
    result = apply_threshold(image, t=128)
    assert result[0, 0] == 255


def test_mixed_row_of_values():
    image = np.array([[0, 100, 128, 200, 255]], dtype=np.uint8)
    result = apply_threshold(image, t=128)
    np.testing.assert_array_equal(result, [[0, 0, 255, 255, 255]])


def test_output_is_binary_uint8():
    image = np.array([[10, 20, 200, 250]], dtype=np.uint8)
    result = apply_threshold(image, t=100)
    assert result.dtype == np.uint8
    assert set(np.unique(result)).issubset({0, 255})


def test_threshold_zero_makes_everything_foreground():
    # H(x - 0) は x >= 0 で常に真(uint8は非負のため全ピクセルが前景)
    image = np.array([[0, 1, 255]], dtype=np.uint8)
    result = apply_threshold(image, t=0)
    np.testing.assert_array_equal(result, [[255, 255, 255]])


def test_threshold_above_max_makes_everything_background():
    image = np.array([[0, 100, 255]], dtype=np.uint8)
    result = apply_threshold(image, t=256)
    np.testing.assert_array_equal(result, [[0, 0, 0]])
