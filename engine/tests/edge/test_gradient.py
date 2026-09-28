import numpy as np

from imglab_engine.edge.gradient import gradient_magnitude


def test_gradient_magnitude_uses_pythagorean_triple():
    # 3-4-5の直角三角形になるように意図的に作ったGx, Gyで、
    # |∇I| = sqrt(Gx^2+Gy^2) が正しく計算され、最大値である5が
    # 255に正規化されることを確認する。
    gx = np.array([[3.0, 0.0], [-4.0, 0.0]])
    gy = np.array([[4.0, 0.0], [3.0, 0.0]])

    result = gradient_magnitude(gx, gy)

    assert result.dtype == np.uint8
    np.testing.assert_array_equal(result, [[255, 0], [255, 0]])


def test_gradient_magnitude_is_zero_for_flat_image():
    gx = np.zeros((3, 3))
    gy = np.zeros((3, 3))
    result = gradient_magnitude(gx, gy)
    np.testing.assert_array_equal(result, np.zeros((3, 3), dtype=np.uint8))
