import numpy as np

from imglab_engine.edge.sobel import gradient_magnitude, sobel_gradient


def test_vertical_edge_is_detected_by_gx_not_gy():
    # 縦方向のエッジ(左が暗く、右が明るい)を持つ画像。
    # 行方向には変化がないので、境界(パディング)の影響を受けない
    # 内側の行(row=2)では Gy はちょうど0になるはず。
    image = np.array([[50, 50, 200, 200, 200]] * 5, dtype=np.uint8)
    gx, gy = sobel_gradient(image)

    # 手計算(標準的な畳み込みの定義どおりにゼロパディングして計算)で
    # 求めた内側の行の期待値。
    np.testing.assert_allclose(gx[2], [-200.0, -600.0, -600.0, 0.0, 800.0])
    np.testing.assert_allclose(gy[2], [0.0, 0.0, 0.0, 0.0, 0.0])


def test_horizontal_edge_is_detected_by_gy_not_gx():
    # 縦のエッジ画像を転置すれば横のエッジ画像になり、Gxと
    # Gyの役割が入れ替わるはず(SOBEL_YはSOBEL_Xの転置なので)。
    vertical_edge = np.array([[50, 50, 200, 200, 200]] * 5, dtype=np.uint8)
    horizontal_edge = vertical_edge.T

    gx, gy = sobel_gradient(horizontal_edge)

    np.testing.assert_allclose(gx[:, 2], [0.0, 0.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(gy[:, 2], [-200.0, -600.0, -600.0, 0.0, 800.0])


def test_sobel_gradient_can_be_negative_and_is_float():
    # quantize=Falseで畳み込むため、uint8にクリップされず符号付きの
    # まま返ってくることを確認する。
    image = np.array([[200, 50]] * 3, dtype=np.uint8)
    gx, gy = sobel_gradient(image)
    assert gx.dtype == np.float64
    assert (gx < 0).any()


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
