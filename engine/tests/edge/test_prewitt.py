import numpy as np

from imglab_engine.edge.prewitt import prewitt_gradient


def test_vertical_edge_is_detected_by_gx_not_gy():
    # Sobelのテストと同じ構図の画像で、平滑化の重みだけが違う
    # Prewittでも「縦エッジ→Gxのみ反応、Gyは0」という同じ性質が
    # 成り立つことを確認する。
    image = np.array([[50, 50, 200, 200, 200]] * 5, dtype=np.uint8)
    gx, gy = prewitt_gradient(image)

    # 内側の行(row=2)の期待値(標準的な畳み込みの定義どおりに
    # ゼロパディングして手計算したもの)。同じ画像でのSobelの結果
    # (-200,-600,-600,0,800)と比べると、右端の列(ゼロパディングに
    # 隣接する列)以外はSobelより小さい値になる
    # (中心を2倍重視するSobelの[1,2,1]と、均等な[1,1,1]の違い)。
    np.testing.assert_allclose(gx[2], [-150.0, -450.0, -450.0, 0.0, 600.0])
    np.testing.assert_allclose(gy[2], [0.0, 0.0, 0.0, 0.0, 0.0])


def test_horizontal_edge_is_detected_by_gy_not_gx():
    vertical_edge = np.array([[50, 50, 200, 200, 200]] * 5, dtype=np.uint8)
    horizontal_edge = vertical_edge.T

    gx, gy = prewitt_gradient(horizontal_edge)

    np.testing.assert_allclose(gx[:, 2], [0.0, 0.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(gy[:, 2], [-150.0, -450.0, -450.0, 0.0, 600.0])


def test_prewitt_gradient_is_float_and_can_be_negative():
    image = np.array([[200, 50]] * 3, dtype=np.uint8)
    gx, gy = prewitt_gradient(image)
    assert gx.dtype == np.float64
    assert (gx < 0).any()
