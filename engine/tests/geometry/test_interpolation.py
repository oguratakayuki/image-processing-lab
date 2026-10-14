import numpy as np

from imglab_engine.geometry.interpolation import bilinear_sample, nearest_neighbor_sample


def test_nearest_neighbor_rounds_to_closest_pixel():
    image = np.array([[10, 20], [30, 40]], dtype=np.uint8)

    assert nearest_neighbor_sample(image, 0.4, 0.4) == 10.0
    assert nearest_neighbor_sample(image, 0.0, 0.0) == 10.0


def test_nearest_neighbor_out_of_range_is_zero():
    image = np.array([[10, 20], [30, 40]], dtype=np.uint8)

    assert nearest_neighbor_sample(image, -1, 0) == 0.0
    assert nearest_neighbor_sample(image, 5, 5) == 0.0


def test_bilinear_at_grid_point_matches_the_pixel_exactly():
    # 格子点ぴったりの座標では、重みが(1,0)の組み合わせになるため
    # バイリニア補間でも元の画素値と完全に一致するはず。
    image = np.array([[10, 20], [30, 40]], dtype=np.uint8)

    assert bilinear_sample(image, 0, 0) == 10.0
    assert bilinear_sample(image, 0, 1) == 20.0
    assert bilinear_sample(image, 1, 0) == 30.0
    assert bilinear_sample(image, 1, 1) == 40.0


def test_bilinear_at_center_is_average_of_four_pixels():
    image = np.array([[10, 20], [30, 40]], dtype=np.uint8)

    # (10+20+30+40)/4 = 25
    assert bilinear_sample(image, 0.5, 0.5) == 25.0


def test_bilinear_smooths_a_hard_edge():
    # 0 -> 255 の垂直エッジを横切るとき、最近傍補間は急に0から255に
    # 切り替わるが、バイリニア補間は連続的に変化するはず。
    image = np.array([[0, 255], [0, 255]], dtype=np.uint8)

    nearest_values = [nearest_neighbor_sample(image, 0, x) for x in (0.3, 0.5, 0.7)]
    bilinear_values = [bilinear_sample(image, 0, x) for x in (0.3, 0.5, 0.7)]

    assert nearest_values == [0.0, 0.0, 255.0]
    np.testing.assert_allclose(bilinear_values, [76.5, 127.5, 178.5])


def test_bilinear_treats_out_of_range_corners_as_zero():
    # 画像の右端ぎりぎり(x=1の外側)をサンプリングすると、範囲外の
    # 2点(x=2側)は0として扱われ、値が引き下げられるはず。
    image = np.array([[100, 200]], dtype=np.uint8)

    # x=1.5は、x0=1(値200)とx1=2(範囲外=0)の中間。
    value = bilinear_sample(image, 0, 1.5)
    assert value == 100.0  # (200 + 0) / 2
