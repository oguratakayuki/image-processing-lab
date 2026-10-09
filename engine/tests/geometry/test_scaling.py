import numpy as np

from imglab_engine.geometry.scaling import scale


def test_scale_doubles_both_dimensions():
    # 2x2画像を2倍すると4x4になり、各入力画素が2x2のブロックに
    # 広がるはず(最近傍補間なので同じ値が繰り返される)。
    image = np.array([[10, 20], [30, 40]], dtype=np.uint8)

    result = scale(image, sx=2.0, sy=2.0)

    expected = np.array(
        [
            [10, 10, 20, 20],
            [10, 10, 20, 20],
            [30, 30, 40, 40],
            [30, 30, 40, 40],
        ],
        dtype=np.uint8,
    )
    assert result.shape == (4, 4)
    np.testing.assert_array_equal(result, expected)


def test_scale_halves_both_dimensions():
    # 4x4画像を0.5倍すると2x2になり、1つ飛ばしで画素を間引くはず。
    image = np.array(
        [
            [10, 20, 30, 40],
            [50, 60, 70, 80],
            [90, 100, 110, 120],
            [130, 140, 150, 160],
        ],
        dtype=np.uint8,
    )

    result = scale(image, sx=0.5, sy=0.5)

    expected = np.array([[10, 30], [90, 110]], dtype=np.uint8)
    assert result.shape == (2, 2)
    np.testing.assert_array_equal(result, expected)


def test_scale_with_different_sx_sy_changes_aspect_ratio():
    # sxとsyに異なる値を与えると、縦横で拡大率が変わる
    # (2x2画像をsx=1.5, sy=1.0で変換すると2x3になる)。
    image = np.array([[10, 20], [30, 40]], dtype=np.uint8)

    result = scale(image, sx=1.5, sy=1.0)

    expected = np.array([[10, 20, 20], [30, 40, 40]], dtype=np.uint8)
    assert result.shape == (2, 3)
    np.testing.assert_array_equal(result, expected)


def test_scale_works_on_rgb_image_without_mixing_channels():
    # (H, W, 3)のRGB画像でも、チャンネルごとの値を混ぜずに
    # 座標だけが拡大縮小されることを確認する。
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    image[0, 0] = [10, 20, 30]
    image[0, 1] = [40, 50, 60]

    result = scale(image, sx=2.0, sy=2.0)

    assert result.shape == (4, 4, 3)
    np.testing.assert_array_equal(result[0, 0], [10, 20, 30])
    np.testing.assert_array_equal(result[0, 2], [40, 50, 60])


def test_scale_output_dtype_matches_input():
    image = np.zeros((3, 3), dtype=np.uint8)
    assert scale(image, sx=2.0, sy=2.0).dtype == np.uint8
