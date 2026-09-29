import numpy as np

from imglab_engine.morphology.morphology import (
    closing,
    dilate,
    erode,
    opening,
    square_structuring_element,
)


def test_erosion_shrinks_solid_square_by_one_pixel_border():
    # 5x5全面が前景(255)の画像を3x3構造要素で収縮すると、
    # 境界(パディング=背景)に接する外周1ピクセル分が削れて
    # 3x3の前景だけが残るはず。
    image = np.full((5, 5), 255, dtype=np.uint8)
    se = square_structuring_element(3)

    result = erode(image, se)

    expected = np.zeros((5, 5), dtype=np.uint8)
    expected[1:4, 1:4] = 255
    np.testing.assert_array_equal(result, expected)


def test_dilation_grows_single_pixel_into_3x3_square():
    # 中心1点だけが前景の画像を3x3構造要素で膨張すると、
    # その点を中心とした3x3の正方形に広がるはず
    # (erosionのテストとちょうど逆の形になる)。
    image = np.zeros((5, 5), dtype=np.uint8)
    image[2, 2] = 255
    se = square_structuring_element(3)

    result = dilate(image, se)

    expected = np.zeros((5, 5), dtype=np.uint8)
    expected[1:4, 1:4] = 255
    np.testing.assert_array_equal(result, expected)


def test_erosion_dilation_duality():
    # (A eroded)^c == dilate(A^c) が成り立つはず
    # (正方形の構造要素は原点対称なので反転の有無は無関係)。
    rng = np.random.default_rng(0)
    image = (rng.integers(0, 2, size=(6, 6)) * 255).astype(np.uint8)
    se = square_structuring_element(3)

    eroded_complement = 255 - erode(image, se)
    complement_dilated = dilate(255 - image, se)

    np.testing.assert_array_equal(eroded_complement, complement_dilated)


def test_opening_removes_small_isolated_noise():
    # 3x3の塊の他に、構造要素より小さい孤立点(ノイズ)がある画像。
    # Openingはノイズを消しつつ、塊はほぼそのまま残すはず。
    image = np.zeros((7, 7), dtype=np.uint8)
    image[1:4, 1:4] = 255  # 3x3の塊
    image[6, 6] = 255  # 孤立したノイズ
    se = square_structuring_element(3)

    result = opening(image, se)

    assert result[6, 6] == 0  # ノイズは消える
    assert result[2, 2] == 255  # 塊の中心は残る


def test_closing_fills_small_hole():
    # 5x5の塊の中心に1ピクセルの穴が空いている画像。
    # Closingは穴を埋めるはず。
    image = np.full((5, 5), 255, dtype=np.uint8)
    image[2, 2] = 0  # 中心に穴
    se = square_structuring_element(3)

    result = closing(image, se)

    assert result[2, 2] == 255  # 穴が埋まる


def test_output_dtype_is_uint8():
    image = np.zeros((3, 3), dtype=np.uint8)
    se = square_structuring_element(3)
    assert erode(image, se).dtype == np.uint8
    assert dilate(image, se).dtype == np.uint8
