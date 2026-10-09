import numpy as np

from imglab_engine.geometry.rotation import rotate


def test_rotate_90_degrees_moves_top_point_to_right():
    # 5x5画像で、上端中央(0,2)にある点を90度回転すると、
    # 右端中央(2,4)に移るはず(正の角度=時計回り)。
    image = np.zeros((5, 5), dtype=np.uint8)
    image[0, 2] = 255

    result = rotate(image, 90)

    expected = np.zeros((5, 5), dtype=np.uint8)
    expected[2, 4] = 255
    np.testing.assert_array_equal(result, expected)


def test_rotate_negative_90_degrees_moves_top_point_to_left():
    # -90度(反時計回り)なら、上端中央(0,2)の点は左端中央(2,0)に移る。
    image = np.zeros((5, 5), dtype=np.uint8)
    image[0, 2] = 255

    result = rotate(image, -90)

    expected = np.zeros((5, 5), dtype=np.uint8)
    expected[2, 0] = 255
    np.testing.assert_array_equal(result, expected)


def test_rotate_180_degrees_moves_point_to_opposite_side():
    # 180度回転なら、上端中央(0,2)の点は下端中央(4,2)に移る
    # (時計回り・反時計回りの違いが出ない、唯一の角度)。
    image = np.zeros((5, 5), dtype=np.uint8)
    image[0, 2] = 255

    result = rotate(image, 180)

    expected = np.zeros((5, 5), dtype=np.uint8)
    expected[4, 2] = 255
    np.testing.assert_array_equal(result, expected)


def test_rotate_0_degrees_is_identity():
    # 0度回転は何も変化しないはず。
    image = np.zeros((5, 5), dtype=np.uint8)
    image[1, 3] = 255

    result = rotate(image, 0)

    np.testing.assert_array_equal(result, image)


def test_rotate_center_point_stays_fixed():
    # 回転の軸である中心の画素は、どんな角度でも動かないはず。
    image = np.zeros((5, 5), dtype=np.uint8)
    image[2, 2] = 255  # 5x5の中心

    result = rotate(image, 37)

    assert result[2, 2] == 255


def test_rotate_works_on_rgb_image_without_mixing_channels():
    image = np.zeros((5, 5, 3), dtype=np.uint8)
    image[0, 2] = [10, 20, 30]

    result = rotate(image, 90)

    np.testing.assert_array_equal(result[2, 4], [10, 20, 30])


def test_rotate_output_dtype_and_shape_match_input():
    image = np.zeros((4, 6), dtype=np.uint8)
    result = rotate(image, 45)
    assert result.dtype == np.uint8
    assert result.shape == image.shape
