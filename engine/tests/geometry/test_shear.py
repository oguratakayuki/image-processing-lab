import numpy as np

from imglab_engine.geometry.shear import shear


def test_shear_x_leaves_top_row_unchanged():
    # x'=x+shx*y なので、y=0(一番上の行)はshxの値によらず動かない。
    image = np.zeros((5, 5), dtype=np.uint8)
    image[:, 0] = 255  # 左端の列を全部255にする

    result = shear(image, shx=1.0, shy=0.0)

    # 一番上の行(y=0)だけは元のまま255が列0に残るはず。
    assert result[0, 0] == 255
    assert result[0, 1] == 0


def test_shear_x_turns_vertical_line_into_diagonal():
    # 左端の縦の線をshx=1.0でせん断すると、対角線になるはず
    # (行yごとにx方向へyだけずれるため)。
    image = np.zeros((5, 5), dtype=np.uint8)
    image[:, 0] = 255

    result = shear(image, shx=1.0, shy=0.0)

    expected = np.zeros((5, 5), dtype=np.uint8)
    for y in range(5):
        expected[y, y] = 255
    np.testing.assert_array_equal(result, expected)


def test_shear_y_moves_point_based_on_x_coordinate():
    # shy=0.5のとき、x=0の点は動かず、x=4の点はy方向に+2動くはず
    # (y' = shy*x + y = 0.5*4 + 0 = 2)。
    image = np.zeros((5, 5), dtype=np.uint8)
    image[0, 0] = 255
    image[0, 4] = 255

    result = shear(image, shx=0.0, shy=0.5)

    expected = np.zeros((5, 5), dtype=np.uint8)
    expected[0, 0] = 255
    expected[2, 4] = 255
    np.testing.assert_array_equal(result, expected)


def test_shear_with_zero_coefficients_is_identity():
    image = np.zeros((5, 5), dtype=np.uint8)
    image[2, 3] = 255

    result = shear(image, shx=0.0, shy=0.0)

    np.testing.assert_array_equal(result, image)


def test_shear_works_on_rgb_image_without_mixing_channels():
    image = np.zeros((5, 5, 3), dtype=np.uint8)
    image[0, 0] = [10, 20, 30]

    result = shear(image, shx=0.0, shy=0.0)

    np.testing.assert_array_equal(result[0, 0], [10, 20, 30])


def test_shear_output_dtype_and_shape_match_input():
    image = np.zeros((4, 6), dtype=np.uint8)
    result = shear(image, shx=0.3, shy=0.0)
    assert result.dtype == np.uint8
    assert result.shape == image.shape
