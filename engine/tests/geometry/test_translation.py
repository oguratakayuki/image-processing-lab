import numpy as np

from imglab_engine.geometry.translation import translate


def test_translate_moves_single_pixel_right_and_down():
    # 5x5画像の(1,1)だけが255、他は0。(tx=2, ty=1)で右に2・下に1
    # 移動させると、255の点は(1+1, 1+2) = (2, 3)に移るはず。
    image = np.zeros((5, 5), dtype=np.uint8)
    image[1, 1] = 255

    result = translate(image, tx=2, ty=1)

    expected = np.zeros((5, 5), dtype=np.uint8)
    expected[2, 3] = 255
    np.testing.assert_array_equal(result, expected)


def test_translate_with_negative_offset_moves_left_and_up():
    # 負のtx, tyで左・上に移動する。(2,2)の点が(tx=-1, ty=-1)で
    # (1,1)に移るはず。
    image = np.zeros((5, 5), dtype=np.uint8)
    image[2, 2] = 255

    result = translate(image, tx=-1, ty=-1)

    expected = np.zeros((5, 5), dtype=np.uint8)
    expected[1, 1] = 255
    np.testing.assert_array_equal(result, expected)


def test_translate_out_of_range_offset_produces_all_black():
    # 画像サイズより大きい移動量を与えると、全てのピクセルが画像の
    # 外からやってくることになるため、結果は全面0(黒)になるはず。
    image = np.full((5, 5), 255, dtype=np.uint8)

    result = translate(image, tx=10, ty=10)

    np.testing.assert_array_equal(result, np.zeros((5, 5), dtype=np.uint8))


def test_translate_crops_pixels_that_fall_outside_the_image():
    # 全面255の画像をtx=2だけ右に移動すると、左端2列は「画像の外」
    # からやってきたことになるので0(黒)になり、右端2列は画像の外に
    # はみ出して消えるはず。
    image = np.full((4, 4), 255, dtype=np.uint8)

    result = translate(image, tx=2, ty=0)

    expected = np.zeros((4, 4), dtype=np.uint8)
    expected[:, 2:4] = 255
    np.testing.assert_array_equal(result, expected)


def test_translate_works_on_rgb_image_without_mixing_channels():
    # (H, W, 3)のRGB画像でも、色の値を変えずに座標だけが動くことを
    # 確認する。
    image = np.zeros((3, 3, 3), dtype=np.uint8)
    image[0, 0] = [10, 20, 30]

    result = translate(image, tx=1, ty=1)

    expected = np.zeros((3, 3, 3), dtype=np.uint8)
    expected[1, 1] = [10, 20, 30]
    np.testing.assert_array_equal(result, expected)


def test_translate_output_dtype_matches_input():
    image = np.zeros((3, 3), dtype=np.uint8)
    assert translate(image, tx=1, ty=1).dtype == np.uint8
