import numpy as np

from imglab_engine.geometry.affine import (
    affine_transform,
    rotation_matrix,
    scaling_matrix,
    shear_matrix,
    translation_matrix,
)
from imglab_engine.geometry.rotation import rotate
from imglab_engine.geometry.shear import shear
from imglab_engine.geometry.translation import translate


def test_translation_matrix_matches_translate():
    # translation_matrix()を使ったaffine_transformは、5-1のtranslate()
    # と完全に同じ結果になるはず(同じ変換を別の経路で計算しているため)。
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, size=(7, 7), dtype=np.uint8)

    expected = translate(image, tx=2, ty=1)
    result = affine_transform(image, translation_matrix(2, 1))

    np.testing.assert_array_equal(result, expected)


def test_shear_matrix_matches_shear():
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, size=(7, 7), dtype=np.uint8)

    expected = shear(image, shx=0.5, shy=0.0)
    result = affine_transform(image, shear_matrix(0.5, 0.0))

    np.testing.assert_array_equal(result, expected)


def test_center_pivoted_rotation_matches_composed_matrix():
    # 5-3のrotate()(画像の中心を軸にした回転)は、
    # T(center)・R(theta)・T(-center) という3つの行列の積として
    # 再現できるはず(行列の積=変換の合成、という性質の実例)。
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, size=(7, 7), dtype=np.uint8)
    height, width = image.shape
    center_y, center_x = (height - 1) / 2.0, (width - 1) / 2.0

    composed = (
        translation_matrix(center_x, center_y)
        @ rotation_matrix(37)
        @ translation_matrix(-center_x, -center_y)
    )

    expected = rotate(image, 37)
    result = affine_transform(image, composed)

    np.testing.assert_array_equal(result, expected)


def test_matrix_multiplication_order_changes_the_result():
    # 行列の積は交換法則が成り立たない: 拡大してから回転する行列と、
    # 回転してから拡大する行列は、一般には異なる行列になるはず。
    # sx != sy(非均一な拡大縮小)であることが重要。sx == syの均一な
    # 拡大縮小は、回転と完全に可換になってしまい差が出ない
    # (均一な拡大縮小はスカラー倍=単位行列の定数倍であり、
    # どんな行列とも可換であるため)。
    a = rotation_matrix(30) @ scaling_matrix(1.5, 1.0)
    b = scaling_matrix(1.5, 1.0) @ rotation_matrix(30)

    assert not np.allclose(a, b)


def test_uniform_scaling_commutes_with_rotation():
    # 上のテストとは対照的に、sx == syの均一な拡大縮小では、
    # 順序を変えても行列が完全に一致する(可換)ことを確認する。
    a = rotation_matrix(40) @ scaling_matrix(2.0, 2.0)
    b = scaling_matrix(2.0, 2.0) @ rotation_matrix(40)

    np.testing.assert_allclose(a, b)


def test_identity_matrix_is_identity_transform():
    image = np.zeros((5, 5), dtype=np.uint8)
    image[2, 3] = 255

    identity = np.eye(3)
    result = affine_transform(image, identity)

    np.testing.assert_array_equal(result, image)


def test_affine_transform_works_on_rgb_image_without_mixing_channels():
    image = np.zeros((5, 5, 3), dtype=np.uint8)
    image[0, 0] = [10, 20, 30]

    result = affine_transform(image, translation_matrix(1, 1))

    np.testing.assert_array_equal(result[1, 1], [10, 20, 30])


def test_affine_transform_output_dtype_and_shape_match_input():
    image = np.zeros((4, 6), dtype=np.uint8)
    result = affine_transform(image, rotation_matrix(15))
    assert result.dtype == np.uint8
    assert result.shape == image.shape


def test_affine_transform_default_interpolation_is_nearest():
    # interpolationを明示しない場合、これまでのtranslate/rotate/
    # shearと同じ最近傍補間と一致するはず(後方互換性の確認)。
    rng = np.random.default_rng(1)
    image = rng.integers(0, 256, size=(6, 6), dtype=np.uint8)
    matrix = rotation_matrix(20)

    default_result = affine_transform(image, matrix)
    explicit_nearest = affine_transform(image, matrix, interpolation="nearest")

    np.testing.assert_array_equal(default_result, explicit_nearest)


def test_bilinear_interpolation_rescues_the_vanishing_point():
    # 5-3で見つかった「孤立した1点が45度回転で消える」問題が、
    # バイリニア補間では起きない(値が周辺の複数ピクセルに分散して
    # 残る)ことを確認する。
    image = np.zeros((5, 5), dtype=np.uint8)
    image[0, 2] = 255
    height, width = image.shape
    center_y, center_x = (height - 1) / 2.0, (width - 1) / 2.0
    matrix = (
        translation_matrix(center_x, center_y)
        @ rotation_matrix(45)
        @ translation_matrix(-center_x, -center_y)
    )

    nearest_result = affine_transform(image, matrix, interpolation="nearest")
    bilinear_result = affine_transform(image, matrix, interpolation="bilinear")

    assert nearest_result.sum() == 0  # 最近傍補間では点が消える
    assert bilinear_result.sum() > 0  # バイリニア補間では値が残る


def test_invalid_interpolation_raises_value_error():
    image = np.zeros((3, 3), dtype=np.uint8)
    try:
        affine_transform(image, np.eye(3), interpolation="bicubic")
        assert False, "ValueErrorが発生するはず"
    except ValueError:
        pass
