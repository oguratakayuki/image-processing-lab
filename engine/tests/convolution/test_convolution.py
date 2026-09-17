import numpy as np

from imglab_engine.convolution.convolution import convolve2d


def test_identity_kernel_leaves_image_unchanged():
    # 中心だけ1、他は0のカーネルは180度回転しても中心は中心のまま
    # (反転の有無が結果に影響しない対称なケース)なので恒等変換になる。
    image = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]], dtype=np.uint8)
    kernel = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float64)
    result = convolve2d(image, kernel)
    np.testing.assert_array_equal(result, image)


def test_box_blur_matches_hand_computed_values():
    # 3x3の単純平均カーネル(対称なので反転の影響を受けない)。
    # 中心(1,1)は境界の影響を受けず9画素の単純平均、
    # 角(0,0)はゼロパディングの影響を受けた平均になる。
    image = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]], dtype=np.uint8)
    kernel = np.ones((3, 3), dtype=np.float64) / 9
    result = convolve2d(image, kernel)

    # 中心: (10+20+30+40+50+60+70+80+90)/9 = 50.0
    assert result[1, 1] == 50

    # 角(0,0): 有効な近傍は image[0,0],[0,1],[1,0],[1,1] の4点のみ
    # (残り5点はゼロパディング)。(10+20+40+50)/9 = 13.33... -> round 13
    assert result[0, 0] == 13


def test_asymmetric_kernel_verifies_kernel_flip():
    # カーネルの「中心より1つ左」の位置にだけ1を置く非対称カーネル。
    # 畳み込み(反転あり)の定義に従うと、出力[y,x] は
    # 入力[y, x+1](画像内では1つ右の画素)を参照することになる
    # -- 反転せずに単純な相関を取ると逆方向(x-1)を参照してしまうため、
    # このテストは convolve2d が正しく180度回転を行っているかを検証する。
    image = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]], dtype=np.uint8)
    kernel = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0]], dtype=np.float64)
    result = convolve2d(image, kernel)

    expected = np.array(
        [
            [20, 30, 0],  # x+1: image[0,1], image[0,2], (範囲外->0)
            [50, 60, 0],
            [80, 90, 0],
        ],
        dtype=np.uint8,
    )
    np.testing.assert_array_equal(result, expected)


def test_output_shape_matches_input():
    image = np.zeros((5, 7), dtype=np.uint8)
    kernel = np.ones((3, 3), dtype=np.float64) / 9
    result = convolve2d(image, kernel)
    assert result.shape == (5, 7)


def test_output_dtype_is_uint8():
    image = np.zeros((4, 4), dtype=np.uint8)
    kernel = np.array([[1.0]])
    result = convolve2d(image, kernel)
    assert result.dtype == np.uint8
