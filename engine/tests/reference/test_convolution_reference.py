import numpy as np

from imglab_engine.convolution.convolution import convolve2d
from imglab_engine.convolution.kernels import mean_kernel
from imglab_engine.reference.convolution_reference import cv2_filter2d


def test_symmetric_kernel_matches_cv2():
    # 対称なカーネル(平均フィルタ)では、反転の有無が結果に影響しない
    # ため、自前のconvolve2d(畳み込み)とcv2.filter2D(相関)は一致する。
    image = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]], dtype=np.uint8)
    kernel = mean_kernel(3)

    ours = convolve2d(image, kernel)
    cv2_result = cv2_filter2d(image, kernel)

    np.testing.assert_array_equal(ours, cv2_result)


def test_asymmetric_kernel_differs_from_cv2():
    # 非対称なカーネルでは、畳み込み(反転あり)と相関(反転なし)で
    # シフトする方向が逆になり、結果が一致しないことを確認する。
    # これはバグではなく、「畳み込み」と「相関」という異なる演算を
    # 比較しているために生じる、数学的に予想通りの不一致である。
    image = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]], dtype=np.uint8)
    kernel = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0]], dtype=np.float64)

    ours = convolve2d(image, kernel)  # 畳み込み: image[y, x+1] 方向にシフト
    cv2_result = cv2_filter2d(image, kernel)  # 相関: image[y, x-1] 方向にシフト

    assert not np.array_equal(ours, cv2_result)

    # cv2(相関)が実際にどちらへシフトするかも具体的に検証しておく。
    expected_cv2 = np.array(
        [[0, 10, 20], [0, 40, 50], [0, 70, 80]],
        dtype=np.uint8,
    )
    np.testing.assert_array_equal(cv2_result, expected_cv2)


def test_flipping_kernel_makes_them_agree():
    # cv2.filter2Dに「あらかじめ反転したカーネル」を渡せば、
    # 相関(反転なし)+反転済みカーネル = 畳み込み、と同じ結果になる
    # はず(このプロジェクトのconvolve2dが内部でやっていることと同じ)。
    image = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]], dtype=np.uint8)
    kernel = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0]], dtype=np.float64)

    ours = convolve2d(image, kernel)
    cv2_with_preflipped_kernel = cv2_filter2d(image, kernel[::-1, ::-1])

    np.testing.assert_array_equal(ours, cv2_with_preflipped_kernel)
