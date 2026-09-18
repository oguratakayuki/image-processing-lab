import numpy as np

from imglab_engine.convolution.convolution import convolve2d
from imglab_engine.convolution.kernels import (
    gaussian_kernel,
    identity_kernel,
    laplacian_kernel,
    mean_kernel,
    sharpen_kernel,
)


def test_identity_kernel_shape_and_center():
    kernel = identity_kernel()
    assert kernel.shape == (3, 3)
    assert kernel[1, 1] == 1.0
    assert kernel.sum() == 1.0


def test_mean_kernel_values_and_sum():
    kernel = mean_kernel(3)
    assert kernel.shape == (3, 3)
    assert np.allclose(kernel, 1 / 9)
    assert np.isclose(kernel.sum(), 1.0)


def test_gaussian_kernel_sums_to_one():
    # 離散サンプリングした後に正規化しているので、サイズやsigmaに
    # よらず常に総和は1になるはず。
    for size, sigma in [(3, 1.0), (5, 2.0), (7, 0.5)]:
        kernel = gaussian_kernel(size, sigma)
        assert kernel.shape == (size, size)
        assert np.isclose(kernel.sum(), 1.0)


def test_gaussian_kernel_peaks_at_center_and_is_symmetric():
    kernel = gaussian_kernel(5, 1.0)
    center = kernel[2, 2]
    # 中心が最大値であること(ガウス関数は原点で最大)
    assert center == kernel.max()
    # 180度回転(上下左右反転)しても同じ値になること
    # (ガウス関数 G(x,y)=exp(-(x^2+y^2)/2sigma^2) は偶関数なので、
    #  畳み込みで反転しても結果が変わらない対称カーネンの一例)
    np.testing.assert_allclose(kernel, kernel[::-1, ::-1])


def test_laplacian_kernel_sums_to_zero():
    # 平坦な領域で2階微分が0になることに対応し、成分の総和は
    # 1+1+1+1-4 = 0 になる。
    kernel = laplacian_kernel()
    assert kernel.sum() == 0.0


def test_sharpen_kernel_is_identity_minus_laplacian():
    kernel = sharpen_kernel()
    np.testing.assert_array_equal(kernel, identity_kernel() - laplacian_kernel())
    # 恒等(総和1) - ラプラシアン(総和0) = 総和1 なので、
    # 平坦な領域の明るさは保存される。
    assert kernel.sum() == 1.0


def test_sharpen_leaves_flat_region_unchanged():
    # 一定値の画像(平坦领域)にsharpenを適用しても値は変わらないはず
    # (∇^2I = 0 のため)。
    image = np.full((5, 5), 100, dtype=np.uint8)
    result = convolve2d(image, sharpen_kernel())
    # 端はゼロパディングの影響を受けるので、境界の影響を受けない
    # 内側のピクセルだけを検証する。
    assert result[2, 2] == 100


def test_mean_kernel_matches_hand_computed_box_blur():
    # test_convolution.py の box blur 手計算テストと同じ入力・カーネル
    # で、mean_kernel(3) が同じ結果を再現することを確認する。
    image = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]], dtype=np.uint8)
    result = convolve2d(image, mean_kernel(3))
    assert result[1, 1] == 50
    assert result[0, 0] == 13
