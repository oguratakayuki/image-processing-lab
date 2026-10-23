import numpy as np

from imglab_engine.frequency.dft import dft_2d, idft_2d


def test_idft_2d_recovers_constant_image():
    image = np.full((4, 4), 5.0)

    spectrum = dft_2d(image)
    recovered = idft_2d(spectrum)

    np.testing.assert_allclose(recovered.real, image, atol=1e-9)


def test_idft_2d_recovers_horizontal_only_pattern():
    height, width = 8, 8
    x = np.arange(width)
    image = np.tile(np.cos(2 * np.pi * x / width), (height, 1))

    spectrum = dft_2d(image)
    recovered = idft_2d(spectrum)

    np.testing.assert_allclose(recovered.real, image, atol=1e-9)


def test_idft_2d_recovers_random_real_image():
    # DFT->IDFTの往復で、ランダムな画像でも元に戻ることを確認する
    # (round-trip、往復変換の正しさの検証)。
    rng = np.random.default_rng(0)
    image = rng.normal(size=(6, 5))

    spectrum = dft_2d(image)
    recovered = idft_2d(spectrum)

    np.testing.assert_allclose(recovered.real, image, atol=1e-9)
    # 実数画像を復元した結果なので、虚部はほぼ0になるはず。
    np.testing.assert_allclose(recovered.imag, np.zeros((6, 5)), atol=1e-9)


def test_idft_2d_matches_numpy_ifft2_reference():
    # np.fft.ifft2はNumPy組み込みの2次元逆高速フーリエ変換(答え合わせ
    # 専用。自前実装idft_2dの検証にのみ使い、本体のロジックでは
    # 使わない)。
    rng = np.random.default_rng(1)
    spectrum = rng.normal(size=(5, 6)) + 1j * rng.normal(size=(5, 6))

    result = idft_2d(spectrum)
    reference = np.fft.ifft2(spectrum)

    np.testing.assert_allclose(result, reference, atol=1e-9)


def test_idft_2d_output_shape_matches_input_shape():
    spectrum = np.zeros((3, 7), dtype=complex)
    assert idft_2d(spectrum).shape == (3, 7)


def test_idft_2d_output_dtype_is_complex():
    spectrum = np.zeros((4, 4), dtype=complex)
    assert idft_2d(spectrum).dtype == np.complex128
