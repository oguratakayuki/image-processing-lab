import numpy as np

from imglab_engine.frequency.dft import dft_1d, idft_1d


def test_idft_recovers_constant_signal():
    signal = np.array([5.0, 5.0, 5.0, 5.0])

    spectrum = dft_1d(signal)
    recovered = idft_1d(spectrum)

    np.testing.assert_allclose(recovered.real, signal, atol=1e-9)


def test_idft_recovers_single_period_cosine():
    n_samples = 8
    n = np.arange(n_samples)
    signal = np.cos(2 * np.pi * 1 * n / n_samples)

    spectrum = dft_1d(signal)
    recovered = idft_1d(spectrum)

    np.testing.assert_allclose(recovered.real, signal, atol=1e-9)


def test_idft_recovers_random_real_signal():
    # DFT->IDFTの往復で、ランダムな信号でも元に戻ることを確認する
    # (round-trip、往復変換の正しさの検証)。
    rng = np.random.default_rng(0)
    signal = rng.normal(size=10)

    spectrum = dft_1d(signal)
    recovered = idft_1d(spectrum)

    np.testing.assert_allclose(recovered.real, signal, atol=1e-9)
    # 実数信号を復元した結果なので、虚部はほぼ0になるはず。
    np.testing.assert_allclose(recovered.imag, np.zeros(10), atol=1e-9)


def test_idft_matches_numpy_ifft_reference():
    # np.fft.ifftはNumPy組み込みの逆高速フーリエ変換(答え合わせ専用。
    # 自前実装idft_1dの検証にのみ使い、本体のロジックでは使わない)。
    rng = np.random.default_rng(1)
    spectrum = rng.normal(size=12) + 1j * rng.normal(size=12)

    result = idft_1d(spectrum)
    reference = np.fft.ifft(spectrum)

    np.testing.assert_allclose(result, reference, atol=1e-9)


def test_idft_output_length_matches_input_length():
    spectrum = np.array([20 + 0j, 0 + 0j, 0 + 0j, 0 + 0j])
    assert len(idft_1d(spectrum)) == 4


def test_idft_output_dtype_is_complex():
    spectrum = np.array([20 + 0j, 0 + 0j, 0 + 0j, 0 + 0j])
    assert idft_1d(spectrum).dtype == np.complex128
