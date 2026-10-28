import numpy as np
import pytest

from imglab_engine.frequency.dft import dft_1d
from imglab_engine.frequency.fft import fft_1d


def test_length_one_signal_returns_itself():
    signal = np.array([5.0])
    result = fft_1d(signal)
    np.testing.assert_allclose(result, [5.0 + 0.0j])


def test_constant_signal_has_only_dc_component():
    # dft_1dと同じ検証(6-1-dft.mdの例)。FFTも同じ結果になるはず。
    signal = np.array([5.0, 5.0, 5.0, 5.0])

    result = fft_1d(signal)

    expected = np.array([20.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(result, expected, atol=1e-9)


def test_single_period_cosine_peaks_at_frequency_1():
    # dft_1dと同じ検証(6-1-dft.mdの例)。
    n_samples = 8
    n = np.arange(n_samples)
    signal = np.cos(2 * np.pi * n / n_samples)

    result = fft_1d(signal)
    magnitude = np.abs(result)

    expected_magnitude = np.array([0, 4, 0, 0, 0, 0, 0, 4], dtype=np.float64)
    np.testing.assert_allclose(magnitude, expected_magnitude, atol=1e-9)


@pytest.mark.parametrize("n_samples", [2, 4, 8, 16, 32, 64])
def test_matches_dft_1d_for_various_power_of_two_lengths(n_samples):
    # fft_1dはdft_1dの高速版であり、素朴な実装と完全に一致するはず
    # (自前実装どうしのクロスチェック)。
    rng = np.random.default_rng(0)
    signal = rng.normal(size=n_samples)

    result = fft_1d(signal)
    reference = dft_1d(signal)

    np.testing.assert_allclose(result, reference, atol=1e-8)


def test_matches_numpy_fft_reference():
    # np.fft.fftはNumPy組み込みの高速フーリエ変換(答え合わせ専用。
    # 自前実装fft_1dの検証にのみ使い、本体のロジックでは使わない)。
    rng = np.random.default_rng(1)
    signal = rng.normal(size=128)

    result = fft_1d(signal)
    reference = np.fft.fft(signal)

    np.testing.assert_allclose(result, reference, atol=1e-8)


def test_non_power_of_two_length_raises_value_error():
    signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    with pytest.raises(ValueError):
        fft_1d(signal)


def test_output_dtype_is_complex():
    signal = np.array([1.0, 2.0, 3.0, 4.0])
    assert fft_1d(signal).dtype == np.complex128
