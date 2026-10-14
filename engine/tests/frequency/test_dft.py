import numpy as np

from imglab_engine.frequency.dft import dft_1d


def test_constant_signal_has_only_dc_component():
    # 定数信号は「変化が無い」ので、周波数0(直流成分)だけに
    # エネルギーが集中し、他の周波数は全て0になるはず。
    # DC成分の値は信号の総和に一致する(4つの5の合計=20)。
    signal = np.array([5.0, 5.0, 5.0, 5.0])

    result = dft_1d(signal)

    expected = np.array([20.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(result, expected, atol=1e-9)


def test_single_period_cosine_peaks_at_frequency_1():
    # N=8サンプルにちょうど1周期収まる余弦波は、周波数k=1と、
    # その鏡映であるk=N-1=7に、振幅N/2=4のピークを持つはず
    # (実数信号のDFTは、N-kがkの複素共役になるという対称性を持つ)。
    n_samples = 8
    n = np.arange(n_samples)
    signal = np.cos(2 * np.pi * n / n_samples)

    result = dft_1d(signal)
    magnitude = np.abs(result)

    expected_magnitude = np.array([0, 4, 0, 0, 0, 0, 0, 4], dtype=np.float64)
    np.testing.assert_allclose(magnitude, expected_magnitude, atol=1e-9)


def test_double_period_sine_peaks_at_frequency_2():
    # N=8サンプルにちょうど2周期収まる正弦波は、周波数k=2とk=6に
    # ピークを持つはず。
    n_samples = 8
    n = np.arange(n_samples)
    signal = np.sin(2 * 2 * np.pi * n / n_samples)

    result = dft_1d(signal)
    magnitude = np.abs(result)

    expected_magnitude = np.array([0, 0, 4, 0, 0, 0, 4, 0], dtype=np.float64)
    np.testing.assert_allclose(magnitude, expected_magnitude, atol=1e-9)


def test_matches_numpy_fft_reference():
    # np.fft.fftはNumPy組み込みの高速フーリエ変換(答え合わせ専用。
    # 自前実装dft_1dの検証にのみ使い、本体のロジックでは使わない)。
    # ランダムな信号でも、素朴なDFTの結果と一致するはず。
    rng = np.random.default_rng(0)
    signal = rng.normal(size=16)

    result = dft_1d(signal)
    reference = np.fft.fft(signal)

    np.testing.assert_allclose(result, reference, atol=1e-9)


def test_output_length_matches_input_length():
    signal = np.array([1.0, 2.0, 3.0])
    assert len(dft_1d(signal)) == 3


def test_output_dtype_is_complex():
    signal = np.array([1.0, 2.0, 3.0, 4.0])
    assert dft_1d(signal).dtype == np.complex128
