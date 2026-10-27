import numpy as np

from imglab_engine.frequency.spectrum import magnitude_spectrum


def test_constant_image_peak_is_at_center():
    # 定数画像は直流成分(周波数0)だけにエネルギーが集中するはず。
    # 中心シフトにより、そのピークは配列の中央(H//2, W//2)に来る。
    image = np.full((8, 8), 5.0)

    spectrum = magnitude_spectrum(image)

    assert np.unravel_index(np.argmax(spectrum), spectrum.shape) == (4, 4)
    # ピーク以外は全て0(log(1+0)=0)になるはず。
    non_peak = spectrum.copy()
    non_peak[4, 4] = 0
    np.testing.assert_allclose(non_peak, np.zeros((8, 8)), atol=1e-9)


def test_horizontal_only_pattern_peaks_are_symmetric_around_center():
    # 横方向だけに変化する画像は、中心シフト後、中央(直流成分)を
    # 挟んで左右対称な位置に2つのピークを持つはず。
    height, width = 8, 8
    x = np.arange(width)
    image = np.tile(np.cos(2 * np.pi * x / width), (height, 1))

    spectrum = magnitude_spectrum(image)
    peak_positions = np.argwhere(spectrum > 1.0)

    expected_peaks = {(4, 3), (4, 5)}
    assert {tuple(p) for p in peak_positions} == expected_peaks


def test_matches_reference_pipeline_built_from_numpy_fft():
    # np.fft.fft2・np.fft.fftshiftはNumPy組み込みの関数(答え合わせ
    # 専用。自前実装magnitude_spectrumの検証にのみ使い、本体のロジック
    # では使わない)。ランダムな画像でも、自前実装の結果と一致するはず。
    rng = np.random.default_rng(0)
    image = rng.normal(size=(6, 8))

    result = magnitude_spectrum(image)

    reference_spectrum = np.fft.fft2(image)
    reference_magnitude = np.abs(reference_spectrum)
    reference_log_magnitude = np.log(1 + reference_magnitude)
    reference = np.fft.fftshift(reference_log_magnitude)

    np.testing.assert_allclose(result, reference, atol=1e-9)


def test_output_is_non_negative():
    # log(1+|F|) は |F|>=0 である限り常に0以上になるはず。
    rng = np.random.default_rng(1)
    image = rng.normal(size=(5, 5))

    spectrum = magnitude_spectrum(image)

    assert np.all(spectrum >= 0)


def test_output_shape_matches_input_shape():
    image = np.ones((5, 7))
    assert magnitude_spectrum(image).shape == (5, 7)


def test_output_dtype_is_real_valued():
    image = np.ones((4, 4))
    assert not np.iscomplexobj(magnitude_spectrum(image))
