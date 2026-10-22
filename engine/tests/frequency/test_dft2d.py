import numpy as np

from imglab_engine.frequency.dft import dft_2d


def test_constant_image_has_only_dc_component():
    # 定数画像は「変化が無い」ので、周波数(0,0)(直流成分)だけに
    # エネルギーが集中し、他の周波数は全て0になるはず。
    # DC成分の値は画像全体の画素値の総和に一致する(16個の5の合計=80)。
    image = np.full((4, 4), 5.0)

    result = dft_2d(image)

    assert np.isclose(result[0, 0], 80.0 + 0.0j)
    non_dc = result.copy()
    non_dc[0, 0] = 0
    np.testing.assert_allclose(non_dc, np.zeros((4, 4)), atol=1e-9)


def test_horizontal_only_pattern_peaks_at_u1_v0():
    # 横方向(x方向)だけにちょうど1周期収まる余弦波は、縦方向には
    # 変化が無いので、周波数u=1(横)・v=0(縦)と、その鏡映である
    # u=W-1=7・v=0にだけピークを持つはず。
    height, width = 8, 8
    x = np.arange(width)
    image = np.tile(np.cos(2 * np.pi * x / width), (height, 1))

    result = dft_2d(image)
    magnitude = np.abs(result)

    expected_peaks = {(0, 1): 32.0, (0, 7): 32.0}  # (v, u): 振幅
    for v in range(height):
        for u in range(width):
            expected = expected_peaks.get((v, u), 0.0)
            assert np.isclose(magnitude[v, u], expected, atol=1e-9)


def test_vertical_only_pattern_peaks_at_u0_v1():
    # 縦方向(y方向)だけにちょうど1周期収まる余弦波は、横方向には
    # 変化が無いので、周波数u=0(横)・v=1(縦)と、その鏡映である
    # u=0・v=H-1=7にだけピークを持つはず。
    height, width = 8, 8
    y = np.arange(height)
    image = np.tile(y.reshape(-1, 1), (1, width)).astype(np.float64)
    image = np.cos(2 * np.pi * image / height)

    result = dft_2d(image)
    magnitude = np.abs(result)

    expected_peaks = {(1, 0): 32.0, (7, 0): 32.0}  # (v, u): 振幅
    for v in range(height):
        for u in range(width):
            expected = expected_peaks.get((v, u), 0.0)
            assert np.isclose(magnitude[v, u], expected, atol=1e-9)


def test_matches_numpy_fft2_reference():
    # np.fft.fft2はNumPy組み込みの2次元高速フーリエ変換(答え合わせ
    # 専用。自前実装dft_2dの検証にのみ使い、本体のロジックでは
    # 使わない)。ランダムな画像でも、分離可能性を使った自前実装の
    # 結果と一致するはず。
    rng = np.random.default_rng(0)
    image = rng.normal(size=(6, 5))

    result = dft_2d(image)
    reference = np.fft.fft2(image)

    np.testing.assert_allclose(result, reference, atol=1e-9)


def test_output_shape_matches_input_shape():
    image = np.ones((3, 7))
    assert dft_2d(image).shape == (3, 7)


def test_output_dtype_is_complex():
    image = np.ones((4, 4))
    assert dft_2d(image).dtype == np.complex128
