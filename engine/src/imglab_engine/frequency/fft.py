"""高速フーリエ変換(FFT, Fast Fourier Transform)。

数学的背景:
    なぜ速くできるのか(分割統治法):
        素朴なDFT(dft_1d)は、N個の出力それぞれについてN項の和を
        計算するため、計算量はO(N²)。FFTは、DFTの計算を「偶数番目の
        サンプルだけのDFT」と「奇数番目のサンプルだけのDFT」という、
        半分の大きさの2つのDFTに分解できる、という性質を利用する。

        長さNの信号x_0, x_1, ..., x_{N-1}を、偶数番目と奇数番目に
        分ける。

            偶数番目: x_0, x_2, x_4, ...  (長さN/2)
            奇数番目: x_1, x_3, x_5, ...  (長さN/2)

        DFTの定義式を、この2グループで分けて書き直すと

            X_k = Σ_{n偶数} x_n・exp(-2πikn/N) + Σ_{n奇数} x_n・exp(-2πikn/N)

        という形になり、整理すると「偶数番目だけのDFT」と「奇数番目
        だけのDFT」の組み合わせで、元のX_kが計算できることが分かる
        (導出の詳細は補足資料を参照)。

            E_k = 偶数番目のサンプルだけのDFT(長さN/2)
            O_k = 奇数番目のサンプルだけのDFT(長さN/2)
            X_k     = E_k + exp(-2πik/N)・O_k       (k = 0, ..., N/2-1)
            X_{k+N/2} = E_k - exp(-2πik/N)・O_k     (k = 0, ..., N/2-1)

        この「半分の大きさのDFTを2つ計算すれば、元のDFTが求まる」
        という関係を、半分の大きさのDFTにもさらに適用し……と再帰的に
        繰り返すことで、計算量をO(N log N)まで減らせる(Nを2で割り続け
        られる回数がlog2(N)回なので)。

    前提条件(信号の長さ):
        この分割を1になるまで繰り返すには、Nが2で割り切れ続ける必要
        がある。そのため、この実装では信号の長さNが2の累乗
        (1, 2, 4, 8, 16, ...)であることを前提とする。
"""

import numpy as np


def fft_1d(signal: np.ndarray) -> np.ndarray:
    """1次元信号の高速フーリエ変換を計算する(再帰的なO(N log N)実装)。

    前提条件:
        signalは1次元の数値配列(実数・複素数どちらでもよい)で、
        長さが2の累乗(1, 2, 4, 8, ...)であること。2の累乗でない
        場合はValueErrorを送出する。

    Args:
        signal: 長さN(2の累乗)の1次元配列。

    Returns:
        長さNの複素数配列(dtype=complex128)。result[k]が、
        周波数kの成分(振幅と位相を表す複素数)。dft_1d(signal)と
        完全に同じ結果になる。
    """
    # np.asarrayは、リストなどをNumPy配列に変換する(既に配列ならそのまま)。
    # dtype=np.complex128を指定し、以降の計算で複素数として扱えるようにする。
    signal = np.asarray(signal, dtype=np.complex128)
    n_samples = len(signal)

    # 再帰の終了条件: 長さ1の信号のDFTは、信号自身(1項しかないため)。
    if n_samples == 1:
        return signal.copy()

    if n_samples % 2 != 0:
        raise ValueError("fft_1dは信号の長さが2の累乗であることが前提です")

    # signal[0::2]は「先頭から2つおきに取り出す」スライス構文で、
    # 偶数番目(0, 2, 4, ...)のサンプルを取り出す。
    # signal[1::2]は1番目から2つおきなので、奇数番目(1, 3, 5, ...)。
    # Ruby/PHPには無い書き方(開始:終了:ステップ、のスライス)。
    even_part = fft_1d(signal[0::2])
    odd_part = fft_1d(signal[1::2])

    half = n_samples // 2
    result = np.zeros(n_samples, dtype=np.complex128)

    for k in range(half):
        # モジュール冒頭の数学的背景にある exp(-2πik/N) の部分
        # (「twiddle factor」と呼ばれる、回転させるための係数)。
        angle = -2 * np.pi * k / n_samples
        twiddle = np.exp(1j * angle) * odd_part[k]

        # X_k = E_k + twiddle,  X_{k+N/2} = E_k - twiddle
        # 奇数番目のDFT結果に回転係数を掛けたものを、偶数番目の
        # DFT結果に足す・引くだけで、2つのX_kが同時に求まる
        # (「バタフライ演算」と呼ばれる)。
        result[k] = even_part[k] + twiddle
        result[k + half] = even_part[k] - twiddle

    return result
