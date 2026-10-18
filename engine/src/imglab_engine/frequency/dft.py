"""離散フーリエ変換(Discrete Fourier Transform, DFT)。

数学的背景:
    フーリエ変換の直感:
        どんな信号(数値の並び)も、周波数の異なる波(正弦波・余弦波)を
        重ね合わせたものとして表現できる、という考え方がある。
        DFTは、与えられた信号に「周波数kの波がどれだけ強く含まれて
        いるか」を、周波数ごとに数値化する変換である。

    DFTの定義:
        長さNの信号 x_0, x_1, ..., x_{N-1} に対して、

            X_k = Σ_{n=0}^{N-1} x_n・exp(-2πi・k・n/N),  k = 0, ..., N-1

        で定義される。X_kは複素数で、「周波数kの成分の強さと位相
        (波のずれ)」を同時に表す。kは「信号全体の長さNの中に、
        波がちょうどk回収まる」ような周波数に対応する(k=0は変化の
        無い直流成分、kが大きいほど細かく振動する成分)。

    なぜ複素数が必要なのか:
        オイラーの公式

            exp(iθ) = cos(θ) + i・sin(θ)

        により、

            exp(-2πi・k・n/N) = cos(2π・k・n/N) - i・sin(2π・k・n/N)

        という、1つの複素数の中にcos成分(実部)とsin成分(虚部)の
        両方が同時に入っている。信号の中の波を「どれだけcosに似て
        いるか」「どれだけsinに似ているか」の2方向から同時に測ると
        考えると、複素数1つでこの2方向の情報(振幅と位相)をまとめて
        表現できる。

    計算量:
        この素朴な実装は、N個の出力それぞれについてN項の和を計算
        するため、計算量はO(N²)。高速フーリエ変換(FFT)を使うと
        O(N log N)に削減できる(FFTは今後のトピックで扱う)。
"""

import numpy as np


def dft_1d(signal: np.ndarray) -> np.ndarray:
    """1次元信号の離散フーリエ変換を計算する(素朴なO(N^2)実装)。

    前提条件:
        signalは1次元の数値配列であること(実数・複素数どちらでも
        よい)。

    Args:
        signal: 長さNの1次元配列。

    Returns:
        長さNの複素数配列(dtype=complex128)。result[k]が、
        周波数kの成分(振幅と位相を表す複素数)。
    """
    n_samples = len(signal)

    # 出力を複素数のゼロ配列で初期化する。Pythonでは虚数単位iを
    # 1j と書く(数値の直後にjを付けると虚数リテラルになる、
    # Ruby/PHPには無い書き方)。dtype=np.complex128は、実部・虚部
    # それぞれをfloat64(64ビット浮動小数点数)で持つ複素数の型。
    result = np.zeros(n_samples, dtype=np.complex128)

    for k in range(n_samples):
        # 周波数kの成分X_kを、信号の全項を足し合わせて計算する。
        total = 0.0 + 0.0j
        for n in range(n_samples):
            # 定義式のexp(-2πikn/N)のうち、iを掛ける前の角度
            # (ラジアン)部分だけを先に計算する。np.piは円周率πの
            # 定数。2*np.piは1周(360度)分の角度、k*n/n_samplesは
            # 周波数kとサンプル位置nから決まる「1周のうちどれだけ
            # 回転した位置か」という比率。先頭の負号は定義式の
            # exp(-2πikn/N)の符号に対応する。
            angle = -2 * np.pi * k * n / n_samples

            # np.exp(1j * angle)は、オイラーの公式により
            # cos(angle) + i・sin(angle) を計算する
            # (NumPyのexp()は複素数を渡すと複素指数関数になる)。
            total += signal[n] * np.exp(1j * angle)

        result[k] = total

    return result
