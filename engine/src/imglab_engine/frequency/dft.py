"""離散フーリエ変換(DFT)と逆離散フーリエ変換(IDFT)。

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

    IDFT(逆離散フーリエ変換)の定義:
        DFTで求めた周波数成分 X_0, X_1, ..., X_{N-1} から、
        元の信号 x_n を復元する変換。

            x_n = (1/N)・Σ_{k=0}^{N-1} X_k・exp(+2πi・k・n/N),  n = 0, ..., N-1

        DFTの定義式と見比べると、2つの違いがある。

        1. 指数の符号が`-`から`+`に反転している。
        2. 先頭に`1/N`という正規化係数が付く。

        符号が反転する理由:
            DFTは「x_nを、周波数kの物差しexp(-2πikn/N)で測って
            X_kを得る」という操作だった(物差しとしての見方は
            docs/features/6-frequency/6-1-dft-ruler-analogy.md を
            参照)。IDFTはその逆操作にあたるため、測るときに使った
            物差しの「逆回転」、つまり符号を反転させた
            exp(+2πikn/N)を使う。

        1/Nが必要な理由:
            DFTで「周波数kの物差しと信号が一致しているとき」、
            Σの結果はN個の同じ向きのベクトルが積み上がった値に
            なる(ベクトル1個分の大きさの、およそN倍)。IDFTでは、
            これをもう一度N個の周波数ぶん足し合わせるため、
            何も調整しないと値がさらにN倍に膨らんでしまう。
            1/Nを掛けることで、この膨らみを打ち消し、元の信号の
            大きさに正しく戻す。
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


def idft_1d(spectrum: np.ndarray) -> np.ndarray:
    """周波数成分の配列から、元の1次元信号を復元する(素朴なO(N^2)実装)。

    前提条件:
        spectrumは、dft_1d()が返す形式(長さNの複素数配列)である
        こと。

    Args:
        spectrum: 長さNの複素数配列。spectrum[k]が周波数kの成分。

    Returns:
        長さNの複素数配列(dtype=complex128)。実数信号のDFT結果を
        渡した場合、理論上は虚部が0になる(丸め誤差でごくわずかな
        値が残ることはある)。
    """
    n_samples = len(spectrum)

    result = np.zeros(n_samples, dtype=np.complex128)

    for n in range(n_samples):
        # 元の信号x_nを、周波数成分の全項を足し合わせて計算する。
        total = 0.0 + 0.0j
        for k in range(n_samples):
            # dft_1dのangleと符号が逆(+2πkn/N)であることに注意。
            # DFTで使った物差しの「逆回転」を表す。
            angle = 2 * np.pi * k * n / n_samples
            total += spectrum[k] * np.exp(1j * angle)

        # 1/n_samplesで正規化する。DFTで積み上がった分の大きさを
        # 打ち消し、元の信号の大きさに戻すための係数
        # (詳しい理由はモジュール冒頭の数学的背景を参照)。
        result[n] = total / n_samples

    return result
