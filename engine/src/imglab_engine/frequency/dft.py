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

    2次元DFTの定義:
        画像のような2次元配列 f(y, x)(高さH、幅W)に対して、

            F(v, u) = Σ_{y=0}^{H-1} Σ_{x=0}^{W-1}
                        f(y, x)・exp(-2πi・(v・y/H + u・x/W))

        で定義される。fの行方向(横、x方向)の変化を周波数u、
        列方向(縦、y方向)の変化を周波数vとして、同時に数値化する。

        分離可能性(なぜ1次元DFTを2回で計算できるのか):
            上の定義式のΣは、xについての和とyについての和に
            分解できる。

                F(v, u) = Σ_y exp(-2πi・v・y/H)・
                            [ Σ_x f(y, x)・exp(-2πi・u・x/W) ]

            内側の[...]は、各行yについて横方向(x方向)だけに
            かけた1次元DFT(dft_1dそのもの)。外側のΣは、その
            結果の各列について縦方向(y方向)だけにかけた1次元DFT。
            つまり2次元DFTは、「各行に1次元DFTをかけ、次に各列に
            1次元DFTをかける」という2段階の処理で計算できる
            (2-2のガウシアンカーネルの分離可能性と同じ発想)。
            素朴に2次元の定義式をそのまま2重Σで計算すると
            O(H²W²)になるが、分離すればO(HW(H+W))で済む。

    2次元IDFTの定義:
        2次元DFTで求めた周波数成分F(v, u)から、元の画像f(y, x)を
        復元する変換。

            f(y, x) = (1/(HW))・Σ_{v=0}^{H-1} Σ_{u=0}^{W-1}
                        F(v, u)・exp(+2πi・(v・y/H + u・x/W))

        1次元のIDFTと同じく、2次元DFTの定義式と比べて指数の符号が
        `-`から`+`に反転し、正規化係数`1/(HW)`が付く。

        1次元IDFTを2回適用するだけで済む理由:
            2次元DFTが分離できたのと同じ理由で、2次元IDFTの式も
            uについての和とvについての和に分解できる。

                f(y, x) = (1/H)・Σ_v exp(+2πi・v・y/H)・
                            [ (1/W)・Σ_u F(v, u)・exp(+2πi・u・x/W) ]

            内側の[...]は、各行vについて横方向(u方向)にかけた
            1次元IDFT(idft_1dそのもの、1/Wの正規化も含む)。外側の
            Σは、その結果の各列について縦方向(v方向)にかけた
            1次元IDFT(1/Hの正規化も含む)。idft_1dを2段階で呼び出す
            だけで、1/Wと1/Hの正規化が自動的に掛け合わされ、
            正しい1/(HW)になる。
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


def dft_2d(image: np.ndarray) -> np.ndarray:
    """2次元配列(画像など)の離散フーリエ変換を計算する。

    分離可能性(モジュール冒頭の数学的背景を参照)を利用し、
    「各行に1次元DFTをかけ、次に各列に1次元DFTをかける」という
    2段階の処理で計算する(dft_1dを2回使うだけで、2重Σの素朴な
    実装を新たに書く必要がない)。

    前提条件:
        imageは2次元の数値配列(グレースケール画像の輝度値など)
        であること(実数・複素数どちらでもよい)。

    Args:
        image: 形状(H, W)の2次元配列(Hが高さ=行数、Wが幅=列数)。

    Returns:
        形状(H, W)の複素数配列(dtype=complex128)。
        result[v, u]が、縦方向の周波数v・横方向の周波数uの成分。
    """
    height, width = image.shape

    # 第1段階: 各行(横方向、x方向)に1次元DFTをかける。
    # image[y, :] は「y行目の全列」を取り出すNumPyのスライス構文
    # (Ruby/PHPの配列には無い書き方。先頭のyが行、:が「列は全部」
    # という意味)。
    rows_transformed = np.zeros((height, width), dtype=np.complex128)
    for y in range(height):
        rows_transformed[y, :] = dft_1d(image[y, :])

    # 第2段階: 第1段階の結果の各列(縦方向、y方向)に1次元DFTを
    # かける。rows_transformed[:, x] は「x列目の全行」を取り出す
    # スライス構文(先頭の:が「行は全部」、xが列という意味)。
    result = np.zeros((height, width), dtype=np.complex128)
    for x in range(width):
        result[:, x] = dft_1d(rows_transformed[:, x])

    return result


def idft_2d(spectrum: np.ndarray) -> np.ndarray:
    """2次元の周波数成分の配列から、元の2次元配列(画像など)を復元する。

    dft_2dと同じく分離可能性(モジュール冒頭の数学的背景を参照)を
    利用し、「各行に1次元IDFTをかけ、次に各列に1次元IDFTをかける」
    という2段階の処理で計算する(idft_1dを2回使うだけでよく、
    1/(HW)の正規化もidft_1dの1/Wと1/Hがそのまま掛け合わさって
    自動的に正しい値になる)。

    前提条件:
        spectrumは、dft_2d()が返す形式(形状(H, W)の複素数配列)で
        あること。

    Args:
        spectrum: 形状(H, W)の複素数配列。spectrum[v, u]が、縦方向の
            周波数v・横方向の周波数uの成分。

    Returns:
        形状(H, W)の複素数配列(dtype=complex128)。実数画像のDFT
        結果を渡した場合、理論上は虚部が0になる(丸め誤差でごく
        わずかな値が残ることはある)。
    """
    height, width = spectrum.shape

    # 第1段階: 各行(横方向、u方向)に1次元IDFTをかける。
    rows_transformed = np.zeros((height, width), dtype=np.complex128)
    for y in range(height):
        rows_transformed[y, :] = idft_1d(spectrum[y, :])

    # 第2段階: 第1段階の結果の各列(縦方向、v方向)に1次元IDFTを
    # かける。
    result = np.zeros((height, width), dtype=np.complex128)
    for x in range(width):
        result[:, x] = idft_1d(rows_transformed[:, x])

    return result
