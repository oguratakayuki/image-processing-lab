"""Sobel法によるエッジ検出。

数学的背景:
    エッジ(輪郭)とは「画像の明るさが急激に変化する場所」であり、
    数学的には輝度関数 I(x, y) の変化率、つまり微分が大きい場所として
    定式化できる。

    1変数関数の導関数の定義:

        f'(x) = lim_{h→0} (f(x+h) - f(x)) / h

    画像は離散的な格子上の値なので極限を取ることはできず、有限差分
    (finite difference)で近似する。中心差分(前後の値を使う近似)を
    使うと:

        f'(x) ≈ (f(x+1) - f(x-1)) / 2

    画像 I(x, y) は2変数関数なので、x方向・y方向それぞれの偏微分を
    考える:

        ∂I/∂x ≈ I(x+1, y) - I(x-1, y)
        ∂I/∂y ≈ I(x, y+1) - I(x, y-1)

    この2つの偏微分をまとめたベクトルが勾配(gradient)である:

        ∇I = (∂I/∂x, ∂I/∂y)

    勾配ベクトルは「輝度が最も急激に増加する方向」を指すベクトルで
    あり、その大きさ(ユークリッドノルム)

        |∇I| = √((∂I/∂x)² + (∂I/∂y)²)

    が「その場所がどれだけエッジらしいか」を表す。これは線形代数で
    習うベクトルの大きさ(2乗和の平方根)がそのまま画像処理に登場する
    例である。

    Sobel法は、単純な差分 I(x+1,y) - I(x-1,y) だけでなく、微分方向に
    直交する方向へ [1, 2, 1] という重みで平滑化(簡易的なノイズ除去)
    を組み合わせた、次の3x3カーネルを使う:

        Gx = [[-1, 0, 1],       Gy = [[-1, -2, -1],
              [-2, 0, 2],             [ 0,  0,  0],
              [-1, 0, 1]]             [ 1,  2,  1]]

    Gxは横方向に [-1,0,1](差分=微分)、縦方向に [1,2,1](平滑化)を
    かけ合わせた形になっている(実際 Gx は縦ベクトル[1,2,1]^Tと
    横ベクトル[-1,0,1]の外積に一致する。カーネルの分離可能性の
    もう1つの例)。GyはGxを転置した形で、微分と平滑化の方向が
    入れ替わっている。

    Gx, Gyはどちらも非対称なカーネル(Gx(i,j) != Gx(-i,-j))である。
    畳み込みトピックで学んだ通り、非対称カーネルでは畳み込み(反転
    あり)と相関(反転なし)で符号や向きが変わる。ここでは
    imglab_engine.convolution.convolve2d による「本物の畳み込み」を
    使う。
"""

import numpy as np
from imglab_engine.convolution.convolution import convolve2d

# x方向の微分(横方向の差分) x 縦方向の平滑化[1,2,1]
SOBEL_X = np.array(
    [
        [-1.0, 0.0, 1.0],
        [-2.0, 0.0, 2.0],
        [-1.0, 0.0, 1.0],
    ]
)

# y方向の微分(縦方向の差分) x 横方向の平滑化[1,2,1] (SOBEL_Xの転置)
SOBEL_Y = np.array(
    [
        [-1.0, -2.0, -1.0],
        [0.0, 0.0, 0.0],
        [1.0, 2.0, 1.0],
    ]
)


def sobel_gradient(channel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sobelカーネルで勾配ベクトルの成分 (Gx, Gy) を計算する。

    勾配は符号付きの値(輝度が減少する方向のエッジでは負になる)を
    取るため、convolve2d を quantize=False で呼び出し、0-255への
    クリップやuint8化を行わない生のfloat64値を返す。
    """
    gx = convolve2d(channel, SOBEL_X, quantize=False)
    gy = convolve2d(channel, SOBEL_Y, quantize=False)
    return gx, gy


def gradient_magnitude(gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    """勾配ベクトルの大きさ(ユークリッドノルム) |∇I| = √(Gx² + Gy²) を計算する。

    大きさは常に非負の値になるので、画像として表示するために
    0-255へ正規化する。理論上の最大値(√2 × 1020 ≈ 1442, 入力が
    0/255の完全な白黒パターンでSobelカーネルの係数が最大化された
    場合)で正規化する方法もあるが、実際の画像でそこまで極端な
    値になることは稀で、コントラストが低く見えてしまう。ここでは
    「この画像の中での最大値」で正規化し、画像ごとにエッジの
    強弱が見やすくなるようにしている(ヒストグラムのビン表示を
    max値で正規化したのと同じ考え方)。
    """
    magnitude = np.sqrt(gx**2 + gy**2)
    max_value = magnitude.max()
    if max_value == 0:
        return magnitude.astype(np.uint8)
    normalized = magnitude / max_value * 255
    return np.round(normalized).astype(np.uint8)
