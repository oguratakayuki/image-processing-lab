"""Prewitt法によるエッジ検出。

Sobelとの違い:
    考え方はSobelと全く同じ(微分方向の差分 × 直交方向の平滑化)だが、
    平滑化に使う重みが異なる。Sobelは [1, 2, 1](2項係数、離散的な
    ガウシアンの近似)で中心を重視した平滑化をしたが、Prewittは
    単純な均等平均 [1, 1, 1] を使う。

        Prewitt_X = [[-1, 0, 1],      Prewitt_Y = [[-1, -1, -1],
                     [-1, 0, 1],                    [ 0,  0,  0],
                     [-1, 0, 1]]                    [ 1,  1,  1]]

    これは畳み込みトピックで学んだ mean_kernel(均等平均)と
    gaussian_kernel(中心を重視した重み付き平均)の違いが、そのまま
    エッジ検出の平滑化方向にも表れている例である。Sobelの方が
    中心行/列の影響を強く受ける分、ノイズに強いとされる
    (Prewittは全ての行が同じ重みなので、外側の行のノイズが
    Sobelより結果に影響しやすい)。
"""

import numpy as np
from imglab_engine.convolution.convolution import convolve2d

# x方向の微分(横方向の差分) x 縦方向の均等平滑化[1,1,1]
PREWITT_X = np.array(
    [
        [-1.0, 0.0, 1.0],
        [-1.0, 0.0, 1.0],
        [-1.0, 0.0, 1.0],
    ]
)

# y方向の微分(縦方向の差分) x 横方向の均等平滑化[1,1,1] (PREWITT_Xの転置)
PREWITT_Y = np.array(
    [
        [-1.0, -1.0, -1.0],
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
    ]
)


def prewitt_gradient(channel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Prewittカーネルで勾配ベクトルの成分 (Gx, Gy) を計算する。

    sobel_gradientと同様、符号付きの値を保つため quantize=False で
    convolve2dを呼び出す。
    """
    gx = convolve2d(channel, PREWITT_X, quantize=False)
    gy = convolve2d(channel, PREWITT_Y, quantize=False)
    return gx, gy
