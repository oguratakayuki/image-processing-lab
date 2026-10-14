"""画素値の補間方法(Interpolation): 最近傍補間とバイリニア補間。

数学的背景:
    [5-2](scaling.py)以降、逆方向マッピングで逆算した入力側の座標は
    一般に非整数になる。この非整数座標の「値」をどう決めるかが
    補間(interpolation)の役割。

    最近傍補間(nearest neighbor):
        非整数座標を四捨五入し、最も近い1つの整数格子点の値を
        そのまま使う。計算が軽く実装も単純だが、拡大・回転で
        隣接ピクセルが同じ値を繰り返してブロック状に見える、
        細かい模様が欠落する([5-3の回転](rotation.py)で見た、
        孤立した1点が消えることがある)などの欠点がある。

    バイリニア補間(bilinear interpolation):
        非整数座標(x, y)を取り囲む4つの整数格子点
        (x0,y0), (x1,y0), (x0,y1), (x1,y1)
        (x0=floor(x), x1=x0+1, y0=floor(y), y1=y0+1)の値を、
        距離に応じた重みで混ぜ合わせる。

        まずx方向に、上の2点・下の2点それぞれを1次元の線形補間
        (端数 fx = x - x0 を使った加重平均)で合成する。

            top    = (1-fx)・I(x0,y0) + fx・I(x1,y0)
            bottom = (1-fx)・I(x0,y1) + fx・I(x1,y1)

        続けてy方向に、topとbottomを同じく線形補間する
        (端数 fy = y - y0)。

            value = (1-fy)・top + fy・bottom

        「x方向の線形補間を2回→その結果をy方向に1回線形補間」という、
        3回の1次元線形補間の組み合わせになっている(これが"bi"linear
        =「2方向の」線形補間、という名前の由来)。最近傍補間のような
        値の急な切り替わりが無く、滑らかな結果になる。

    範囲外の扱い:
        4つの格子点のうち画像の外にあるものは、[5-1の平行移動]
        (translation.py)などと同じく0(黒)として扱う。
"""

import math

import numpy as np


def _pixel_or_zero(image: np.ndarray, y: int, x: int):
    """座標(y, x)が画像の範囲内ならその画素値を、範囲外なら0を返す。

    Args:
        image: (H, W)または(H, W, 3)のuint8配列。
        y, x: 整数の格子点座標。

    Returns:
        float64の画素値。(H, W)画像ならスカラー、(H, W, 3)画像なら
        長さ3の配列。
    """
    height, width = image.shape[:2]
    if 0 <= y < height and 0 <= x < width:
        return image[y, x].astype(np.float64)
    if image.ndim == 2:
        return np.float64(0.0)
    return np.zeros(image.shape[2], dtype=np.float64)


def nearest_neighbor_sample(image: np.ndarray, y: float, x: float):
    """座標(y, x)に最も近い整数格子点の画素値を返す。

    Args:
        image: (H, W)または(H, W, 3)のuint8配列。
        y, x: サンプリングしたい座標(非整数でもよい)。

    Returns:
        float64の画素値(範囲外なら0)。
    """
    return _pixel_or_zero(image, round(y), round(x))


def bilinear_sample(image: np.ndarray, y: float, x: float):
    """座標(y, x)を囲む4点の加重平均で画素値を補間する。

    Args:
        image: (H, W)または(H, W, 3)のuint8配列。
        y, x: サンプリングしたい座標(非整数でもよい)。

    Returns:
        float64の画素値(補間に使う4点のうち範囲外のものは0として
        扱う)。
    """
    # math.floor()は小数点以下を切り捨てる(負の数でも「より小さい
    # 方の整数」に丸める。例: floor(-0.5) = -1)。四捨五入のround()
    # とは挙動が異なる点に注意。
    y0 = math.floor(y)
    x0 = math.floor(x)
    y1 = y0 + 1
    x1 = x0 + 1

    # 端数(fractional part): 格子点y0からどれだけ進んだ位置かを
    # 0.0〜1.0の割合で表す。
    fy = y - y0
    fx = x - x0

    # x方向にまず2回、1次元の線形補間をする。
    top = _pixel_or_zero(image, y0, x0) * (1 - fx) + _pixel_or_zero(image, y0, x1) * fx
    bottom = (
        _pixel_or_zero(image, y1, x0) * (1 - fx) + _pixel_or_zero(image, y1, x1) * fx
    )

    # 続けてy方向に1回、線形補間する。
    return top * (1 - fy) + bottom * fy
