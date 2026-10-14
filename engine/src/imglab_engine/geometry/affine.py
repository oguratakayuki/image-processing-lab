"""同次座標によるアフィン変換の統一表現(Homogeneous Coordinates / Affine Transform)。

数学的背景:
    なぜ3次元にする必要があるのか:
        [5-1の平行移動](translation.py)は、2x2の線形変換行列だけでは
        表現できなかった。線形写像 f(v) = M・v は、必ず原点を原点に
        写す(f(0,0) = M・(0,0) = (0,0))が、平行移動は原点を
        (tx, ty)に動かしてしまうため、本質的に「線形」ではない
        (アフィン変換ではあるが、線形写像ではない)。

        この制約を回避するのが同次座標(homogeneous coordinates)。
        2次元の点(x, y)を、3次元のベクトル(x, y, 1)として表す
        (末尾に1を付け足すだけ)。すると、2x2行列では表現できな
        かった平行移動も、3x3行列の掛け算として表現できる。

            [x']   [1  0  tx]   [x]
            [y'] = [0  1  ty] ・ [y]
            [1 ]   [0  0  1 ]   [1]

        1行目・2行目を展開すると x' = x + tx, y' = y + ty となり、
        平行移動の定義そのものになる。末尾の1は、行列を掛けた後も
        常に1のまま保たれる(3行目: 0・x+0・y+1・1 = 1)。

    線形部分と平行移動を1つの行列にまとめる:
        [5-2の拡大縮小](scaling.py)・[5-3の回転](rotation.py)・
        [5-4のせん断](shear.py)で使った2x2の線形変換行列
        M = [[a,b],[c,d]] と、平行移動(tx,ty)を、1つの3x3行列に
        まとめられる。

            [a  b  tx]
            [c  d  ty]
            [0  0  1 ]

        これが「アフィン変換行列」。1行目・2行目を展開すると
        x' = a・x + b・y + tx, y' = c・x + d・y + ty となり、
        「線形変換(a,b,c,d) + 平行移動(tx,ty)」が同時に表現されている。

    変換の合成 = 行列の掛け算:
        「拡大してからさらに回転する」のような、複数の変換を順番に
        適用する処理は、それぞれの変換行列を掛け合わせた1つの行列
        として表現できる(行列の積は「変換の合成」に対応するという、
        線形代数の基本性質)。

            M_combined = M_rotate ・ M_scale

        M_combinedを1回適用するのは、「先にM_scaleを適用し、その
        結果にM_rotateを適用する」のと同じ結果になる(行列の掛け算は
        右から順番に効いていく)。掛ける順序を変える(M_scale・
        M_rotate)と一般に結果が変わる(行列の積は交換法則が成り立た
        ない)ことにも注意。

        [5-3の回転](rotation.py)は「画像の中心を軸に回転する」ために、
        コードの中で「中心を引く→回転→中心を足す」という3段階の
        計算を直接書いていたが、これは

            M = T(center) ・ R(theta) ・ T(-center)

        という3つの行列の積として表現できる。このモジュールの
        `affine_transform()`と、下記の行列生成関数を組み合わせると、
        5-1〜5-4で個別に実装した全ての変換を、この1つの汎用関数で
        再現できる。

    逆方向マッピングとの関係:
        [逆方向マッピング](translation.py)の考え方はここでも同じ。
        出力の各点に対応する入力側の座標を、変換行列の逆行列で
        逆算する。アフイン変換行列(3x3、下段が[0,0,1]の形)の逆行列
        は、2x2の線形部分Mの逆行列M⁻¹([5-4で学んだ行列式を使う
        一般の公式](shear.py)で求める)と、平行移動部分を使って

            M⁻¹_affine = [M⁻¹      -M⁻¹・t]
                          [0  0        1    ]

        という形になる(導出は補足資料を参照)。このモジュールの
        実装では、この式をそのままコードにしている。
"""

import math

import numpy as np

from .interpolation import bilinear_sample, nearest_neighbor_sample


def translation_matrix(tx: float, ty: float) -> np.ndarray:
    """平行移動を表す3x3のアフィン変換行列を作る。

    Args:
        tx: x方向の移動量。
        ty: y方向の移動量。

    Returns:
        3x3のfloat64配列。
    """
    return np.array(
        [
            [1.0, 0.0, tx],
            [0.0, 1.0, ty],
            [0.0, 0.0, 1.0],
        ]
    )


def scaling_matrix(sx: float, sy: float) -> np.ndarray:
    """拡大縮小を表す3x3のアフィン変換行列を作る([5-2](scaling.py)の
    2x2対角行列を、平行移動部分0の3x3に拡張したもの)。

    Args:
        sx: x方向の拡大率。
        sy: y方向の拡大率。

    Returns:
        3x3のfloat64配列。
    """
    return np.array(
        [
            [sx, 0.0, 0.0],
            [0.0, sy, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def rotation_matrix(degrees: float) -> np.ndarray:
    """原点を軸にした回転を表す3x3のアフィン変換行列を作る([5-3]
    (rotation.py)の2x2回転行列を、平行移動部分0の3x3に拡張したもの)。

    Args:
        degrees: 回転角度(度数法)。正の値で時計回り。

    Returns:
        3x3のfloat64配列。
    """
    theta = math.radians(degrees)
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    return np.array(
        [
            [cos_theta, -sin_theta, 0.0],
            [sin_theta, cos_theta, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def shear_matrix(shx: float, shy: float) -> np.ndarray:
    """せん断を表す3x3のアフィン変換行列を作る([5-4](shear.py)の
    2x2せん断行列を、平行移動部分0の3x3に拡張したもの)。

    Args:
        shx: 横方向のせん断係数。
        shy: 縦方向のせん断係数。

    Returns:
        3x3のfloat64配列。
    """
    return np.array(
        [
            [1.0, shx, 0.0],
            [shy, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def affine_transform(
    image: np.ndarray, matrix: np.ndarray, interpolation: str = "nearest"
) -> np.ndarray:
    """3x3のアフィン変換行列で画像を変換する(逆方向マッピング)。

    前提条件:
        imageは(H, W)のGrayscale画像、または(H, W, 3)のRGB画像の
        どちらでもよい([5-1](translation.py)〜[5-4](shear.py)と
        同じく、座標だけを動かす変換のため、色の値そのものには
        一切手を加えない)。matrixは3行3列で、3行目が[0, 0, 1]の
        アフィン変換行列であること(それ以外の形は想定していない)。

    Args:
        image: 変換する画像。(H, W)または(H, W, 3)のuint8配列。
        matrix: 3x3のアフィン変換行列。`translation_matrix()`等の
            生成関数や、それらを`@`演算子で掛け合わせた合成行列を
            渡す。
        interpolation: 補間方法。[5-6](interpolation.py)の
            `"nearest"`(最近傍補間、既定値)か`"bilinear"`
            (バイリニア補間)のいずれか。

    Returns:
        変換後の画像。入力と同じ形状・dtype。枠からはみ出した部分は
        切り捨てられ、新しく現れた部分は0(黒)で埋められる。
    """
    # interpolationの文字列に応じて、実際にサンプリングに使う関数を
    # 選ぶ。この後のループでは、選んだ関数がどちらであるかを気にせず
    # sampler(image, src_y, src_x) という同じ形で呼び出せる。
    if interpolation == "nearest":
        sampler = nearest_neighbor_sample
    elif interpolation == "bilinear":
        sampler = bilinear_sample
    else:
        raise ValueError(f"unknown interpolation: {interpolation!r}")

    height, width = image.shape[:2]

    # 3x3行列の左上2x2を線形部分、右上2x1を平行移動部分として
    # 取り出す(NumPyのスライスで、行列の一部分だけを切り出している)。
    a, b, tx = matrix[0, 0], matrix[0, 1], matrix[0, 2]
    c, d, ty = matrix[1, 0], matrix[1, 1], matrix[1, 2]

    # 2x2線形部分の行列式。[5-4のせん断](shear.py)で導入した、
    # 一般の2x2行列の逆行列を求める公式をそのまま使う。
    det = a * d - b * c

    # 2x2線形部分の逆行列(行列式を使う一般公式)。
    inv_a, inv_b = d / det, -b / det
    inv_c, inv_d = -c / det, a / det

    # 平行移動部分の逆変換: -M^-1・t
    inv_tx = -(inv_a * tx + inv_b * ty)
    inv_ty = -(inv_c * tx + inv_d * ty)

    output = np.zeros_like(image)

    for out_y in range(height):
        for out_x in range(width):
            # 逆行列(線形部分 + 平行移動部分)を出力座標に適用して、
            # 入力側の座標を逆算する。
            src_x = inv_a * out_x + inv_b * out_y + inv_tx
            src_y = inv_c * out_x + inv_d * out_y + inv_ty

            # 選んだ補間方法(最近傍 or バイリニア)で画素値を求める。
            # 範囲外が絡む場合は、samplerの内部で0として扱われる。
            value = sampler(image, src_y, src_x)

            # 補間結果はfloat64なので、画像と同じ型(uint8)に戻す。
            # np.round()は配列・スカラーどちらにも使える(bilinearは
            # (3,)配列、nearestはスカラーまたは(3,)配列になりうる)。
            output[out_y, out_x] = np.round(value).astype(image.dtype)

    return output
