"""回転(Rotation)。

数学的背景:
    角度theta(度数法)だけ回転させる変換は、2x2の回転行列による
    線形写像として表せる。

        [x']   [cosθ  -sinθ]   [x]
        [y'] = [sinθ   cosθ] * [y]

    [5-2の拡大縮小](scaling.py)と同じく逆方向マッピングを使う。
    出力の各点に対応する入力側の座標を、逆変換(逆行列)で逆算する。

    回転行列の逆行列:
        回転行列は「直交行列」(各列ベクトルの大きさが1で、互いに
        直交する)という特別な性質を持ち、直交行列の逆行列は
        転置行列(行と列を入れ替えたもの)に一致する。さらに回転の
        場合、転置を取ることは「逆向きに回転する」ことと同じになる
        (cos(-θ)=cosθ, sin(-θ)=-sinθ という性質から)。

            [cosθ  -sinθ]^-1   [cosθ   sinθ]   [cos(-θ)  -sin(-θ)]
            [sinθ   cosθ]    = [-sinθ  cosθ] = [sin(-θ)   cos(-θ)]

        つまり「thetaだけ回転の逆変換」は「-thetaだけ回転」に
        一致する。[5-2の対角行列の逆行列](scaling.py)(対角成分の
        逆数を取るだけ)とは違う形の「逆行列の求め方」の例になっている。

    回転中心の問題:
        原点(0, 0)を中心に回転させると、画像全体が枠の外にずれて
        しまう(回転は原点からの距離・角度を保つ変換なので、原点から
        離れた画像全体が弧を描くように移動する)。これを避けるため、
        このモジュールでは**画像の中心**を基準に回転させる。出力の
        座標からいったん中心を引いて「中心からの相対座標」にし、
        逆回転を適用し、最後にまた中心を足して元の座標系に戻す。

    最近傍補間と、はみ出た部分の扱い:
        [5-2](scaling.py)と同じく、逆算した座標は一般に非整数になる
        ため最近傍補間(四捨五入)を使う。ただし[5-2]のクランプ(範囲外
        を端の値で埋める)とは異なり、ここでは[5-1の平行移動]
        (translation.py)と同じ「範囲外は黒で埋める」方式を使う。
        回転で画像の四隅が回転後に枠の外へ出ていく(例えば45度回転
        すると、元の四隅は入力画像の外側に対応する)のは、丸め誤差
        ではなく正しい挙動だからである。クランプしてしまうと、
        本来何も無いはずの領域に端の画素が間違って引き伸ばされて
        しまう。
"""

import math

import numpy as np


def rotate(image: np.ndarray, degrees: float) -> np.ndarray:
    """画像を中心を軸にdegrees度回転する(逆方向マッピング、最近傍補間)。

    前提条件:
        imageは(H, W)のGrayscale画像、または(H, W, 3)のRGB画像の
        どちらでもよい([5-1](translation.py)・[5-2](scaling.py)と
        同じく、座標だけを動かす変換のため、色の値そのものには
        一切手を加えない)。

    Args:
        image: 回転する画像。(H, W)または(H, W, 3)のuint8配列。
        degrees: 回転角度(度数法)。正の値で時計回りに回転する
            (通常の数学の座標系ではyが上向きなので反時計回りが
            正になるが、画像座標系はyが下向きのため、見た目では
            時計回りになる)。

    Returns:
        回転後の画像。入力と同じ形状・dtype。画像の中心を軸に
        回転し、枠からはみ出した部分は切り捨てられ、新しく現れた
        部分は0(黒)で埋められる。
    """
    height, width = image.shape[:2]

    # 画像の中心座標。(N-1)/2としているのは、座標が0からN-1までの
    # N個の整数で構成されているため、ちょうど真ん中の値は
    # (0 + (N-1)) / 2 になるという理由による。
    center_y = (height - 1) / 2.0
    center_x = (width - 1) / 2.0

    # math.radians()は度数法(degrees)を弧度法(radian)に変換する
    # Python標準ライブラリの関数。NumPyのcos/sinも弧度法を前提と
    # しているため、ここで変換しておく。
    theta = math.radians(degrees)
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)

    output = np.zeros_like(image)

    for out_y in range(height):
        for out_x in range(width):
            # 中心を原点とみなした、出力座標の相対位置。
            rel_y = out_y - center_y
            rel_x = out_x - center_x

            # 逆回転(-theta分の回転)を適用して、入力側の相対座標を
            # 求める。上のdocstringで導出した逆行列をそのまま使う。
            src_rel_x = cos_theta * rel_x + sin_theta * rel_y
            src_rel_y = -sin_theta * rel_x + cos_theta * rel_y

            # 中心を足し戻して、画像全体の座標系に戻す。
            src_y = src_rel_y + center_y
            src_x = src_rel_x + center_x

            # 最近傍補間: 非整数座標を四捨五入する。
            nearest_y = round(src_y)
            nearest_x = round(src_x)

            # 入力座標は実際に画像の中に存在するかどうか。
            if 0 <= nearest_y < height and 0 <= nearest_x < width:
                output[out_y, out_x] = image[nearest_y, nearest_x]
            # 範囲外(回転によって画像の外から来た/はみ出した部分)
            # なら、outputはすでに0で初期化済みなので何もしない。

    return output
