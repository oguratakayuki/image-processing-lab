"""せん断(Shear)。

数学的背景:
    せん断は、一方の座標をもう一方の座標に比例してずらす変換で
    あり、長方形を平行四辺形に歪ませる。2x2のせん断行列による
    線形写像として表せる。

        [x']   [1    shx]   [x]
        [y'] = [shy  1  ] * [y]

        つまり: x' = x + shx*y,  y' = shy*x + y

    shx(横方向のせん断係数)は、yが大きい行ほどxを大きくずらす
    (縦方向の位置に応じて横にずれる)。shyはその逆で、xが大きい
    列ほどyを大きくずらす。

    せん断行列の逆行列(一般の2x2行列の逆行列):
        [5-2の対角行列](scaling.py)は対角成分の逆数を取るだけ、
        [5-3の回転行列](rotation.py)は転置を取るだけ、という
        それぞれ特別な形の逆行列の求め方だったが、せん断行列は
        どちらの特別な性質も持たない一般的な2x2行列である。
        一般の2x2行列 M = [[a,b],[c,d]] の逆行列は、
        行列式(determinant) det = a*d - b*c を使って

            M^-1 = (1/det) * [[d, -b], [-c, a]]

        と計算できる(detが0だと逆行列が存在しない=変換が
        不可逆になる)。せん断行列(a=1, b=shx, c=shy, d=1)に
        当てはめると、det = 1 - shx*shy であり、

            [1    shx]^-1   1        [1    -shx]
            [shy  1  ]    = ------- * [-shy  1  ]
                            1-shx*shy

        逆方向マッピングでは、この逆行列を出力座標に掛けて
        入力側の座標を逆算する。

        src_x = (out_x - shx*out_y) / det
        src_y = (-shy*out_x + out_y) / det

    原点を基準にする(回転のような中心合わせは不要):
        [5-3の回転](rotation.py)は原点を中心に回転させると画像が
        枠の外に丸ごとずれてしまうため、画像の中心を軸にする必要が
        あった。せん断は性質が異なり、`y=0`の行(画像の一番上の行)
        は`x'=x+shx*0=x`となり全く動かない。原点(左上の角)を
        基準にしたまま、行ごとに少しずつ横にずらしていく変換として
        素直に扱える(平行移動・拡大縮小と同じく中心合わせ不要)。

    補間・境界処理:
        [5-3の回転](rotation.py)と同じく、逆算した座標が画像の
        範囲外になる場合は[5-1の平行移動](translation.py)と同じ
        「黒で埋める」方式を使う(せん断でも画像の一部が枠の外に
        押し出される領域が本当に生じるため、クランプではなく
        黒埋めが適切)。
"""

import numpy as np


def shear(image: np.ndarray, shx: float, shy: float) -> np.ndarray:
    """画像をせん断変換する(逆方向マッピング、最近傍補間)。

    前提条件:
        imageは(H, W)のGrayscale画像、または(H, W, 3)のRGB画像の
        どちらでもよい([5-1](translation.py)〜[5-3](rotation.py)と
        同じく、座標だけを動かす変換のため、色の値そのものには
        一切手を加えない)。

    Args:
        image: せん断する画像。(H, W)または(H, W, 3)のuint8配列。
        shx: 横方向のせん断係数。0で変化なし。行(y座標)が大きい
            ほどxが大きくずれる。
        shy: 縦方向のせん断係数。0で変化なし。列(x座標)が大きい
            ほどyが大きくずれる。

    Returns:
        せん断変換後の画像。入力と同じ形状・dtype。枠からはみ出した
        部分は切り捨てられ、新しく現れた部分は0(黒)で埋められる。
    """
    height, width = image.shape[:2]

    # せん断行列の行列式。0になると逆行列が存在しない
    # (shx*shy=1となる組み合わせは避ける想定)。
    det = 1 - shx * shy

    output = np.zeros_like(image)

    for out_y in range(height):
        for out_x in range(width):
            # せん断行列の逆行列を出力座標に適用して、入力側の
            # 座標を逆算する。
            src_x = (out_x - shx * out_y) / det
            src_y = (-shy * out_x + out_y) / det

            # 最近傍補間: 非整数座標を四捨五入する。
            nearest_y = round(src_y)
            nearest_x = round(src_x)

            # 入力座標は実際に画像の中に存在するかどうか。
            if 0 <= nearest_y < height and 0 <= nearest_x < width:
                output[out_y, out_x] = image[nearest_y, nearest_x]
            # 範囲外(せん断によって画像の外から来た/はみ出した部分)
            # なら、outputはすでに0で初期化済みなので何もしない。

    return output
