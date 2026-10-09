"""拡大縮小(Scaling)。

数学的背景:
    拡大縮小は、座標(x, y)をそれぞれ独立にsx倍・sy倍する変換であり、
    2x2の対角行列による線形写像として表せる。

        [x']   [sx  0 ]   [x]
        [y'] = [0   sy] * [y]

        つまり: x' = sx * x,  y' = sy * y

    [5-1の平行移動](translation.py)と同じく逆方向マッピングを使う。
    出力の各点(out_x, out_y)について、対応する入力側の座標を
    「逆変換」で逆算する。対角行列の逆行列は、対角成分の逆数を
    取るだけでよい(対角行列同士の積が対角成分の積になる性質から、
    [[sx,0],[0,sy]] * [[1/sx,0],[0,1/sy]] = 単位行列、となることで
    確認できる)。

        src_x = out_x / sx
        src_y = out_y / sy

    平行移動との決定的な違い:
        平行移動は整数シフトだったため、逆算した入力側の座標は常に
        整数だった。拡大縮小では、sx, syが整数倍以外の値(例: 1.5倍)
        になりうるため、逆算した座標 src_x, src_y は一般に非整数に
        なる。しかし画像データは整数座標の格子点にしか存在しない
        ため、非整数座標の「値」をそのままでは取得できない。

    最近傍補間(nearest neighbor interpolation):
        この問題への最も単純な対処法が、逆算した非整数座標を
        四捨五入して、最も近い整数座標の画素値をそのまま使う方法。
        計算が軽く実装も単純だが、拡大時に同じ画素が繰り返されて
        ブロック状に見える、縮小時に細部が失われやすいという
        欠点がある(周辺4点の加重平均を取る「バイリニア補間」で
        滑らかにする方法は別のトピックで扱う)。

        丸めた座標が画像の範囲をわずかに超える場合(丸め誤差による
        境界ちょうどのケース)は、範囲内に収まるよう上限・下限で
        挟み込む(クランプ, clamping)。[5-1の平行移動](translation.py)
        では範囲外を黒で埋めたが、拡大縮小では「画像の端の画素を
        そのまま引き伸ばす」方が見た目として自然なため、別の対処
        (クランプ)を選んでいる。

    出力画像のサイズ:
        [5-1の平行移動](translation.py)は出力サイズが入力と同じ
        (画像の外にはみ出た部分を切り捨てる)だったが、拡大縮小は
        「画像そのものを拡大・縮小する」操作なので、出力画像の
        サイズ自体を sx, sy 倍にする(例: 20x20画像を2倍すると
        40x40になる)。
"""

import numpy as np


def scale(image: np.ndarray, sx: float, sy: float) -> np.ndarray:
    """画像を(sx, sy)倍に拡大縮小する(逆方向マッピング、最近傍補間)。

    前提条件:
        imageは(H, W)のGrayscale画像、または(H, W, 3)のRGB画像の
        どちらでもよい([5-1の平行移動](translation.py)と同じく、
        座標だけを動かす変換のため、色の値そのものには一切手を
        加えない)。

    Args:
        image: 拡大縮小する画像。(H, W)または(H, W, 3)のuint8配列。
        sx: x方向(横方向)の拡大率。2.0なら2倍、0.5なら半分になる。
        sy: y方向(縦方向)の拡大率。sxと独立に指定できる(異なる値を
            指定すると縦横比が変わる)。

    Returns:
        拡大縮小後の画像。(H, W)画像なら(round(H*sy), round(W*sx))、
        (H, W, 3)画像なら(round(H*sy), round(W*sx), 3)のuint8配列。
    """
    height, width = image.shape[:2]

    # 出力画像のサイズを、入力サイズにsx, syを掛けて求める。
    # Pythonのround()は「0.5のときは最も近い偶数に丸める」という
    # 銀行丸め(round half to even)を使う(Ruby/PHPのroundのような
    # 「0.5は常に切り上げ」ではない)。例えばround(2.5)は2、
    # round(3.5)は4になる。画像サイズの計算では通常問題にならない
    # ほど小さい誤差だが、Pythonの丸めの挙動として覚えておく価値がある。
    new_height = round(height * sy)
    new_width = round(width * sx)

    # image.shapeが(H, W, 3)ならチャンネル数も含めて出力の形を作り、
    # (H, W)ならチャンネル軸なしで作る。image.shape[2:]は、
    # (H, W, 3)なら(3,)、(H, W)なら(空のタプル)になるスライスで、
    # これをタプル同士の連結(+)で出力の形状に継ぎ足している。
    output_shape = (new_height, new_width) + image.shape[2:]
    output = np.zeros(output_shape, dtype=image.dtype)

    for out_y in range(new_height):
        for out_x in range(new_width):
            # 逆方向マッピング: 出力の点(out_y, out_x)に対応する
            # 入力側の座標を、拡大縮小の逆変換(sx, syで割る)で逆算する。
            # sx, syが1以外のとき、この時点では一般に非整数になる。
            src_y = out_y / sy
            src_x = out_x / sx

            # 最近傍補間: 非整数座標を四捨五入して、最も近い整数座標
            # (入力画像上の実在するピクセル)に丸める。
            nearest_y = round(src_y)
            nearest_x = round(src_x)

            # クランプ: 丸めた座標が画像の範囲をわずかに超える場合
            # (丸め誤差による境界ちょうどのケース)に備えて、範囲内の
            # 最も近い値に強制的に収める。
            # max(0, ...)で下限0を保証し、min(..., height-1)で
            # 上限height-1を保証する(Ruby/PHPのclamp相当の処理を、
            # Pythonにはclamp関数が無いためmax/minの組み合わせで書く)。
            clamped_y = max(0, min(nearest_y, height - 1))
            clamped_x = max(0, min(nearest_x, width - 1))

            output[out_y, out_x] = image[clamped_y, clamped_x]

    return output
