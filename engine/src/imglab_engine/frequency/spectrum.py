"""2次元DFTの結果を、人の目で見やすい形に変換する。

数学的背景:
    振幅スペクトラム:
        2次元DFTの結果F(v, u)は複素数で、そのままでは画像として
        表示できない。「どれだけその周波数成分が強く含まれているか」
        だけを見たい場合は、絶対値(振幅)|F(v, u)|を使う。

    対数スケールが必要な理由:
        自然画像のDFT結果は、直流成分F(0, 0)(画素値の総和)が他の
        成分より桁違いに大きくなりやすい。振幅をそのまま画像として
        表示すると、直流成分だけが明るく、他の全ての点がほぼ真っ黒
        になってしまい、高周波成分の様子が見えなくなる。
        log(1 + |F(v, u)|)という対数スケールに変換すると、大きい値
        は圧縮され、小さい値との差が縮まり、全体の濃淡が見やすくなる
        (+1するのは、|F|=0の点でlog(0)が未定義になるのを避けるため)。

    中心シフトが必要な理由:
        dft_2d()が返す配列は、直流成分F(0, 0)が配列の左上(インデックス
        [0, 0])に来る並び方になっている。人がスペクトラム画像を見る
        ときは、「低周波(変化が緩やか)が中央に集まり、高周波(変化が
        激しい)ほど外側に広がる」という並びの方が直感的に理解しやすい。
        そこで、配列を4つの象限に分割し、対角の象限を入れ替えること
        で、直流成分が配列の中央に来るように並べ替える。
"""

import numpy as np

from imglab_engine.frequency.dft import dft_2d


def _shift_to_center(array_2d: np.ndarray) -> np.ndarray:
    """2次元配列の象限を入れ替え、左上(インデックス[0,0])にある値を中央へ移動する。

    前提条件:
        array_2dは2次元の数値配列であること。

    Args:
        array_2d: 形状(H, W)の2次元配列。

    Returns:
        形状(H, W)の2次元配列。array_2d[0, 0]の値が、
        array_2d[H//2, W//2]の位置に移動している。
    """
    height, width = array_2d.shape
    half_height, half_width = height // 2, width // 2

    # 配列を4つの象限(左上・右上・左下・右下)に分割する。
    # array_2d[:half_height, :half_width] は「上半分の行、左半分の列」
    # を取り出すスライス構文(Ruby/PHPの配列には無い書き方)。
    top_left = array_2d[:half_height, :half_width]
    top_right = array_2d[:half_height, half_width:]
    bottom_left = array_2d[half_height:, :half_width]
    bottom_right = array_2d[half_height:, half_width:]

    # 対角にある象限どうしを入れ替えて結合する
    # (左上⇔右下、右上⇔左下)。これにより、元々[0,0]にあった値は
    # 右下の象限の中に含まれていたので、結合後は配列の中央付近に来る。
    # np.concatenateは、複数の配列を指定した軸(axis)に沿って連結する関数。
    # axis=1は横方向(列を増やす方向)、axis=0は縦方向(行を増やす方向)の連結。
    top_half = np.concatenate([bottom_right, bottom_left], axis=1)
    bottom_half = np.concatenate([top_right, top_left], axis=1)
    return np.concatenate([top_half, bottom_half], axis=0)


def magnitude_spectrum(image: np.ndarray) -> np.ndarray:
    """2次元DFTの振幅スペクトラムを、表示用に整えて計算する。

    処理の流れ: dft_2d()で周波数成分を求める→絶対値(振幅)を取る→
    対数スケールに変換する→直流成分が中央に来るよう並べ替える。

    前提条件:
        imageは2次元の数値配列(グレースケール画像の輝度値など)で
        あること(実数・複素数どちらでもよい)。

    Args:
        image: 形状(H, W)の2次元配列。

    Returns:
        形状(H, W)の実数配列(dtype=float64)。値が大きいほど、その
        位置に対応する周波数成分が強く含まれていることを示す。
        配列の中央([H//2, W//2])が直流成分(最も低い周波数)に対応し、
        中央から離れるほど高い周波数に対応する。
    """
    spectrum = dft_2d(image)

    # np.absは複素数の絶対値(振幅)を計算する関数。
    # 複素数 a+bi の絶対値は sqrt(a^2+b^2) で、信号処理の文脈では
    # 「その周波数成分がどれだけ強く含まれているか」を表す。
    magnitude = np.abs(spectrum)

    # np.logは自然対数(底がe)を計算する関数。1を足してから対数を
    # 取るのは、magnitude=0の点でlog(0)=マイナス無限大になってしまう
    # のを避けるため(モジュール冒頭の数学的背景を参照)。
    log_magnitude = np.log(1 + magnitude)

    return _shift_to_center(log_magnitude)
