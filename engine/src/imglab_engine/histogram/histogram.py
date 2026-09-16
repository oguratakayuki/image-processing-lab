"""ヒストグラム(度数分布)の計算。

数学的背景:
    ヒストグラムは、各ピクセル値 v (0〜255) が画像中に何回出現するかを
    数えた度数分布である。

        h(v) = |{(i, j) : I(i, j) = v}|,  v = 0, 1, ..., 255

    総ピクセル数 N = H*W で割った p(v) = h(v)/N は、ピクセル値という
    確率変数の経験分布(empirical probability mass function)とみなせる。
    この p(v) から平均・分散を計算できる(mean = Σ v*p(v) など)ことが、
    後の統計・ノイズのトピックに直結する。
"""

import numpy as np


def compute_histogram(channel: np.ndarray) -> np.ndarray:
    """Grayscale画像(uint8, 値域0-255)のヒストグラムを計算する。

    channel: (H, W) の単一チャンネル画像
    戻り値: 長さ256のint64配列。result[v] はピクセル値vの出現回数。

    実装上の注意:
        `hist = np.zeros(256); hist[channel.ravel()] += 1` のような
        素朴な書き方は使えない。NumPyのファンシーインデックスによる
        代入は、同じインデックスが複数回登場しても加算が積み上がらず
        「最後に書き込んだ値」で上書きされてしまうため(重複インデックス
        への += が正しく累積されないという既知の罠)。
        代わりに `np.bincount` を使うと、各値の出現回数を正しく数えられる。
    """
    return np.bincount(channel.ravel(), minlength=256)[:256]
