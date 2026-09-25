"""OpenCVによる畳み込み(相関)の参照実装。答え合わせ・比較専用。

このモジュールは `imglab_engine.convolution` 側からは一切importされない
(依存の向きを「自前実装からOpenCVへは依存しない」の一方向に保つため)。
テストやAPIの「比較モード」からのみ利用する。

数学的な注意(重要):
    OpenCVの `cv2.filter2D` は、名前とは裏腹に畳み込み(Convolution)
    ではなく相関(Correlation)を計算する(カーネルを反転しない)。
    OpenCVの公式ドキュメントにも「本当の畳み込みが欲しいなら
    flip()でカーネルを反転してから使え」と明記されている。

    そのため:
        - 対称なカーネル(mean, gaussianなど。K(i,j)=K(-i,-j))では、
          反転してもしなくても同じカーネルになるので、
          自前の convolve2d(畳み込み) と cv2.filter2D(相関) は
          同じ結果になる。
        - 非対称なカーネル(例: 中心の1つ左だけが1のカーネル)では、
          畳み込みと相関で「シフトする方向が逆」になり、
          結果が一致しない。

    この関数はcv2の挙動をそのまま反映しており、意図的に反転を
    加えていない(「本物の畳み込み」に揃えたい場合は、呼び出し側で
    `kernel[::-1, ::-1]` を渡せばよい)。

    境界処理について:
        OpenCVのデフォルトの境界処理は BORDER_REFLECT_101(鏡映)で
        あり、本プロジェクトのconvolve2dが採用しているゼロパディング
        とは異なる。境界処理の違いが比較結果に混ざらないよう、
        ここでは明示的に `borderType=cv2.BORDER_CONSTANT` (ゼロ埋め)
        を指定している。
"""

import cv2
import numpy as np


def cv2_filter2d(channel: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """cv2.filter2D(相関, ゼロパディング)で畳み込み相当の処理を行う。

    convolve2dと比較する際は、境界処理(ゼロパディング)を揃えた上で、
    対称カーネルなら一致、非対称カーネルなら不一致になることを
    確認するために使う。
    """
    result = cv2.filter2D(
        channel.astype(np.float64),
        ddepth=-1,
        kernel=kernel,
        borderType=cv2.BORDER_CONSTANT,
    )
    return np.clip(np.round(result), 0, 255).astype(np.uint8)
