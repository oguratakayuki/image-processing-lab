"""勾配ベクトル(Gx, Gy)から大きさを求める、微分手法に依存しない共通処理。

Sobel・Prewittはどちらも「x方向・y方向の勾配を求めてから大きさを
計算する」という後半部分は共通しているため、ここに切り出している。
"""

import numpy as np


def gradient_magnitude(gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    """勾配ベクトルの大きさ(ユークリッドノルム) |∇I| = √(Gx² + Gy²) を計算する。

    大きさは常に非負の値になるので、画像として表示するために
    0-255へ正規化する。理論上の最大値(カーネルの係数と入力の
    値域0-255から決まる固定値)で正規化する方法もあるが、実際の
    画像でそこまで極端な値になることは稀で、コントラストが低く
    見えてしまう。ここでは「この画像の中での最大値」で正規化し、
    画像ごとにエッジの強弱が見やすくなるようにしている
    (ヒストグラムのビン表示をmax値で正規化したのと同じ考え方)。
    """
    magnitude = np.sqrt(gx**2 + gy**2)
    max_value = magnitude.max()
    if max_value == 0:
        return magnitude.astype(np.uint8)
    normalized = magnitude / max_value * 255
    return np.round(normalized).astype(np.uint8)
