"""明るさ(Brightness)・コントラスト(Contrast)調整。

数学的背景:
    ピクセル値に対するアフィン変換(affine transformation)。

        g(x) = alpha * x + beta

    - alpha (コントラスト係数): 1より大きいと値同士の差が強調され
      (コントラスト増)、1より小さいと値同士が近づく(コントラスト減)。
      x=0 を固定点とするスケーリングなので、暗いピクセルほど動きが
      小さく、明るいピクセルほど大きく動く。
    - beta (明るさオフセット): 全ピクセルを一律にシフトする(明るさ調整)。

    grayscale.py の f(v) = w・v は「線形写像」(f(0)=0 が必ず成り立つ)
    だったが、g(x) = alpha*x + beta は beta という定数項を持つため
    「アフィン変換」になる(線形写像に平行移動を加えたもの)。
    beta!=0 のとき g(0) = beta != 0 となり、原点が原点に写らない
    点が線形写像との違い。
"""

import numpy as np


def adjust_brightness_contrast(
    image: np.ndarray, alpha: float = 1.0, beta: float = 0.0
) -> np.ndarray:
    """画像の明るさ・コントラストを調整する。

    g(x) = alpha * x + beta を全ピクセルに要素ごとに適用する。
    Grayscale画像((H, W))・RGB画像((H, W, 3))のどちらに対しても
    同じ式がそのまま使える(チャンネルの有無によらない要素ごとの
    演算のため)。
    """
    image_float = image.astype(np.float64)

    # アフィン変換: 各要素に一律 alpha 倍 + beta シフトを適用
    transformed = alpha * image_float + beta

    # 飽和(saturation): grayscale変換のときは重みの和が1という制約で
    # 自動的に0-255に収まっていたが、alpha/betaは任意の値を取り得る
    # ためここでは明示的なクリップが必要。
    clipped = np.clip(transformed, 0, 255)

    # Quantization: 四捨五入してuint8に戻す。
    return np.round(clipped).astype(np.uint8)
