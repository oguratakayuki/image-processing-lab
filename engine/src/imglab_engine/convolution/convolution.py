"""畳み込み(Convolution)。

数学的背景:
    連続関数の畳み込みは積分で定義される。

        (f * h)(x) = ∫ f(τ) h(x - τ) dτ

    離散2次元(画像)版は、積分を和に置き換えたもの。カーネルの
    サイズを (2k+1) x (2k+1) とし、中心を原点とする添字 i, j を
    -k から k まで動かすと:

        (I * K)(x, y) = Σ_{i=-k}^{k} Σ_{j=-k}^{k} K(i, j) * I(x - i, y - j)

    ポイントは I(x - i, y - j) という「引き算」になっている点。
    これは「カーネルを180°回転(上下左右反転)してから、入力画像の
    対応する位置と掛けて足し合わせる」ことと数学的に同値である
    (証明: i' = -i, j' = -j と置換すると、和を取る範囲 [-k,k] は
    符号を反転しても同じ集合になるので、
        Σ_{i,j} K(i,j) I(x-i,y-j) = Σ_{i',j'} K(-i',-j') I(x+i',y+j')
    となり、K(-i',-j') はまさに K を180°回転したものである)。

    多くの画像処理ライブラリ(OpenCVの filter2D など)が「畳み込み」
    と呼んでいる処理は、実はこの反転を行わない「相関(Correlation)」

        (I ⋆ K)(x, y) = Σ_{i,j} K(i, j) * I(x + i, y + j)

    であることが多い。対称なカーネル(平均フィルタ・ガウシアン等、
    K(i,j) = K(-i,-j) を満たすもの)では反転しても同じ結果になる
    ため区別が問題にならないが、非対称なカーネル(後述のSobelなど、
    K(i,j) != K(-i,-j))では畳み込みと相関で結果が変わる。

    この実装では数式に忠実に、カーネルを反転してから積和を取る
    「本物の畳み込み」を行う。

    また、畳み込みには周波数領域との重要な関係がある(畳み込み定理):
    空間領域での畳み込みは、フーリエ変換した後の周波数領域では
    単純な「掛け算」になる。

        F[I * K] = F[I] * F[K]   (F はフーリエ変換, 右辺は要素ごとの積)

    この関係は後の周波数領域(DFT/FFT)のトピックで重要になる
    (大きな画像・大きなカーネルの畳み込みをFFTで高速化できる理由)。
"""

import numpy as np


def convolve2d(channel: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Grayscale画像とカーネルの2次元畳み込みを計算する("same"サイズ出力)。

    channel: (H, W) の単一チャンネル画像 (uint8, 値域0-255)
    kernel:  (kh, kw) の畳み込みカーネル。kh, kw は奇数を想定
             (中心ピクセルが一意に定まるようにするため)。

    境界処理: ゼロパディング(画像の外側は0とみなす)を採用する。
    これは最も単純な境界処理で、出力サイズを入力と同じ (H, W) に
    保つために、上下左右を kh//2, kw//2 だけゼロで埋めてから計算する。

    実装:
        1. カーネルを180°回転する(上述の通り、畳み込み = 反転 + 相関)。
           `kernel[::-1, ::-1]` は行方向・列方向の両方を逆順にする
           スライスで、これが180°回転に相当する。
        2. 画像をゼロパディングする。
        3. 出力の各画素 (y, x) について、パディング後の画像から
           カーネルと同じ形の局所領域(パッチ)を切り出し、カーネルとの
           要素ごとの積の総和 Σ_{i,j} K_flip(i,j) * patch(i,j) を計算する。
           これがまさに畳み込みの定義式そのもの。
    """
    kernel_height, kernel_width = kernel.shape
    flipped_kernel = kernel[::-1, ::-1]  # 180°回転 = 畳み込みの定義に必要な反転

    pad_h, pad_w = kernel_height // 2, kernel_width // 2
    padded = np.pad(
        channel.astype(np.float64),
        ((pad_h, pad_h), (pad_w, pad_w)),
        mode="constant",
        constant_values=0,
    )

    height, width = channel.shape
    output = np.zeros((height, width), dtype=np.float64)

    for y in range(height):
        for x in range(width):
            # パディング後の画像から、出力位置(y,x)に対応する
            # カーネルと同じ形の局所パッチを切り出す。
            patch = padded[y : y + kernel_height, x : x + kernel_width]
            # Σ_{i,j} K_flip(i,j) * patch(i,j) を計算(畳み込みの定義式)。
            output[y, x] = np.sum(patch * flipped_kernel)

    # Quantization: 四捨五入し、0-255の範囲に飽和させてuint8に戻す。
    # (平均フィルタのように総和が1のカーネルならクリップ不要だが、
    #  シャープ化フィルタなど総和がずれるカーネルでは範囲外に出うる
    #  ため、明るさ/コントラスト調整のときと同様にclipが必要)
    clipped = np.clip(output, 0, 255)
    return np.round(clipped).astype(np.uint8)
