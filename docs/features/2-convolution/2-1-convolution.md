# 2-1. 畳み込み（Convolution）

[← 2. 畳み込みへ戻る](README.md)　｜　**2-1** · [2-2](2-2-kernel-presets.md) · [2-3](2-3-opencv-comparison.md)

| | |
|---|---|
| 学習テーマ | 離散畳み込み、カーネル反転、畳み込み定理 |
| 実装ファイル | [`engine/src/imglab_engine/convolution/convolution.py`](https://github.com/oguratakayuki/image-processing-lab/blob/main/engine/src/imglab_engine/convolution/convolution.py) |
| 関数 | `convolve2d()` |
| テスト | [`engine/tests/convolution/test_convolution.py`](https://github.com/oguratakayuki/image-processing-lab/blob/main/engine/tests/convolution/test_convolution.py) |

## 概念

各出力ピクセルの値を、その周辺（近傍）ピクセルの重み付き和として計算する処理。重みのパターンを表す小さな行列を**カーネル（Kernel）**と呼ぶ。

## 数学的背景

連続関数の畳み込みは積分で定義される。

```
(f * h)(x) = ∫ f(τ) h(x - τ) dτ
```

（`f`と`h`が何を指すかは[補足資料](2-1-convolution-fh-notation.md)を参照）

離散2次元（画像）版は、積分を和に置き換えたもの。カーネルサイズを`(2k+1)×(2k+1)`とし、中心を原点とする添字`i, j`を`-k`から`k`まで動かすと：

```
(I * K)(x, y) = Σ Σ K(i, j) * I(x - i, y - j)
                (i,j について -k から k まで)
```

ポイントは`I(x - i, y - j)`という「引き算」になっている点。これは**「カーネルを180°回転（上下左右反転）してから、入力画像の対応する位置と掛けて足し合わせる」ことと数学的に同値**である（`i'=-i, j'=-j`と置換すると、和を取る範囲`[-k,k]`は符号を反転しても同じ集合になるため）。

## 畳み込み（Convolution）と相関（Correlation）の違い

多くの画像処理ライブラリ（OpenCVの`filter2D`など）が「畳み込み」と呼んでいる処理は、実はこの反転を行わない**相関（Correlation）**であることが多い。

```
(I ⋆ K)(x, y) = Σ K(i, j) * I(x + i, y + j)
```

対称なカーネル（平均フィルタ・ガウシアン等、`K(i,j) = K(-i,-j)`を満たすもの）では反転しても同じ結果になるため区別が問題にならないが、非対称なカーネル（[エッジ検出のSobelなど](../3-edge/README.md)）では畳み込みと相関で結果が変わる。この実装では数式に忠実に、カーネルを反転してから積和を取る「本物の畳み込み」を行う（実際の違いは[2-3. OpenCV比較](2-3-opencv-comparison.md)で検証する）。

## 畳み込み定理（周波数領域との関係）

畳み込みには周波数領域との重要な関係がある。

```
F[I * K] = F[I] * F[K]   (Fはフーリエ変換、右辺は要素ごとの積)
```

空間領域での畳み込みは、フーリエ変換した後の周波数領域では単純な「掛け算」になる。この関係は後の周波数領域（DFT/FFT）のトピックで重要になる（大きな画像・大きなカーネルの畳み込みをFFTで高速化できる理由）。

## 実装コード

```python
import numpy as np


def convolve2d(
    channel: np.ndarray, kernel: np.ndarray, *, quantize: bool = True
) -> np.ndarray:
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
            patch = padded[y : y + kernel_height, x : x + kernel_width]
            output[y, x] = np.sum(patch * flipped_kernel)

    if not quantize:
        return output  # 符号付き・範囲制限なしのfloat64をそのまま返す

    clipped = np.clip(output, 0, 255)
    return np.round(clipped).astype(np.uint8)
```

### 解説

1. **`flipped_kernel = kernel[::-1, ::-1]`**：行方向・列方向の両方を逆順にするスライスで、これが180°回転に相当する（畳み込み＝反転＋相関、の実装そのもの）。
2. **`np.pad(..., mode="constant", constant_values=0)`**：ゼロパディング（画像の外側は0とみなす）。出力サイズを入力と同じ`(H, W)`に保つため、上下左右を`kh//2, kw//2`だけゼロで埋める。
3. **`patch = padded[y:y+kh, x:x+kw]`**＋**`np.sum(patch * flipped_kernel)`**：出力の各画素`(y,x)`について、パディング後の画像からカーネルと同じ形の局所パッチを切り出し、カーネルとの要素ごとの積の総和を計算する。これがまさに畳み込みの定義式そのもの。forループは画素位置についてのみ回し、カーネルとの積和自体はNumPyでベクトル化している。
4. **`quantize`引数**：`True`（既定）なら0-255にクリップして`uint8`で返す。`False`なら丸め・クリップをせず符号付きの`float64`のまま返す。[エッジ検出のSobelフィルタ](../3-edge/README.md)のように結果が負の値を取りうる場合、`uint8`への変換で符号の情報が失われてしまうため、`quantize=False`で生の値を受け取れるようにしている。

## 関連

- [2-2. カーネルプリセット](2-2-kernel-presets.md) — このconvolve2dにいろいろなカーネルを渡すことで様々なフィルタになる
- [2-3. OpenCVとの比較](2-3-opencv-comparison.md) — 畳み込みと相関の違いを実データで検証
- [3. エッジ検出](../3-edge/README.md) — convolve2dをそのまま再利用（カーネルが変わるだけ）

---

前へ: [← 1-4. 閾値処理](../1-color/1-4-threshold.md)　｜　次へ: [2-2. カーネルプリセット →](2-2-kernel-presets.md)
