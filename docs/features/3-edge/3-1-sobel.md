# 3-1. Sobelエッジ検出

[← 3. エッジ検出へ戻る](README.md)　｜　**3-1** · [3-2](3-2-prewitt.md) · [3-3](3-3-laplacian.md)

| | |
|---|---|
| 学習テーマ | 有限差分近似、勾配ベクトル、ユークリッドノルム |
| 実装ファイル | [`engine/src/imglab_engine/edge/sobel.py`](https://github.com/oguratakayuki/image-processing-lab/blob/main/engine/src/imglab_engine/edge/sobel.py)<br>[`engine/src/imglab_engine/edge/gradient.py`](https://github.com/oguratakayuki/image-processing-lab/blob/main/engine/src/imglab_engine/edge/gradient.py) |
| 関数 | `sobel_gradient()`, `gradient_magnitude()` |
| テスト | [`engine/tests/edge/test_sobel.py`](https://github.com/oguratakayuki/image-processing-lab/blob/main/engine/tests/edge/test_sobel.py)<br>[`engine/tests/edge/test_gradient.py`](https://github.com/oguratakayuki/image-processing-lab/blob/main/engine/tests/edge/test_gradient.py) |

## 概念

エッジ（輪郭）とは「画像の明るさが急激に変化する場所」であり、数学的には輝度関数`I(x,y)`の**変化率＝微分**が大きい場所として定式化できる。

## 数学的背景

### 微分から勾配ベクトルへ

1変数関数の導関数の定義：

```
f'(x) = lim(h→0) (f(x+h) - f(x)) / h
```

画像は離散的な格子上の値なので極限は取れず、**有限差分（finite difference）**で近似する。中心差分を使うと：

```
f'(x) ≈ (f(x+1) - f(x-1)) / 2
```

画像`I(x,y)`は2変数関数なので、x方向・y方向それぞれの偏微分を考える：

```
∂I/∂x ≈ I(x+1,y) - I(x-1,y)
∂I/∂y ≈ I(x,y+1) - I(x,y-1)
```

この2つの偏微分をまとめたベクトルが**勾配（gradient）**である：

```
∇I = (∂I/∂x, ∂I/∂y)
```

勾配ベクトルは「輝度が最も急激に増加する方向」を指すベクトルであり、その大きさ（**ユークリッドノルム**）

```
|∇I| = √((∂I/∂x)² + (∂I/∂y)²)
```

が「その場所がどれだけエッジらしいか」を表す。線形代数で習う「ベクトルの大きさ（2乗和の平方根）」がそのまま画像処理に登場する例になっている。

### Sobelカーネルの構成

単純な差分`I(x+1,y) - I(x-1,y)`だけでなく、微分方向に直交する方向へ`[1,2,1]`という重みで平滑化（簡易的なノイズ除去）を組み合わせた、次の3×3カーネルを使う。

```
Gx = |-1  0  1|     Gy = |-1 -2 -1|
     |-2  0  2|          | 0  0  0|
     |-1  0  1|          | 1  2  1|
```

`Gx`は横方向に`[-1,0,1]`（差分＝微分）、縦方向に`[1,2,1]`（平滑化）をかけ合わせた形になっている（実際`Gx`は縦ベクトル`[1,2,1]`と横ベクトル`[-1,0,1]`の外積に一致する。[カーネルの分離可能性](../../math/gaussian-2d-function.md)のもう1つの例）。`Gy`は`Gx`を転置した形で、微分と平滑化の方向が入れ替わっている。

`Gx`, `Gy`はどちらも[非対称なカーネル](../2-convolution/2-1-convolution.md)（`Gx(i,j) ≠ Gx(-i,-j)`）である。[畳み込みトピックで学んだ通り](../2-convolution/2-1-convolution.md)、非対称カーネルでは畳み込み（反転あり）と相関（反転なし）で符号や向きが変わる。ここでは`convolve2d`による「本物の畳み込み」を使う。

## 実装コード

```python
import numpy as np
from imglab_engine.convolution.convolution import convolve2d

SOBEL_X = np.array(
    [
        [-1.0, 0.0, 1.0],
        [-2.0, 0.0, 2.0],
        [-1.0, 0.0, 1.0],
    ]
)

SOBEL_Y = np.array(
    [
        [-1.0, -2.0, -1.0],
        [0.0, 0.0, 0.0],
        [1.0, 2.0, 1.0],
    ]
)


def sobel_gradient(channel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # channel: (H, W)のGrayscale画像(uint8, 値域0-255)。
    # 返却値: (Gx, Gy)のタプル。どちらも(H, W)の符号付きfloat64配列。
    gx = convolve2d(channel, SOBEL_X, quantize=False)
    gy = convolve2d(channel, SOBEL_Y, quantize=False)
    return gx, gy
```

```python
def gradient_magnitude(gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    # gx, gy: 勾配のx成分・y成分。どちらも(H, W)の符号付きfloat64配列。
    # 返却値: 勾配の大きさ|∇I|を0-255に正規化した(H, W)のuint8配列。
    magnitude = np.sqrt(gx**2 + gy**2)
    max_value = magnitude.max()
    if max_value == 0:
        return magnitude.astype(np.uint8)
    normalized = magnitude / max_value * 255
    return np.round(normalized).astype(np.uint8)
```

### 解説

1. **`quantize=False`で`convolve2d`を呼ぶ**：勾配は符号付きの値（輝度が減少する方向のエッジでは負になる）を取る。[1-1以来使ってきた`uint8`への量子化](../1-color/1-1-grayscale.md)は0-255にクリップしてしまい符号の情報が失われるため、ここでは生の`float64`のまま受け取る。
2. **`gradient_magnitude`**：`np.sqrt(gx**2 + gy**2)`が数式`|∇I| = √(Gx²+Gy²)`そのもの。大きさは常に非負なので、画像として表示するために「この画像の中での最大値」で0-255に正規化する（[ヒストグラムの正規化](../1-color/1-3-histogram.md)と同じ考え方）。

## 具体的な計算例

縦方向のエッジ画像（左が暗く`50`、右が明るく`200`）を使う。

```
I = | 50  50  200  200  200 |   （全ての行が同じパターン）
```

境界（パディング）の影響を受けない内側の行で計算すると：

```
Gx = | -200  -600  -600    0   800 |
Gy = |    0     0     0    0     0 |
```

縦方向には変化がないので`Gy`はちょうど`0`。横方向の境界（列1と列2の間）付近で`Gx`が大きな値を持つ。これは[`test_sobel.py`](https://github.com/oguratakayuki/image-processing-lab/blob/main/engine/tests/edge/test_sobel.py)の`test_vertical_edge_is_detected_by_gx_not_gy`で検証している値と一致する。

`gradient_magnitude`は`Gy=0`なので`|∇I| = |Gx|`になり、画像全体（境界行も含む）の最大値`848.53`で正規化すると：

```
|Gx| (正規化前):  200   600   600     0   800
magnitude (正規化後): 60   180   180     0   240
```

正規化の基準値`848.53`が`800`より大きいのは、**画像の上下端（ゼロパディングに接する行）で`Gy`が0でない値を持つため**。縦エッジ画像なのに上下端だけ余分な勾配が出るのは、ゼロパディングという境界処理そのものが「画像の外側に人工的なエッジ（0との境目）」を作ってしまうことによる（境界処理に起因するアーティファクト）。

## 関連

- [2-1. 畳み込み](../2-convolution/2-1-convolution.md) — `convolve2d`をそのまま再利用（カーネルが変わるだけ）
- [3-2. Prewittエッジ検出](3-2-prewitt.md) — 平滑化の重みだけを変えた兄弟アルゴリズム
- [3-3. Laplacianエッジ検出](3-3-laplacian.md) — 1階微分（勾配）と対比される2階微分アプローチ

---

前へ: [← 2-3. OpenCVとの比較](../2-convolution/2-3-opencv-comparison.md)　｜　次へ: [3-2. Prewittエッジ検出 →](3-2-prewitt.md)
