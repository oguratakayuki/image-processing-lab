# 3-2. Prewittエッジ検出

[← 3. エッジ検出へ戻る](README.md)　｜　[3-1](3-1-sobel.md) · **3-2** · [3-3](3-3-laplacian.md)

| | |
|---|---|
| 学習テーマ | Sobelと同構造（平滑化重みの違い） |
| 実装ファイル | [`engine/src/imglab_engine/edge/prewitt.py`](https://github.com/oguratakayuki/image-processing-lab/blob/main/engine/src/imglab_engine/edge/prewitt.py) |
| 関数 | `prewitt_gradient()` |
| テスト | [`engine/tests/edge/test_prewitt.py`](https://github.com/oguratakayuki/image-processing-lab/blob/main/engine/tests/edge/test_prewitt.py) |

## 概念

考え方は[3-1のSobel](3-1-sobel.md)と全く同じ（微分方向の差分 × 直交方向の平滑化）。違いは平滑化に使う重みだけ。

## 数学的背景

Sobelは`[1,2,1]`（2項係数、離散的なガウシアンの近似）で中心を重視した平滑化をしたが、Prewittは単純な均等平均`[1,1,1]`を使う。

```
Prewitt_X = |-1  0  1|     Prewitt_Y = |-1 -1 -1|
            |-1  0  1|                 | 0  0  0|
            |-1  0  1|                 | 1  1  1|
```

これは[畳み込みトピックで学んだ`mean_kernel`（均等平均）と`gaussian_kernel`（中心を重視した重み付き平均）](../2-convolution/2-2-kernel-presets.md)の違いが、そのままエッジ検出の平滑化方向にも表れている例である。Sobelの方が中心行/列の影響を強く受ける分、ノイズに強いとされる（Prewittは全ての行が同じ重みなので、外側の行のノイズがSobelより結果に影響しやすい）。

## 実装コード

```python
import numpy as np
from imglab_engine.convolution.convolution import convolve2d

PREWITT_X = np.array(
    [
        [-1.0, 0.0, 1.0],
        [-1.0, 0.0, 1.0],
        [-1.0, 0.0, 1.0],
    ]
)

PREWITT_Y = np.array(
    [
        [-1.0, -1.0, -1.0],
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
    ]
)


def prewitt_gradient(channel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gx = convolve2d(channel, PREWITT_X, quantize=False)
    gy = convolve2d(channel, PREWITT_Y, quantize=False)
    return gx, gy
```

### 解説

構造は[`sobel_gradient()`](3-1-sobel.md)と全く同じで、カーネルの中身（平滑化の重み）だけが違う。`quantize=False`で符号付きのまま返す理由も同じ（[3-1の解説](3-1-sobel.md)を参照）。勾配の大きさを求める`gradient_magnitude()`も[3-1で説明したもの](3-1-sobel.md)をそのまま共用する（Sobel専用だったものをPrewittでも使える共通処理として`edge/gradient.py`に切り出してある）。

## 具体的な計算例

[3-1と同じ縦エッジ画像](3-1-sobel.md)（左`50`・右`200`）で計算すると、内側の行は：

```
Prewitt_Gx = | -150  -450  -450    0   600 |
Prewitt_Gy = |    0     0     0    0     0 |
```

[Sobelの結果](3-1-sobel.md)（`-200, -600, -600, 0, 800`）と比べると、同じ位置関係（境界付近で大きな値、符号も同じ）だが、**絶対値が一回り小さい**。これは平滑化の重みの合計がSobelの`1+2+1=4`に対してPrewittは`1+1+1=3`と小さいことに対応する。[`test_prewitt.py`](https://github.com/oguratakayuki/image-processing-lab/blob/main/engine/tests/edge/test_prewitt.py)で検証している値と一致する。

## 関連

- [3-1. Sobelエッジ検出](3-1-sobel.md) — 同じ構造、平滑化の重みだけが異なる
- [2-2. カーネルプリセット](../2-convolution/2-2-kernel-presets.md) — `mean_kernel`と`gaussian_kernel`の違いがここでも繰り返されている

---

前へ: [← 3-1. Sobelエッジ検出](3-1-sobel.md)　｜　次へ: [3-3. Laplacianエッジ検出 →](3-3-laplacian.md)
