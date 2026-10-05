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

これは[畳み込みトピックで学んだ`mean_kernel`（均等平均）と`gaussian_kernel`（中心を重視した重み付き平均）](../2-convolution/2-2-kernel-presets.md)の違いが、そのままエッジ検出の平滑化方向にも表れている例である。教科書的には「Sobelの方が中心行/列を重視する分ノイズに強い」としばしば説明されるが、この点は実際に検証すると単純には言えない（下記「ノイズ耐性の検証」を参照）。

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

## ノイズ耐性の検証

「Sobelの方がノイズに強い」という通説を、実際にノイズを加えて検証してみる。縦エッジ画像（[3-1と同じもの](3-1-sobel.md)）に標準偏差`15`のガウスノイズを500回加え、`Gx`の応答のばらつきを測定する。

```
Sobel   Gx: 平均 -598.5, 標準偏差 50.6  → 相対ばらつき 8.45%
Prewitt Gx: 平均 -449.0, 標準偏差 36.1  → 相対ばらつき 8.04%
```

絶対値の標準偏差はSobelの方が大きいが、応答の大きさ自体もSobelの方が大きい（重みの合計が`4` vs `3`）ため、単純比較はできない。応答の大きさに対する相対的なばらつき（標準偏差／平均）で比べると、**ほぼ互角、むしろPrewittがわずかに低い**という結果になった。

数学的に見ても説明がつく。独立なノイズに対する平滑化カーネルの分散縮小率は`(重みの2乗和)/(重みの合計)²`で決まる。Sobelの平滑化`[1,2,1]`は`(1²+2²+1²)/(1+2+1)² = 6/16 = 0.375`、Prewittの`[1,1,1]`は`(1²+1²+1²)/(1+1+1)² = 3/9 = 0.333`——この指標ではむしろ**Prewittの方が小さく（良く）**なる。「中心重視の重みがガウシアンに近いので滑らかでノイズに強い」という直感的な説明は、単純な画素ノイズに対しては検証した限り裏付けられなかった。

## 3-1（Sobel）との使い分け

- **実務ではSobelが事実上の標準**：OpenCVをはじめ多くのライブラリでデフォルト採用されているのはSobelで、Prewittは比較のための実装として存在する、という位置づけが実態に近い。迷ったらSobelを使っておけば間違いない。
- **Prewittの存在意義は教材的な価値**：「微分×平滑化」という構造はそのままに、平滑化の重みだけを変えるとどうなるかを確認するための比較対象という性格が強い（[`mean_kernel`と`gaussian_kernel`の対比](../2-convolution/2-2-kernel-presets.md)と同じ構図）。
- **「ノイズ耐性」は決定的な判断材料にはならない**：上記の検証の通り、少なくとも単純な画素ノイズに対する相対的な頑健性に明確な差は確認できなかった。Sobelが好まれる理由は、ノイズ耐性そのものというより、連続的な勾配の近似としての滑らかさ（異方性の少なさ）や、単に広く使われているデファクトスタンダードであることが大きいと考えられる。

## 関連

- [3-1. Sobelエッジ検出](3-1-sobel.md) — 同じ構造、平滑化の重みだけが異なる
- [2-2. カーネルプリセット](../2-convolution/2-2-kernel-presets.md) — `mean_kernel`と`gaussian_kernel`の違いがここでも繰り返されている

---

前へ: [← 3-1. Sobelエッジ検出](3-1-sobel.md)　｜　次へ: [3-3. Laplacianエッジ検出 →](3-3-laplacian.md)
