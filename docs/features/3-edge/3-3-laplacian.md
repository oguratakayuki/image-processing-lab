# 3-3. Laplacianエッジ検出

[← 3. エッジ検出へ戻る](README.md)　｜　[3-1](3-1-sobel.md) · [3-2](3-2-prewitt.md) · **3-3**

| | |
|---|---|
| 学習テーマ | 2階微分、零交差 |
| 実装ファイル | [`engine/src/imglab_engine/edge/laplacian.py`](https://github.com/oguratakayuki/image-processing-lab/blob/main/engine/src/imglab_engine/edge/laplacian.py) |
| 関数 | `laplacian_edge_response()` |
| テスト | [`engine/tests/edge/test_laplacian.py`](https://github.com/oguratakayuki/image-processing-lab/blob/main/engine/tests/edge/test_laplacian.py) |

## 概念

[3-1のSobel](3-1-sobel.md)・[3-2のPrewitt](3-2-prewitt.md)は1階微分（勾配）の大きさでエッジを検出したが、Laplacianは**2階微分**を使う、質の異なるアプローチである。

## 数学的背景

```
∇²I = ∂²I/∂x² + ∂²I/∂y²
```

1階微分は「エッジの位置で値が大きくなる（山型）」性質を使っていたが、2階微分は「エッジをまたいで符号が反転する」性質を使う。理想的な階段状のエッジでは、2階微分はエッジの手前で正、直後で負（あるいはその逆）になり、エッジのちょうど中心で`0`を横切る（これを**零交差, zero crossing**と呼ぶ）。1階微分のピーク位置を探すより、2階微分の符号反転位置を探す方が、エッジの位置をピクセル単位で正確に特定しやすいという利点がある（詳しくは[2-2のカーネルプリセット](../2-convolution/2-2-kernel-presets.md)の「なぜ2階微分を使うのか」で実測値付きで比較している）。

このモジュールでは、[畳み込みトピックで既に実装済みの`laplacian_kernel()`](../2-convolution/2-2-kernel-presets.md)をそのまま再利用する。新しいカーネルを追加する必要はなく、「同じ畳み込みという操作に、異なるカーネルを渡すだけで別のアルゴリズムになる」ことを改めて示す例になっている。

## 実装コード

```python
import numpy as np
from imglab_engine.convolution.convolution import convolve2d
from imglab_engine.convolution.kernels import laplacian_kernel


def laplacian_edge_response(channel: np.ndarray) -> np.ndarray:
    return convolve2d(channel, laplacian_kernel(), quantize=False)
```

### 解説

たった1行。`laplacian_kernel()`（[2-2で定義済み、コード⇔数式対応は専用資料を参照](../../math/laplacian-kernel.md)）を`convolve2d`に渡しているだけ。Sobel/Prewittの勾配成分と同様、符号（零交差の位置）が重要な意味を持つため、`quantize=False`で符号付きの`float64`のまま返す。

## 具体的な計算例

階段状のエッジ画像（列2までが`20`、列3以降が`200`）を使うと：

```
入力:               20   20   20  200  200
Laplacian応答:        0    0  100 -100    0
```

境界（列2と列3の間）のちょうど両側で`+100`と`-100`に符号が反転しており、その**切り替わる場所がエッジの実際の位置と一致**する。これが「零交差」の具体例（詳細な計算過程・Sobelとの比較は[2-2のカーネルプリセット](../2-convolution/2-2-kernel-presets.md)、各マスの値の導出は[laplacian-kernel：コード⇔数式対応](../../math/laplacian-kernel.md)を参照）。

## 関連

- [2-2. カーネルプリセット](../2-convolution/2-2-kernel-presets.md) — `laplacian_kernel`の定義元。1階微分との比較もここにある
- [3-1. Sobelエッジ検出](3-1-sobel.md) — 対比される1階微分アプローチ
- [laplacian-kernel：コード⇔数式対応](../../math/laplacian-kernel.md) — カーネルの各マスの値の導出

---

前へ: [← 3-2. Prewittエッジ検出](3-2-prewitt.md)　｜　次へ: [4. 二値画像処理 →](../4-morphology/README.md)（詳細ページは未整備）
