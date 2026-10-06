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

## Sobel/Prewitt（1階微分）との使い分け

1階微分と2階微分は「どちらもエッジ検出に使える」が、**得られる情報の種類が異なる**。

- **Sobel/Prewitt（1階微分）**：`Gx`・`Gy`という2成分を持つ**ベクトル**。エッジの強さだけでなく、どちら向きの変化かという**方向**も分かる
- **Laplacian（2階微分）**：方向を持たない**スカラー**1つだけ。「変化があったか」は分かるが「どちら向きか」は失われている（回転させても同じ値になる、という意味で等方的）

「位置の特定精度」という1点だけで見ると、理想的でノイズの無い画像ではLaplacianの零交差の方がSobel/Prewittの「山のピーク」より鋭い（上記で確認した通り）。しかし**ノイズへの頑健性**で比較すると逆の結果になる。

同じ縦エッジ画像に標準偏差`15`のガウスノイズを500回加えて比較すると：

```
Sobel  |勾配|（エッジあり）: 平均601.0, 標準偏差50.3  → 相対ばらつき  8.4%
Laplacian（エッジあり）    : 平均-156.8, 標準偏差65.9 → 相対ばらつき 42.0%
```

Laplacianの方が相対的なばらつきが**5倍**大きい。さらに、エッジが全く無い平坦な画像にノイズだけを加えた場合の誤反応を比較すると：

```
Sobel の誤反応     : 平均66.3（本来の信号601.0に対して約11%）
Laplacian の誤反応 : 平均52.9（本来の信号156.8に対して約34%）
```

Laplacianは「エッジが無い場所でもノイズだけで誤って反応してしまう」割合がSobelよりずっと大きい。これは2階微分が1階微分よりノイズを強く増幅してしまう（微分を2回取ると、高周波成分＝ノイズがより強調される）という性質に由来する。

| 観点 | Sobel/Prewitt（1階微分） | Laplacian（2階微分） |
|---|---|---|
| エッジの向き | 分かる | 分からない |
| エッジ位置の特定精度（理想的な場合） | ピークに幅がある | ゼロ交差で鋭い |
| ノイズへの頑健性 | 比較的強い | 弱い（誤反応しやすい） |

実務では、素のLaplacianをそのまま使うのではなく、先にガウシアンでぼかしてノイズを抑えてから使う「LoG（Laplacian of Gaussian）」という手法がよく使われる（このプロジェクトでは未実装）。「Laplacianの方が精度が高い」というのは、位置特定精度という1つの軸だけの話であり、ノイズ耐性まで含めた総合的な優劣ではない、という点に注意が必要。

## 関連

- [2-2. カーネルプリセット](../2-convolution/2-2-kernel-presets.md) — `laplacian_kernel`の定義元。1階微分との比較もここにある
- [3-1. Sobelエッジ検出](3-1-sobel.md) — 対比される1階微分アプローチ
- [laplacian-kernel：コード⇔数式対応](../../math/laplacian-kernel.md) — カーネルの各マスの値の導出
- [なぜ1階ではなく2階か](../2-convolution/2-2-kernel-presets.md#なぜ2階微分を使うのかゼロ交差によるエッジの検出) — 零交差によるエッジ位置の特定を実測値付きで比較
- [複数のエッジが近くにある場合](../2-convolution/2-2-kernel-presets.md#複数のエッジが近くにある場合) — エッジ同士が近いとゼロ交差が干渉することを実測値で確認

---

前へ: [← 3-2. Prewittエッジ検出](3-2-prewitt.md)　｜　次へ: [4-1. Erosion（収縮） →](../4-morphology/4-1-erosion.md)
