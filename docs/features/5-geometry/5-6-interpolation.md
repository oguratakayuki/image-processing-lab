# 5-6. バイリニア補間（Bilinear Interpolation）

[← 5. 幾何学的変換へ戻る](README.md)　｜　[5-1](5-1-translation.md) · [5-2](5-2-scaling.md) · [5-3](5-3-rotation.md) · [5-4](5-4-shear.md) · [5-5](5-5-affine.md) · **5-6**

| | |
|---|---|
| 学習テーマ | 線形補間、バイリニア補間 |
| 実装ファイル | [`engine/src/imglab_engine/geometry/interpolation.py`](https://github.com/oguratakayuki/image-processing-lab/blob/main/engine/src/imglab_engine/geometry/interpolation.py) |
| 関数 | `nearest_neighbor_sample()`, `bilinear_sample()` |
| テスト | [`engine/tests/geometry/test_interpolation.py`](https://github.com/oguratakayuki/image-processing-lab/blob/main/engine/tests/geometry/test_interpolation.py)、[`engine/tests/geometry/test_affine.py`](https://github.com/oguratakayuki/image-processing-lab/blob/main/engine/tests/geometry/test_affine.py) |

## 超ざっくり言うと

[5-2〜5-5](README.md)で使ってきた「最も近い1点をそのまま使う」最近傍補間の代わりに、「周囲4点を混ぜ合わせる」バイリニア補間を使うと、拡大・回転の結果がより滑らかになる。

## 概念

[5-2](5-2-scaling.md)以降、逆方向マッピングで逆算した入力側の座標は一般に非整数になり、「その非整数座標の値をどう決めるか」という補間（interpolation）の問題が常について回っていた。これまでは最も単純な**最近傍補間**（nearest neighbor）を使ってきたが、ここでは周囲4点の**距離に応じた加重平均**を取る**バイリニア補間**を導入する。

## 主な用途

- **画質の向上**：拡大・回転した画像のブロック状のギザギザ（最近傍補間特有のアーティファクト）を滑らかにする
- **[5-3の回転](5-3-rotation.md)で見つかった限界の解消**：孤立した1ピクセルの点が45度回転で完全に消えてしまう問題が、バイリニア補間では起きない（後述）
- 多くの画像処理ライブラリ（OpenCVの`cv2.resize`や`cv2.warpAffine`のデフォルトなど）で標準的に使われている補間方法

## 数学的背景

### 要件

逆算した非整数座標`(x, y)`の画素値を、最近傍補間より滑らかな方法で決めたい。

### やるべきこと

非整数座標`(x, y)`を取り囲む4つの整数格子点を考える。

```
x0 = floor(x),  x1 = x0 + 1
y0 = floor(y),  y1 = y0 + 1
```

この4点`(x0,y0), (x1,y0), (x0,y1), (x1,y1)`の値を、距離に応じた重みで混ぜ合わせる。まず`x`方向に、上の2点・下の2点それぞれを1次元の線形補間（端数`fx = x - x0`を使った加重平均）で合成する。

```
top    = (1-fx)・I(x0,y0) + fx・I(x1,y0)
bottom = (1-fx)・I(x0,y1) + fx・I(x1,y1)
```

続けて`y`方向に、`top`と`bottom`を同じく線形補間する（端数`fy = y - y0`）。

```
value = (1-fy)・top + fy・bottom
```

「`x`方向の線形補間を2回→その結果を`y`方向に1回線形補間」という、3回の1次元線形補間の組み合わせになっている（これが"bi"linear＝「2方向の」線形補間、という名前の由来）。最近傍補間のような値の急な切り替わりが無く、滑らかな結果になる。

補間に使う4点のうち、画像の外にあるものは[5-1の平行移動](5-1-translation.md)などと同じく0（黒）として扱う。

### コードにすると

`math.floor()`で4つの整数格子点を求め、`x`方向の線形補間を2回（`top`, `bottom`）、続けて`y`方向の線形補間を1回行う——これがそのまま後述の`bilinear_sample()`の実装になる。[5-5のアフィン変換](5-5-affine.md)の`affine_transform()`に`interpolation="bilinear"`を渡すと、これまでの最近傍補間の代わりにこの関数が使われる。

## 実装コード

```python
import math

import numpy as np


def _pixel_or_zero(image: np.ndarray, y: int, x: int):
    # image: (H, W)または(H, W, 3)のuint8配列。
    # y, x: 整数の格子点座標。
    # 返却値: float64の画素値。(H, W)画像ならスカラー、(H, W, 3)画像なら長さ3の配列。
    height, width = image.shape[:2]
    if 0 <= y < height and 0 <= x < width:
        # 範囲内: その画素の値をfloat64に変換して返す。
        return image[y, x].astype(np.float64)
    if image.ndim == 2:
        # 範囲外 かつ Grayscale画像(2次元): スカラーの0を返す。
        return np.float64(0.0)
    # 範囲外 かつ RGB画像(3次元): 範囲内の場合(image[y,x]が長さ3の
    # 配列になる)と形を揃えるため、チャンネル数ぶんのゼロ配列を返す
    # (後続の加重平均の計算で、配列の形が食い違わないようにするため)。
    return np.zeros(image.shape[2], dtype=np.float64)


def nearest_neighbor_sample(image: np.ndarray, y: float, x: float):
    # image: (H, W)または(H, W, 3)のuint8配列。
    # y, x: サンプリングしたい座標(非整数でもよい)。
    # 返却値: float64の画素値(範囲外なら0)。
    return _pixel_or_zero(image, round(y), round(x))


def bilinear_sample(image: np.ndarray, y: float, x: float):
    # image: (H, W)または(H, W, 3)のuint8配列。
    # y, x: サンプリングしたい座標(非整数でもよい)。
    # 返却値: float64の画素値(4点のうち範囲外のものは0として扱う)。
    y0 = math.floor(y)
    x0 = math.floor(x)
    y1 = y0 + 1
    x1 = x0 + 1

    fy = y - y0
    fx = x - x0

    top = _pixel_or_zero(image, y0, x0) * (1 - fx) + _pixel_or_zero(image, y0, x1) * fx
    bottom = (
        _pixel_or_zero(image, y1, x0) * (1 - fx) + _pixel_or_zero(image, y1, x1) * fx
    )

    return top * (1 - fy) + bottom * fy
```

### 解説

1. **`_pixel_or_zero(image, y, x)`**：座標`(y, x)`が画像の範囲内ならその画素値を、範囲外なら0を返す共通ヘルパー。[5-1の範囲内判定](5-1-translation.md)と同じ考え方を、値を返す形に書き換えたもの。
2. **`nearest_neighbor_sample`**：`round()`で最も近い整数座標に丸めてから、`_pixel_or_zero`を1回呼ぶだけ。これまでの`affine_transform()`に直接書かれていた最近傍補間のロジックを、独立した関数として切り出したもの。
3. **`math.floor(y)`**：小数点以下を切り捨てる。`round()`（四捨五入）とは異なる丸め方で、負の数でも「より小さい方の整数」に丸められる（例：`floor(-0.5) = -1`）。
4. **`top = ... * (1 - fx) + ... * fx`**：`x`方向の線形補間。`fx=0`なら`x0`の値そのまま、`fx=1`なら`x1`の値そのまま、その間は比例配分になる。
5. **`return top * (1 - fy) + bottom * fy`**：`y`方向の線形補間。`top`・`bottom`という「`x`方向にすでに補間済みの2つの値」を、さらに`y`方向に補間する。

## 具体的な計算例

`0 → 255`の垂直エッジ（2×2画像、左列が0・右列が255）を横切るように、`x`座標を`0.3, 0.5, 0.7`とサンプリングして比較する。

```
入力:
  0 255
  0 255

x座標:          0.3     0.5     0.7
nearest:        0.0     0.0   255.0
bilinear:      76.5   127.5   178.5
```

最近傍補間は`x=0.5`から`x=0.7`の間で`0`から`255`へ**一瞬で切り替わる**が、バイリニア補間は`76.5 → 127.5 → 178.5`と**連続的に変化**している。[`test_interpolation.py`](https://github.com/oguratakayuki/image-processing-lab/blob/main/engine/tests/geometry/test_interpolation.py)の`test_bilinear_smooths_a_hard_edge`で検証している値と一致する。

**[5-3で見つかった「孤立した1点が消える」問題の解消**も確認する。5×5画像で上端中央`(0,2)`の点だけが前景の画像を、画像の中心を軸に45度回転すると：

```
入力:
  0   0 255   0   0
  0   0   0   0   0
  0   0   0   0   0
  0   0   0   0   0
  0   0   0   0   0

最近傍補間（5-3で見た通り、点が消える）:
  0   0   0   0   0
  0   0   0   0   0
  0   0   0   0   0
  0   0   0   0   0
  0   0   0   0   0

バイリニア補間（値が周辺に分散して残る）:
  0   0   0  66  44
  0   0   0 106  66
  0   0   0   0   0
  0   0   0   0   0
  0   0   0   0   0
```

最近傍補間では出力が全て`0`になり点が完全に消えてしまうが、バイリニア補間では値が複数のピクセルに分散しながらも残っている（[`test_affine.py`](https://github.com/oguratakayuki/image-processing-lab/blob/main/engine/tests/geometry/test_affine.py)の`test_bilinear_interpolation_rescues_the_vanishing_point`で検証）。[5-3のドキュメント](5-3-rotation.md)で「実装のバグではなく、最近傍補間そのものの限界」と説明した問題が、より高度な補間方法で実際に解消されることを確認できた。

## 関連

- [5-3. 回転](5-3-rotation.md) — 「孤立した1点が消える」という、最近傍補間の限界が最初に見つかった項目
- [5-5. 同次座標によるアフィン変換の統一](5-5-affine.md) — `affine_transform()`の`interpolation`引数からこのモジュールを呼び出す

---

前へ: [← 5-5. 同次座標によるアフィン変換の統一](5-5-affine.md)　｜　次へ: なし（現時点で最後の実装済み項目）
