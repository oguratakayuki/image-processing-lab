# 5-5. 同次座標によるアフィン変換の統一

[← 5. 幾何学的変換へ戻る](README.md)　｜　[5-1](5-1-translation.md) · [5-2](5-2-scaling.md) · [5-3](5-3-rotation.md) · [5-4](5-4-shear.md) · **5-5**

| | |
|---|---|
| 学習テーマ | 同次座標（homogeneous coordinates）、アフィン変換行列、変換の合成 |
| 実装ファイル | [`engine/src/imglab_engine/geometry/affine.py`](https://github.com/oguratakayuki/image-processing-lab/blob/main/engine/src/imglab_engine/geometry/affine.py) |
| 関数 | `affine_transform()`, `translation_matrix()`, `scaling_matrix()`, `rotation_matrix()`, `shear_matrix()` |
| テスト | [`engine/tests/geometry/test_affine.py`](https://github.com/oguratakayuki/image-processing-lab/blob/main/engine/tests/geometry/test_affine.py) |

## 超ざっくり言うと

[5-1〜5-4](README.md)でバラバラに実装した4つの変換（平行移動・拡大縮小・回転・せん断）を、全部まとめて「1つの3×3行列」として表現する方法。複数の変換を組み合わせたいときは、行列同士を掛け算するだけでよくなる。

## 概念

[5-1の平行移動](5-1-translation.md)は、2×2の線形変換行列だけでは表現できなかった（線形写像は必ず原点を原点に写すが、平行移動は原点を動かしてしまう）。この制約を解消するのが**同次座標（homogeneous coordinates）**——2次元の座標`(x, y)`を、末尾に`1`を付け足した3次元のベクトル`(x, y, 1)`として表す技法。これにより、平行移動も3×3行列の掛け算として表現でき、[5-2](5-2-scaling.md)・[5-3](5-3-rotation.md)・[5-4](5-4-shear.md)の2×2変換行列と、平行移動を**1つの3×3行列**にまとめられる。

## 主な用途

- **複数の変換の合成**：「拡大してから回転する」のような処理を、行列の掛け算1回にまとめて、1回のループで画像を変換できる（本来は変換ごとに画像を1回ずつ書き換える必要があるところを、先に行列だけ合成しておける）
- **グラフィックスAPIの基礎**：OpenGL・CSSの`transform: matrix()`など、2D/3Dグラフィックス全般がこの3×3（または3Dなら4×4）のアフィン変換行列の仕組みを使っている
- **[5-3の回転](5-3-rotation.md)の仕組みの一般化**：「画像の中心を軸に回転する」という処理も、平行移動と回転の合成として説明し直せる

## 数学的背景

### 要件

[5-1〜5-4](README.md)の4つの変換を、バラバラの実装のままではなく、1つの統一的な枠組みで表現したい。また、複数の変換を順番に適用する処理を、効率よく1つの変換にまとめたい。

### やるべきこと

2次元の点`(x, y)`を、同次座標`(x, y, 1)`として表す。すると、平行移動も含めた全ての変換が、3×3行列の掛け算として書ける。

```
⎡x'⎤   ⎡a  b  tx⎤ ⎡x⎤
⎢y'⎥ = ⎢c  d  ty⎥ ⎢y⎥
⎣1 ⎦   ⎣0  0  1 ⎦ ⎣1⎦
```

1行目・2行目を展開すると`x' = a・x + b・y + tx`、`y' = c・x + d・y + ty`——「[5-2〜5-4の2×2線形変換](README.md)（`a,b,c,d`）」と「[5-1の平行移動](5-1-translation.md)（`tx,ty`）」が、同時に1つの式で表現されている。3行目は`0・x + 0・y + 1・1 = 1`で、同次座標の末尾が常に`1`のまま保たれることを保証しているだけ。

**変換の合成＝行列の掛け算**。「拡大してから回転する」処理は、それぞれの行列を掛け合わせた1つの行列として表現できる。

```
M_combined = M_rotate ・ M_scale
```

`M_combined`を1回適用するのは、「先に`M_scale`を適用し、その結果に`M_rotate`を適用する」のと同じ結果になる（行列の掛け算は右から順に効いていく）。[5-3の回転](5-3-rotation.md)は「画像の中心を軸に回転する」ために、コードの中で「中心を引く→回転→中心を足す」という3段階の計算を直接書いていたが、これは`M = T(center)・R(theta)・T(-center)`という3つの行列の積として表現し直せる（実際に一致することをテストで検証済み）。

**掛ける順序を変えると結果が変わる**（行列の積は交換法則が成り立たない）。ただし**例外が1つある**：`sx = sy`の均一な拡大縮小は、回転と完全に可換（順序を入れ替えても行列が一致する）になる。均一な拡大縮小はスカラー倍（単位行列の定数倍）であり、スカラー倍はどんな行列とも可換、という線形代数の性質から来ている。`sx ≠ sy`の非均一な拡大縮小なら、回転との非可換性が実際に現れる（実測で確認済み、後述）。

逆方向マッピングで使う逆行列も、2×2の線形部分の逆行列（[5-4で学んだ行列式を使う一般公式](5-4-shear.md)）と、平行移動部分を使って求められる。

### コードにすると

3×3行列の左上2×2（線形部分）と右上2×1（平行移動部分）を取り出し、線形部分の逆行列（行列式を使う一般公式）を計算、平行移動部分の逆変換`-M⁻¹・t`を求める。あとは[5-1〜5-4と同じ逆方向マッピング](5-1-translation-inverse-mapping.md)の骨格（出力の各点について逆算、最近傍補間、範囲外なら黒）をそのまま使う——これがそのまま後述の`affine_transform()`の実装になる。

## 実装コード

```python
import math

import numpy as np


def translation_matrix(tx: float, ty: float) -> np.ndarray:
    # tx: x方向の移動量。ty: y方向の移動量。
    # 返却値: 3x3のfloat64配列。
    return np.array(
        [
            [1.0, 0.0, tx],
            [0.0, 1.0, ty],
            [0.0, 0.0, 1.0],
        ]
    )


def scaling_matrix(sx: float, sy: float) -> np.ndarray:
    # sx: x方向の拡大率。sy: y方向の拡大率。
    # 返却値: 3x3のfloat64配列。
    return np.array(
        [
            [sx, 0.0, 0.0],
            [0.0, sy, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def rotation_matrix(degrees: float) -> np.ndarray:
    # degrees: 回転角度(度数法)。正の値で時計回り。
    # 返却値: 3x3のfloat64配列。
    theta = math.radians(degrees)
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    return np.array(
        [
            [cos_theta, -sin_theta, 0.0],
            [sin_theta, cos_theta, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def shear_matrix(shx: float, shy: float) -> np.ndarray:
    # shx: 横方向のせん断係数。shy: 縦方向のせん断係数。
    # 返却値: 3x3のfloat64配列。
    return np.array(
        [
            [1.0, shx, 0.0],
            [shy, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def affine_transform(image: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    # image: 変換する画像。(H, W)または(H, W, 3)のuint8配列。
    # matrix: 3x3のアフィン変換行列(3行目は[0,0,1])。
    # 返却値: 変換後の画像。入力と同じ形状・dtype。
    height, width = image.shape[:2]

    a, b, tx = matrix[0, 0], matrix[0, 1], matrix[0, 2]
    c, d, ty = matrix[1, 0], matrix[1, 1], matrix[1, 2]

    det = a * d - b * c

    inv_a, inv_b = d / det, -b / det
    inv_c, inv_d = -c / det, a / det

    inv_tx = -(inv_a * tx + inv_b * ty)
    inv_ty = -(inv_c * tx + inv_d * ty)

    output = np.zeros_like(image)

    for out_y in range(height):
        for out_x in range(width):
            src_x = inv_a * out_x + inv_b * out_y + inv_tx
            src_y = inv_c * out_x + inv_d * out_y + inv_ty

            nearest_y = round(src_y)
            nearest_x = round(src_x)

            # 入力座標は実際に画像の中に存在するかどうか。
            if 0 <= nearest_y < height and 0 <= nearest_x < width:
                output[out_y, out_x] = image[nearest_y, nearest_x]

    return output
```

### 解説

1. **`translation_matrix`/`scaling_matrix`/`rotation_matrix`/`shear_matrix`**：[5-1〜5-4](README.md)それぞれの2×2変換行列を、平行移動部分が0の3×3行列に拡張したもの。`translation_matrix`だけは2×2部分が恒等行列（`[[1,0],[0,1]]`）で、`tx, ty`の列だけが意味を持つ。
2. **`a, b, tx = matrix[0, 0], matrix[0, 1], matrix[0, 2]`**：NumPyの2次元配列のインデックス`matrix[行, 列]`で、3×3行列から線形部分（`a,b,c,d`）と平行移動部分（`tx,ty`）を取り出す。
3. **`det = a * d - b * c`**：[5-4と同じ、2×2線形部分の行列式](5-4-shear.md)。
4. **`inv_a, inv_b = d / det, -b / det`**：[5-4と同じ逆行列の公式](5-4-shear.md)で、線形部分だけの逆行列を求める。
5. **`inv_tx = -(inv_a * tx + inv_b * ty)`**：平行移動部分の逆変換`-M⁻¹・t`。線形部分の逆行列`M⁻¹`を、平行移動ベクトル`(tx,ty)`に適用してから符号を反転する。
6. それ以降（`for out_y...`から先）は、[5-1〜5-4と全く同じ逆方向マッピングの骨格](5-1-translation-inverse-mapping.md)。

## 具体的な計算例

[5-1の平行移動`translate()`](5-1-translation.md)と、`affine_transform(image, translation_matrix(tx, ty))`が完全に一致することを確認する。5×5画像で`(1,1)`だけが前景の画像を、`tx=2, ty=1`で変換すると：

```
入力:
  0   0   0   0   0
  0 255   0   0   0
  0   0   0   0   0
  0   0   0   0   0
  0   0   0   0   0

translate(tx=2, ty=1):              affine_transform(translation_matrix(2,1)):
  0   0   0   0   0                   0   0   0   0   0
  0   0   0   0   0                   0   0   0   0   0
  0   0   0 255   0                   0   0   0 255   0
  0   0   0   0   0                   0   0   0   0   0
  0   0   0   0   0                   0   0   0   0   0
```

完全に一致する（[`test_affine.py`](https://github.com/oguratakayuki/image-processing-lab/blob/main/engine/tests/geometry/test_affine.py)の`test_translation_matrix_matches_translate`で検証）。

**合成の非可換性**も実際の行列で確認する。`2倍・1倍`（`sx=2, sy=1`）の拡大縮小と`30度`の回転を、順序を変えて合成すると：

```
R・S（拡大縮小→回転）:              S・R（回転→拡大縮小）:
 1.732  -0.5     0                  1.732  -1      0
 1       0.866   0                  0.5     0.866  0
 0       0       1                  0       0      1
```

2つの行列は異なる（最大で`0.5`の差がある成分もある）。一方、`sx=sy=2`のように均一な拡大縮小にすると、`R・S`と`S・R`は**完全に一致**する（[`test_affine.py`](https://github.com/oguratakayuki/image-processing-lab/blob/main/engine/tests/geometry/test_affine.py)の`test_uniform_scaling_commutes_with_rotation`で検証）——Web UIのデモでsx・syをわざと別の値にしているのはこのため。

## 関連

- [逆方向マッピングとは（補足資料）](5-1-translation-inverse-mapping.md) — このモジュールでも同じ骨格を再利用している
- [5-1. 平行移動](5-1-translation.md) — 2×2線形変換行列だけでは表現できなかった、同次座標が必要な理由そのもの
- [5-3. 回転](5-3-rotation.md) — 「中心を軸にする」処理を、平行移動との合成として説明し直せる
- [5-4. せん断](5-4-shear.md) — 2×2線形部分の逆行列の公式（行列式）をそのまま再利用

---

前へ: [← 5-4. せん断](5-4-shear.md)　｜　次へ: なし（現時点で最後の実装済み項目）
