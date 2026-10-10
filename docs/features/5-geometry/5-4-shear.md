# 5-4. せん断（Shear）

[← 5. 幾何学的変換へ戻る](README.md)　｜　[5-1](5-1-translation.md) · [5-2](5-2-scaling.md) · [5-3](5-3-rotation.md) · **5-4**

| | |
|---|---|
| 学習テーマ | せん断行列、一般の2×2行列の逆行列（行列式） |
| 実装ファイル | [`engine/src/imglab_engine/geometry/shear.py`](https://github.com/oguratakayuki/image-processing-lab/blob/main/engine/src/imglab_engine/geometry/shear.py) |
| 関数 | `shear()` |
| テスト | [`engine/tests/geometry/test_shear.py`](https://github.com/oguratakayuki/image-processing-lab/blob/main/engine/tests/geometry/test_shear.py) |

## 超ざっくり言うと

長方形の画像を、平行四辺形に歪ませる操作。一番上の行は動かず、下の行に行くほど横方向のズレが大きくなる。

## 概念

[5-2の拡大縮小](5-2-scaling.md)・[5-3の回転](5-3-rotation.md)と同じ「線形変換行列＋[逆方向マッピング](5-1-translation-inverse-mapping.md)＋最近傍補間」の枠組みに、3つ目の基本変換（せん断）を追加する。拡大縮小・回転はそれぞれ対角行列・直交行列という特別な形だったが、せん断行列はどちらの性質も持たない**一般の2×2行列**であり、逆行列の求め方も一般的な方法（行列式を使う）になる。

## 主な用途

- **文字のイタリック化**：フォントを斜めに傾けて疑似的な斜体を作る
- **画像の歪み補正**：カメラの傾きなどで生じた平行四辺形的な歪みを補正する
- **データ拡張（data augmentation）**：学習データに歪みのバリエーションを加える
- **後続のアフィン変換の一部品**：[5-2](5-2-scaling.md)・[5-3](5-3-rotation.md)と同じく、せん断も[同次座標によるアフィン変換の統一表現](README.md)で組み合わされる基本変換の1つ

この変換も座標だけを動かし、色の値そのものには一切手を加えないため、RGB画像にそのまま適用できる。

## 数学的背景

### 要件

画像を、縦方向の位置に応じて横にずらす（横方向の位置に応じて縦にずらす）ことで、長方形を平行四辺形に歪ませたい。

### やるべきこと

せん断は、2×2の**せん断行列**による線形写像として表せる。

```
⎡x'⎤   ⎡1    shx⎤ ⎡x⎤
⎢  ⎥ = ⎢        ⎥ ⎢ ⎥
⎣y'⎦   ⎣shy   1 ⎦ ⎣y⎦

つまり: x' = x + shx・y,  y' = shy・x + y
```

`shx`（横方向のせん断係数）は、`y`が大きい行ほど`x`を大きくずらす。`shy`はその逆で、`x`が大きい列ほど`y`を大きくずらす。

[逆方向マッピング](5-1-translation-inverse-mapping.md)を使うには、この行列の逆行列が必要になる。[5-2の対角行列](5-2-scaling.md)は対角成分の逆数を取るだけ、[5-3の回転行列](5-3-rotation.md)は転置を取るだけという、それぞれ特別な求め方ができた。一方、せん断行列はそのどちらの性質も持たない、一般的な2×2行列である。

一般の2×2行列`M = [[a,b],[c,d]]`の逆行列は、**行列式（determinant）**`det = a・d - b・c`を使って

```
       1      ⎡ d  -b⎤
M⁻¹ = ---- ・  ⎢      ⎥
      det      ⎣-c   a⎦
```

と計算できる（`det`が0だと逆行列が存在しない＝変換が不可逆になる。[なぜそうなるかは補足資料を参照](5-4-shear-determinant-zero.md)）。せん断行列（`a=1, b=shx, c=shy, d=1`）に当てはめると、`det = 1 - shx・shy`であり、

```
⎡1    shx⎤⁻¹        1     ⎡1    -shx⎤
⎢        ⎥    =  --------- ⎢         ⎥
⎣shy   1 ⎦       1-shx・shy ⎣-shy   1 ⎦
```

つまり`src_x = (out_x - shx・out_y) / det`、`src_y = (-shy・out_x + out_y) / det`で入力側の座標を逆算する。

**原点を基準にする**：[5-3の回転](5-3-rotation.md)は原点を中心に回転させると画像全体が枠の外にずれてしまうため、画像の中心を軸にする必要があった。せん断は性質が異なり、`y=0`の行（画像の一番上の行）は`x'=x+shx・0=x`となり全く動かない。原点（左上の角）を基準にしたまま、行ごとに少しずつ横にずらしていく変換として素直に扱える（[5-1](5-1-translation.md)・[5-2](5-2-scaling.md)と同じく中心合わせ不要）。

補間・境界処理は[5-3の回転](5-3-rotation.md)と同じ方式（最近傍補間、範囲外は黒で埋める）を再利用する。せん断でも画像の一部が枠の外に押し出される領域が本当に生じるため、[5-2のクランプ](5-2-scaling.md)ではなく黒埋めが適切。

### コードにすると

出力画像の各点`(out_y, out_x)`について、せん断行列の逆行列（行列式`det`を使った式）を適用して入力側の座標を逆算し、`round()`で最近傍補間、範囲内かどうかを判定して値をコピーする（範囲外なら黒のまま）——これがそのまま後述の`shear()`の実装になる。

## 実装コード

```python
import numpy as np


def shear(image: np.ndarray, shx: float, shy: float) -> np.ndarray:
    # image: せん断する画像。(H, W)または(H, W, 3)のuint8配列。
    # shx: 横方向のせん断係数。0で変化なし。
    # shy: 縦方向のせん断係数。0で変化なし。
    # 返却値: せん断変換後の画像。入力と同じ形状・dtype。
    height, width = image.shape[:2]

    det = 1 - shx * shy

    output = np.zeros_like(image)

    for out_y in range(height):
        for out_x in range(width):
            src_x = (out_x - shx * out_y) / det
            src_y = (-shy * out_x + out_y) / det

            nearest_y = round(src_y)
            nearest_x = round(src_x)

            # 入力座標は実際に画像の中に存在するかどうか。
            if 0 <= nearest_y < height and 0 <= nearest_x < width:
                output[out_y, out_x] = image[nearest_y, nearest_x]

    return output
```

### 解説

1. **`det = 1 - shx * shy`**：せん断行列の行列式。「やるべきこと」で導出した逆行列の式の分母にあたる。
2. **`src_x = (out_x - shx * out_y) / det`, `src_y = (-shy * out_x + out_y) / det`**：逆行列を出力座標に適用して、入力側の座標を逆算する。
3. **`nearest_y = round(src_y)`, `nearest_x = round(src_x)`**：[5-2と同じ最近傍補間](5-2-scaling.md)。
4. **`if 0 <= nearest_y < height and 0 <= nearest_x < width:`**：[5-1・5-3と同じ範囲内判定](5-1-translation.md)。クランプではなく、範囲外ならゼロ初期化済みの値（黒）のままにする。

## 具体的な計算例

5×5画像で、左端の列（`x=0`）だけが前景（255）の画像を`shx=1.0`でせん断する。

```
入力:
255   0   0   0   0
255   0   0   0   0
255   0   0   0   0
255   0   0   0   0
255   0   0   0   0

出力 (shx=1.0, shy=0.0):
255   0   0   0   0
  0 255   0   0   0
  0   0 255   0   0
  0   0   0 255   0
  0   0   0   0 255
```

縦の線が、対角線に変わっている。`y=0`の行（`x'=x+1.0・0=x`）は動かず、`y`が1増えるごとに`x`方向に1ずつずれていく——`shx`が「行の位置に応じて横にずらす」という定義通りの結果。[`test_shear.py`](https://github.com/oguratakayuki/image-processing-lab/blob/main/engine/tests/geometry/test_shear.py)の`test_shear_x_turns_vertical_line_into_diagonal`で検証している値と一致する。

`shy`方向も確認する。`(0,0)`と`(0,4)`の2点を`shy=0.5`でせん断すると：

```
入力:                         出力 (shx=0.0, shy=0.5):
255   0   0   0   0           255   0   0   0   0
  0   0   0   0   0             0   0   0   0   0
  0   0   0   0   0             0   0   0   0 255
  0   0   0   0   0             0   0   0   0   0
  0   0   0   0   0             0   0   0   0   0
```

`x=0`の点`(0,0)`は`y'=0.5・0+0=0`で動かないが、`x=4`の点`(0,4)`は`y'=0.5・4+0=2`で2段下に移る——`shy`が「列の位置に応じて縦にずらす」という定義通り。

## 関連

- [逆方向マッピングとは（補足資料）](5-1-translation-inverse-mapping.md) — 順方向マッピングとの比較を、穴が空く具体例とともに解説
- [行列式が0だと逆行列が存在しない理由（補足資料）](5-4-shear-determinant-zero.md) — 複数の入力点が1点に潰れる具体例
- [5-2. 拡大縮小](5-2-scaling.md) — 対角行列という特別な形の逆行列の求め方との対比
- [5-3. 回転](5-3-rotation.md) — 直交行列という特別な形の逆行列の求め方との対比、補間・境界処理の方式が共通

---

前へ: [← 5-3. 回転](5-3-rotation.md)　｜　次へ: なし（現時点で最後の実装済み項目）
