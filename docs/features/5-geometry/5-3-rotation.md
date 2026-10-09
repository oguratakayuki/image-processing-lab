# 5-3. 回転（Rotation）

[← 5. 幾何学的変換へ戻る](README.md)　｜　[5-1](5-1-translation.md) · [5-2](5-2-scaling.md) · **5-3**

| | |
|---|---|
| 学習テーマ | 回転行列、直交行列の逆行列、回転中心 |
| 実装ファイル | [`engine/src/imglab_engine/geometry/rotation.py`](https://github.com/oguratakayuki/image-processing-lab/blob/main/engine/src/imglab_engine/geometry/rotation.py) |
| 関数 | `rotate()` |
| テスト | [`engine/tests/geometry/test_rotation.py`](https://github.com/oguratakayuki/image-processing-lab/blob/main/engine/tests/geometry/test_rotation.py) |

## 超ざっくり言うと

画像を、中心を軸にぐるっと回転させる操作。正の角度で時計回りに回る。

## 概念

[5-2の拡大縮小](5-2-scaling.md)で導入した「線形変換行列＋逆方向マッピング＋最近傍補間」の枠組みを、今度は**回転行列**に適用する。拡大縮小と同じく、逆算した入力座標は一般に非整数になるため、最近傍補間をそのまま再利用する。回転ならではの新しい論点は、**回転行列の逆行列の求め方**と、**回転の軸（中心）をどこに置くか**の2点。

## 主な用途

- **画像の向きの補正**：スキャンした書類の傾き補正など
- **データ拡張（data augmentation）**：学習データを様々な角度で水増しする
- **後続のアフィン変換の一部品**：[5-2](5-2-scaling.md)と同じく、回転も[同次座標によるアフィン変換の統一表現](README.md)で組み合わされる基本変換の1つ

この変換も座標だけを動かし、色の値そのものには一切手を加えないため、RGB画像にそのまま適用できる。

## 数学的背景

### 要件

画像を、指定した角度`theta`（度数法）だけ回転させたい。回転させても画像が枠の外にずれていかないように、**画像の中心を軸**に回転させる。

### やるべきこと

角度`theta`の回転は、2×2の**回転行列**による線形写像として表せる。

```
⎡x'⎤   ⎡cosθ  -sinθ⎤ ⎡x⎤
⎢  ⎥ = ⎢           ⎥ ⎢ ⎥
⎣y'⎦   ⎣sinθ   cosθ⎦ ⎣y⎦
```

[5-2](5-2-scaling.md)と同じく逆方向マッピングを使うため、この行列の**逆行列**が必要になる。回転行列は「直交行列」（各列ベクトルの大きさが1で、互いに直交する）という特別な性質を持ち、直交行列の逆行列は転置行列（行と列を入れ替えたもの）に一致する。さらに回転の場合、転置を取ることは「逆向き（`-theta`）に回転する」ことと同じになる（`cos(-θ)=cosθ`、`sin(-θ)=-sinθ`という性質から）。

```
⎡cosθ  -sinθ⎤⁻¹   ⎡ cosθ  sinθ⎤
⎢           ⎥   = ⎢           ⎥
⎣sinθ   cosθ⎦      ⎣-sinθ  cosθ⎦
```

[5-2の対角行列の逆行列](5-2-scaling.md)（対角成分の逆数を取るだけ）とは違う、回転行列ならではの逆行列の求め方になっている。

**回転中心の問題**：原点`(0, 0)`を中心に回転させると、画像全体が枠の外にずれてしまう（回転は原点からの距離・角度を保つ変換なので、原点から離れた画像全体が弧を描くように移動する）。これを避けるため、**画像の中心**を基準に回転させる。出力の座標からいったん中心を引いて「中心からの相対座標」にし、逆回転を適用し、最後にまた中心を足して元の座標系に戻す。

逆算した座標は一般に非整数になるため[5-2と同じ最近傍補間](5-2-scaling.md)を使う。ただし[5-2のクランプ](5-2-scaling.md)（範囲外を端の値で埋める）とは異なり、ここでは[5-1の平行移動](5-1-translation.md)と同じ「範囲外は黒で埋める」方式を使う。回転で画像の四隅が枠の外へ出ていくのは丸め誤差ではなく正しい挙動であり、クランプしてしまうと本来何も無いはずの領域に端の画素が間違って引き伸ばされてしまうため。

### コードにすると

出力画像の各点`(out_y, out_x)`について、中心からの相対座標に変換→逆回転を適用→中心を足し戻す、という3段階で入力側の座標を逆算し、`round()`で最近傍補間、範囲内かどうかを判定して値をコピーする（範囲外なら黒のまま）——これがそのまま後述の`rotate()`の実装になる。

## 実装コード

```python
import math

import numpy as np


def rotate(image: np.ndarray, degrees: float) -> np.ndarray:
    # image: 回転する画像。(H, W)または(H, W, 3)のuint8配列。
    # degrees: 回転角度(度数法)。正の値で時計回りに回転する。
    # 返却値: 回転後の画像。入力と同じ形状・dtype。
    height, width = image.shape[:2]

    center_y = (height - 1) / 2.0
    center_x = (width - 1) / 2.0

    theta = math.radians(degrees)
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)

    output = np.zeros_like(image)

    for out_y in range(height):
        for out_x in range(width):
            rel_y = out_y - center_y
            rel_x = out_x - center_x

            src_rel_x = cos_theta * rel_x + sin_theta * rel_y
            src_rel_y = -sin_theta * rel_x + cos_theta * rel_y

            src_y = src_rel_y + center_y
            src_x = src_rel_x + center_x

            nearest_y = round(src_y)
            nearest_x = round(src_x)

            # 入力座標は実際に画像の中に存在するかどうか。
            if 0 <= nearest_y < height and 0 <= nearest_x < width:
                output[out_y, out_x] = image[nearest_y, nearest_x]

    return output
```

### 解説

1. **`center_y = (height - 1) / 2.0`**：画像の中心座標。座標が`0`から`height-1`までの`height`個の整数で構成されているため、ちょうど真ん中の値は`(0 + (height-1)) / 2`になる（`center_x`も同様）。
2. **`theta = math.radians(degrees)`**：`math.radians()`は度数法を弧度法（ラジアン）に変換するPython標準ライブラリの関数。NumPy・Pythonの`math.cos`/`math.sin`はいずれも弧度法を前提とするため、ここで変換しておく。
3. **`rel_y = out_y - center_y`, `rel_x = out_x - center_x`**：出力座標を、中心を原点とみなした相対座標に変換する。
4. **`src_rel_x = cos_theta * rel_x + sin_theta * rel_y`, `src_rel_y = -sin_theta * rel_x + cos_theta * rel_y`**：逆回転（`-theta`分の回転）を適用して、入力側の相対座標を求める。上の「やるべきこと」で導出した逆行列をそのままコードにしたもの。
5. **`src_y = src_rel_y + center_y`, `src_x = src_rel_x + center_x`**：中心を足し戻して、画像全体の座標系に戻す。
6. **`nearest_y = round(src_y)`, `nearest_x = round(src_x)`**：[5-2と同じ最近傍補間](5-2-scaling.md)。
7. **`if 0 <= nearest_y < height and 0 <= nearest_x < width:`**：[5-1と同じ範囲内判定](5-1-translation.md)。クランプではなく、範囲外ならゼロ初期化済みの値（黒）のままにする。

## 具体的な計算例

5×5画像で、上端中央`(0,2)`にある点を回転する。

```
入力:
  0   0 255   0   0
  0   0   0   0   0
  0   0   0   0   0
  0   0   0   0   0
  0   0   0   0   0

90度回転:              -90度回転:              180度回転:
  0   0   0   0   0       0   0   0   0   0       0   0   0   0   0
  0   0   0   0   0       0   0   0   0   0       0   0   0   0   0
  0   0   0   0 255     255   0   0   0   0       0   0   0   0   0
  0   0   0   0   0       0   0   0   0   0       0   0   0   0   0
  0   0   0   0   0       0   0   0   0   0       0   0 255   0   0
```

90度（時計回り）で右端中央、-90度（反時計回り）で左端中央、180度で下端中央に移っている。正の角度が時計回りになるのは、画像座標系では`y`が下向きのため（通常の数学の座標系の反時計回りが、画像では見た目上は時計回りになる）。[`test_rotation.py`](https://github.com/oguratakayuki/image-processing-lab/blob/main/engine/tests/geometry/test_rotation.py)で検証している値と一致する。

**最近傍補間の限界**：同じ点を45度回転させると、**全てのピクセルが0になり、点が消えてしまう**。

```
45度回転:
  0   0   0   0   0
  0   0   0   0   0
  0   0   0   0   0
  0   0   0   0   0
  0   0   0   0   0
```

これは実際に計算してみると分かる通り、5×5のどの出力座標`(out_y, out_x)`も、逆算して最近傍補間した結果が`(0, 2)`に一致しないために起こる（実装上のバグではなく、孤立した1ピクセルの点が、回転角度次第では出力のどの格子点からも「最も近い」とみなされず、丸め込みの過程で消えてしまうという、最近傍補間そのものの限界）。7×7画像で3×3の塊（面積のある領域）を45度回転させると、この問題は起きない。

```
入力:                       45度回転:
0 0 0 0 0 0 0               0 0 0 0 0 0 0
0 0 0 0 0 0 0               0 0 0 255 0 0 0
0 0 255 255 255 0 0         0 0 255 255 255 0 0
0 0 255 255 255 0 0         0 255 255 255 255 255 0
0 0 255 255 255 0 0         0 0 255 255 255 0 0
0 0 0 0 0 0 0               0 0 0 255 0 0 0
0 0 0 0 0 0 0               0 0 0 0 0 0 0
```

3×3の正方形が、45度回転してひし形になっている。面積のある領域なら、周辺の出力座標のどれかが必ず拾ってくれるため、最近傍補間でも破綻しない。

## 関連

- [5-2. 拡大縮小](5-2-scaling.md) — 線形変換行列・逆方向マッピング・最近傍補間という同じ枠組みを使う
- [5-1. 平行移動](5-1-translation.md) — 範囲外を黒で埋めるという境界処理が共通

---

前へ: [← 5-2. 拡大縮小](5-2-scaling.md)　｜　次へ: なし（現時点で最後の実装済み項目）
