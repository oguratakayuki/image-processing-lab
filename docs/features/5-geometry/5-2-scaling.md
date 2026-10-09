# 5-2. 拡大縮小（Scaling）

[← 5. 幾何学的変換へ戻る](README.md)　｜　[5-1](5-1-translation.md) · **5-2**

| | |
|---|---|
| 学習テーマ | 線形変換行列（対角行列）、逆行列、最近傍補間 |
| 実装ファイル | [`engine/src/imglab_engine/geometry/scaling.py`](https://github.com/oguratakayuki/image-processing-lab/blob/main/engine/src/imglab_engine/geometry/scaling.py) |
| 関数 | `scale()` |
| テスト | [`engine/tests/geometry/test_scaling.py`](https://github.com/oguratakayuki/image-processing-lab/blob/main/engine/tests/geometry/test_scaling.py) |

## 超ざっくり言うと

画像を縦横それぞれ指定した倍率で拡大・縮小する操作。拡大すると同じ画素が繰り返されてブロック状になり、縮小すると画素が間引かれる。

## 概念

[5-1の平行移動](5-1-translation.md)は整数シフトだったため、逆算した入力座標は常に整数だった。拡大縮小では、逆算した入力座標が**整数になるとは限らない**という新しい問題が出てくる。この「非整数座標の画素値をどう決めるか」という問題（補間, interpolation）への最も単純な答えが、今回扱う**最近傍補間**。

## 主な用途

- **画像のリサイズ**：サムネイル作成、表示サイズに合わせた縮小・拡大
- **データ拡張（data augmentation）**：学習データを異なる縮尺で水増しする
- **後続のアフィン変換の一部品**：拡大縮小は回転・せん断と並ぶ基本変換の1つで、[同次座標によるアフィン変換の統一表現](README.md)で組み合わされる

この変換も[5-1](5-1-translation.md)と同じく座標だけを動かし、色の値そのものには一切手を加えないため、RGB画像にそのまま適用できる。

## 数学的背景

### 要件

画像を、横方向に`sx`倍・縦方向に`sy`倍したい（`sx`と`sy`は独立に指定でき、異なる値にすると縦横比が変わる）。

### やるべきこと

拡大縮小は、座標`(x, y)`をそれぞれ独立に`sx`倍・`sy`倍する変換であり、2×2の**対角行列**による線形写像として表せる。

```
[x']   [sx  0 ]   [x]
[y'] = [0   sy] * [y]

つまり: x' = sx・x,  y' = sy・y
```

[5-1](5-1-translation.md)と同じく逆方向マッピングを使う。出力の各点`(out_x, out_y)`について、対応する入力側の座標を、この変換の**逆行列**で逆算する。対角行列の逆行列は、対角成分の逆数を取るだけでよい（`[[sx,0],[0,sy]]`と`[[1/sx,0],[0,1/sy]]`を掛け合わせると単位行列になることで確認できる）。

```
src_x = out_x / sx
src_y = out_y / sy
```

`sx, sy`が1以外のとき、`src_x, src_y`は一般に非整数になる。しかし画像データは整数座標の格子点にしか存在しないため、非整数座標の値をそのままでは取得できない。この問題への最も単純な対処法が**最近傍補間（nearest neighbor interpolation）**：逆算した非整数座標を四捨五入して、最も近い整数座標の画素値をそのまま使う。

丸めた座標が画像の範囲をわずかに超える場合（丸め誤差による境界ちょうどのケース）は、範囲内に収まるよう上限・下限で挟み込む（クランプ, clamping）。[5-1](5-1-translation.md)は範囲外を黒で埋めたが、拡大縮小では「画像の端の画素をそのまま引き伸ばす」方が見た目として自然なため、別の対処（クランプ）を選んでいる。

出力画像のサイズも、入力の`(height, width)`に`sy, sx`を掛けたものになる（[5-1](5-1-translation.md)は入力と同じサイズの枠内で動かしていたが、拡大縮小は画像そのものの大きさを変える操作のため）。

### コードにすると

出力画像の各点`(out_y, out_x)`についてループし、`src_y = out_y / sy`・`src_x = out_x / sx`で入力側の座標を逆算、`round()`で最も近い整数座標に丸め、範囲内に収まるようクランプしてから入力の値をコピーする——これがそのまま後述の`scale()`の実装になる。

## 実装コード

```python
import numpy as np


def scale(image: np.ndarray, sx: float, sy: float) -> np.ndarray:
    # image: 拡大縮小する画像。(H, W)または(H, W, 3)のuint8配列。
    # sx: x方向(横方向)の拡大率。2.0なら2倍、0.5なら半分になる。
    # sy: y方向(縦方向)の拡大率。sxと独立に指定できる。
    # 返却値: 拡大縮小後の画像。(round(H*sy), round(W*sx))または
    #         (round(H*sy), round(W*sx), 3)のuint8配列。
    height, width = image.shape[:2]

    new_height = round(height * sy)
    new_width = round(width * sx)

    output_shape = (new_height, new_width) + image.shape[2:]
    output = np.zeros(output_shape, dtype=image.dtype)

    for out_y in range(new_height):
        for out_x in range(new_width):
            src_y = out_y / sy
            src_x = out_x / sx

            nearest_y = round(src_y)
            nearest_x = round(src_x)

            clamped_y = max(0, min(nearest_y, height - 1))
            clamped_x = max(0, min(nearest_x, width - 1))

            output[out_y, out_x] = image[clamped_y, clamped_x]

    return output
```

### 解説

1. **`height, width = image.shape[:2]`**：[5-1](5-1-translation.md)と同じく、Grayscale/RGBどちらの形状でも先頭2要素だけを取り出す。
2. **`new_height = round(height * sy)`, `new_width = round(width * sx)`**：出力画像のサイズを、入力サイズに拡大率を掛けて求める。ここで`round()`を使っているが、Pythonの`round()`は「0.5のときは最も近い**偶数**に丸める」という**銀行丸め（round half to even）**を採用している（Ruby/PHPの`round`のような「0.5は常に切り上げ」ではない）。例えば`round(2.5)`は`2`、`round(3.5)`は`4`になる。画像サイズの計算では通常問題にならないほど小さい誤差だが、Pythonの丸めの挙動の癖として覚えておく価値がある。
3. **`output_shape = (new_height, new_width) + image.shape[2:]`**：`image.shape[2:]`は、`(H, W, 3)`なら`(3,)`、`(H, W)`なら空のタプル`()`になるスライス。これをタプル同士の連結（`+`演算子）で出力の形状に継ぎ足すことで、Grayscale/RGB両対応の形状を1行で組み立てている。
4. **`src_y = out_y / sy`, `src_x = out_x / sx`**：出力の点`(out_y, out_x)`に対応する入力側の座標を、拡大縮小の逆変換（`sx, sy`で割る）で逆算する。
5. **`nearest_y = round(src_y)`, `nearest_x = round(src_x)`**：最近傍補間。非整数座標を四捨五入して、最も近い整数座標（入力画像上の実在するピクセル）に丸める。
6. **`clamped_y = max(0, min(nearest_y, height - 1))`**：丸めた座標が画像の範囲をわずかに超える場合に備えて、範囲内の最も近い値に強制的に収める（クランプ）。Pythonには他言語にあるような専用の`clamp`関数が無いため、`max`と`min`を組み合わせて書く（`min(nearest_y, height-1)`で上限を、`max(0, ...)`で下限を保証する）。`clamped_x`も同様。
7. **`output[out_y, out_x] = image[clamped_y, clamped_x]`**：クランプ済みの入力座標から値をコピーする。[5-1](5-1-translation.md)のような範囲外判定（`if`文で弾く）ではなく、常にこの1行が実行される（クランプにより必ず有効な座標になっているため）。

## 具体的な計算例

2×2画像を2倍に拡大する。

```
入力:
10 20
30 40

出力 (sx=2.0, sy=2.0):
10 10 20 20
10 10 20 20
30 30 40 40
30 30 40 40
```

各入力画素が2×2のブロックに広がっている。[`test_scaling.py`](https://github.com/oguratakayuki/image-processing-lab/blob/main/engine/tests/geometry/test_scaling.py)の`test_scale_doubles_both_dimensions`で検証している値と一致する。

縮小も確認する。4×4画像を0.5倍にすると：

```
入力:
 10  20  30  40
 50  60  70  80
 90 100 110 120
130 140 150 160

出力 (sx=0.5, sy=0.5):
 10  30
 90 110
```

1つ飛ばしで画素が間引かれている（`out_y=0`→`src_y=0`→`10,30`の行、`out_y=1`→`src_y=2`→`90,110`の行）。

## 関連

- [5-1. 平行移動](5-1-translation.md) — 同じ逆方向マッピングの構造を使う最初の例。整数シフトのみで補間が不要だった
- [1-2. 明るさ・コントラスト調整](../1-color/1-2-brightness-contrast.md) — 「線形写像→アフィン変換」という数学の流れの起点

---

前へ: [← 5-1. 平行移動](5-1-translation.md)　｜　次へ: なし（現時点で最後の実装済み項目）
