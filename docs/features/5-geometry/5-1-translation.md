# 5-1. 平行移動（Translation）

[← 5. 幾何学的変換へ戻る](README.md)　｜　**5-1**

| | |
|---|---|
| 学習テーマ | 座標変換、逆方向マッピング（inverse mapping） |
| 実装ファイル | [`engine/src/imglab_engine/geometry/translation.py`](https://github.com/oguratakayuki/image-processing-lab/blob/main/engine/src/imglab_engine/geometry/translation.py) |
| 関数 | `translate()` |
| テスト | [`engine/tests/geometry/test_translation.py`](https://github.com/oguratakayuki/image-processing-lab/blob/main/engine/tests/geometry/test_translation.py) |

## 超ざっくり言うと

画像全体を、指定した分だけ横・縦にずらす操作。はみ出した部分は消え、反対側には黒い余白ができる。

## 概念

これまでの4グループは「ある座標のピクセルの**値**をどう計算するか」を扱ってきたが、平行移動は「ピクセルの**値はそのまま**で、**どの座標に置くか**」を扱う、質の異なる変換。画像を縦横にスライドさせるだけのシンプルな操作だが、後続の回転・拡大縮小で必要になる「逆方向マッピング」という考え方を、ここで最初に導入する。

## 主な用途

- **位置合わせ（アライメント）**：複数の画像を同じ基準点に揃える前処理
- **データ拡張（data augmentation）**：機械学習の学習データを水増しする際に、同じ画像を少しずらしたバリエーションを作る
- **後続の幾何学的変換の基礎**：回転・拡大縮小・せん断も「出力の各点から入力の対応点を逆算する」という同じ構造で実装でき、平行移動はその最も単純な例になる

## 数学的背景

### 要件

画像上の全てのピクセルを、同じ向き・同じ距離だけずらしたい。ずらす量はベクトル`(tx, ty)`で指定する（`tx`は横方向、`ty`は縦方向）。

### やるべきこと

素直に考えると、入力の各点`(x, y)`を`(x+tx, y+ty)`に書き込めばよさそうに思える（順方向マッピング）。

```
(x, y) -> (x + tx, y + ty)
```

しかしこの実装方針には将来的な問題がある。平行移動のような整数シフトでは起きないが、後続の回転・拡大縮小では、変換後の座標が整数になるとは限らない。出力側の整数座標を「書き込まれた点」と「書き込まれなかった点（穴）」に分けて考えると、非整数座標への書き込みを四捨五入で処理する限り、出力側に穴が空く可能性がある。

そこで、**出力側から始めて、対応する入力側の点を逆算する**という方針（逆方向マッピング, inverse mapping）を使う。出力の各点`(out_x, out_y)`について、そこに写ってくる入力側の点を、変換の逆変換で求める。

```
out(out_x, out_y) = in(out_x - tx, out_y - ty)
```

平行移動の逆変換は、符号を反転するだけ（`(tx, ty) -> (-tx, -ty)`）。出力側を1点ずつ漏れなく埋めていく方式なので、穴は原理的に発生しない。

逆算した入力側の座標`(out_x - tx, out_y - ty)`が画像の範囲外になる場合（画像の端からはみ出す／新しい領域が現れる場合）は、[畳み込み・モルフォロジー演算と同じゼロパディングの考え方](../4-morphology/4-1-erosion.md)で、`0`（黒）を割り当てる。

### コードにすると

出力画像の各点`(out_y, out_x)`についてループし、`src_y = out_y - ty`・`src_x = out_x - tx`で入力側の座標を逆算、範囲内なら入力の値をコピー、範囲外ならゼロ初期化済みの値（黒）のままにする——これがそのまま後述の`translate()`の実装になる。

## 実装コード

```python
import numpy as np


def translate(image: np.ndarray, tx: int, ty: int) -> np.ndarray:
    # image: 平行移動する画像。(H, W)または(H, W, 3)のuint8配列。
    # tx: x方向(横方向、列方向)の移動量(ピクセル)。正の値で右に移動する。
    # ty: y方向(縦方向、行方向)の移動量(ピクセル)。正の値で下に移動する。
    # 返却値: 平行移動後の画像。入力と同じ形状・dtype。画像の端から
    #         はみ出した部分は切り捨てられ、新しく現れた部分は
    #         0(黒)で埋められる。
    height, width = image.shape[:2]

    output = np.zeros_like(image)

    for out_y in range(height):
        for out_x in range(width):
            src_y = out_y - ty
            src_x = out_x - tx

            if 0 <= src_y < height and 0 <= src_x < width:
                output[out_y, out_x] = image[src_y, src_x]

    return output
```

### 解説

1. **`height, width = image.shape[:2]`**：`image.shape`は`(H, W)`（Grayscale）なら2要素、`(H, W, 3)`（RGB）なら3要素のタプル。`[:2]`のスライスでどちらでも先頭2要素（高さ・幅）だけを取り出せるため、この関数はGrayscale画像にもRGB画像にもそのまま使える（[4のモルフォロジー演算](../4-morphology/4-1-erosion.md)と違い、色を一切見ないのでGrayscale変換の前処理が不要）。
2. **`output = np.zeros_like(image)`**：`image`と同じ形状・同じ型の配列を作り、全要素を`0`で初期化する。まず画像全体を黒にしておき、対応する入力が見つかった点だけ後から値を上書きする、という組み立て方。
3. **`for out_y in range(height): for out_x in range(width):`**：出力画像の全ての点を、1つずつ漏れなく処理する二重ループ。これが「出力側から始める」逆方向マッピングの骨格。
4. **`src_y = out_y - ty`, `src_x = out_x - tx`**：平行移動の逆変換（符号を反転するだけ）で、出力の点`(out_y, out_x)`に対応する入力側の座標を逆算する。
5. **`if 0 <= src_y < height and 0 <= src_x < width:`**：逆算した座標が画像の範囲内にあるかを確認する（Pythonでは`0 <= x < n`と連続して書くと「`0 <= x`かつ`x < n`」という意味になる）。
6. **`output[out_y, out_x] = image[src_y, src_x]`**：範囲内なら、入力画像のその位置の値（Grayscaleなら輝度1つ、RGBならR,G,Bの3つ）をそのままコピーする。範囲外の場合はこの行が実行されず、`output`は手順2で初期化した`0`（黒）のまま残る。

## 具体的な計算例

5×5画像で、`(1,1)`だけが前景(255)の画像を`tx=2, ty=1`で平行移動する。

```
入力:
  0   0   0   0   0
  0 255   0   0   0
  0   0   0   0   0
  0   0   0   0   0
  0   0   0   0   0

出力 (tx=2, ty=1):
  0   0   0   0   0
  0   0   0   0   0
  0   0   0 255   0
  0   0   0   0   0
  0   0   0   0   0
```

`(1,1)`の点が、横に2・縦に1動いた`(2,3)`（縦インデックス2, 横インデックス3）に移っている。[`test_translation.py`](https://github.com/oguratakayuki/image-processing-lab/blob/main/engine/tests/geometry/test_translation.py)の`test_translate_moves_single_pixel_right_and_down`で検証している値と一致する。

端のクロップも確認する。4×4全面が前景(255)の画像を`tx=2, ty=0`で平行移動すると：

```
入力:
255 255 255 255
255 255 255 255
255 255 255 255
255 255 255 255

出力 (tx=2, ty=0):
  0   0 255 255
  0   0 255 255
  0   0 255 255
  0   0 255 255
```

右に2マス分ずれ、左端2列は「画像の外（範囲外）からやってきた」ことになるため黒(0)、元々あった右端2列分の前景は画像の外にはみ出して消えている。

## 関連

- [1-2. 明るさ・コントラスト調整](../1-color/1-2-brightness-contrast.md) — 「線形写像→アフィン変換」という同じ数学の流れを、値ではなく座標に適用した例
- [4-1. Erosion（収縮）](../4-morphology/4-1-erosion.md) — ゼロパディング（画像の外は0とみなす）という境界処理の考え方が共通

---

前へ: [← 4-4. Closing（クロージング）](../4-morphology/4-4-closing.md)　｜　次へ: なし（現時点で最後の実装済み項目）
