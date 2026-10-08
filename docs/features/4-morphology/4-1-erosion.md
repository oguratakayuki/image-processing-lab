# 4-1. Erosion（収縮）

[← 4. 二値画像処理へ戻る](README.md)　｜　**4-1** · [4-2](4-2-dilation.md) · [4-3](4-3-opening.md) · [4-4](4-4-closing.md)

| | |
|---|---|
| 学習テーマ | 集合演算（AND）、構造要素 |
| 実装ファイル | [`engine/src/imglab_engine/morphology/morphology.py`](https://github.com/oguratakayuki/image-processing-lab/blob/main/engine/src/imglab_engine/morphology/morphology.py) |
| 関数 | `erode()`, `square_structuring_element()` |
| テスト | [`engine/tests/morphology/test_morphology.py`](https://github.com/oguratakayuki/image-processing-lab/blob/main/engine/tests/morphology/test_morphology.py) |

## 超ざっくり言うと

白い部分（前景）を、全方向に1段階ずつ削って小さくする操作。細い部分は消え、太い部分は輪郭が内側に後退する。

## 概念

[4. 二値画像処理](README.md)では、二値画像（[前景255・背景0とは何か](../../qa/README.md)）をピクセル値の並びとしてではなく、**「前景ピクセルの座標の集合」**として捉える。Erosion（収縮）は、この集合を内側に向かって縮める操作。小さな図形（構造要素）を画像の上でずらしながら、「構造要素の形がすっぽり前景に収まる場所だけ」を新しい前景として残す。

## 主な用途

- **小さなノイズ・孤立点の除去**：構造要素より小さい点は収縮で完全に消える（単独で使う他、[4-3のOpening](4-3-opening.md)の前半としても使われる）
- **くっついた物体の分離**：2つの物体が細い部分でつながっている二値画像を収縮すると、細い連結部分が先に消えて物体同士が分離する
- **輪郭線の抽出**：元の前景から収縮後の前景を引く（`A - erode(A)`）と、前景の外周1〜数ピクセル分の輪郭線だけが残る

**注意：この「ノイズ除去」はあくまで前段の変換ありき**。Erosionは`0`/`255`の二値画像しか見ておらず、色（RGB）を一切見ていない。このプロジェクトのパイプラインは`RGB → Grayscale変換([1-1](../1-color/1-1-grayscale.md)) → 閾値処理([1-4](../1-color/1-4-threshold.md)) → Erosion`という順になっており、「ノイズが消せるかどうか」は実際にはGrayscale変換と閾値処理の時点で、ノイズと背景が前景/背景の異なる側に正しく分離できているかに懸かっている。例えば赤背景`(255,0,0)`に黄色いノイズ点`(255,255,0)`がある場合、Grayscale変換後の値はそれぞれ`76`と`226`と十分に離れるため`t=100`で正しく分離でき、Erosionでノイズを消せる（実際に計算して確認済み）。しかし背景とノイズの色がGrayscale変換後に近い値になる組み合わせ（[Grayscale変換は不可逆](../1-color/1-1-grayscale.md)なので、異なる色が同じ明るさに写ることがある）だと、閾値処理の時点で両者を区別できず、Erosionでは原理的に除去できない。

→ 実際の計算例は[docs/qa/README.md](../../qa/README.md)を参照。

## 数学的背景

### 要件

二値画像を「前景ピクセルの座標の集合」として捉える。

```
A = {(x, y) : I(x, y) = 255}
```

この`A`に対して、小さな図形（構造要素, structuring element）`B`が**完全に収まる場所だけ**を新しい前景として残したい。構造要素の形が収まりきらない場所（前景の輪郭付近や、構造要素より小さい領域）は削り落とす。

### やるべきこと

`B`を点`z`だけ平行移動した集合を`B_z`と書くと、「構造要素が完全に収まる」とは`B_z`が`A`の部分集合であること（`B_z ⊆ A`）。これを満たす`z`だけを集めればよい。

```
A ⊖ B = {z : B_z ⊆ A}
```

`B_z ⊆ A`は、「構造要素が指す近傍のピクセルが、1つ残らず全て前景かどうか」という判定と同値——これは論理演算の**AND**そのものである。

### コードにすると

各点`z`について、構造要素が指す近傍パッチを切り出し、`np.all(patch == 255)`で「全て前景か」を判定する。これがそのまま後述の`erode()`の実装になる。

（これは[2. 畳み込み](../2-convolution/README.md)で学んだ「近傍を見て新しい値を決める」という操作の型と共通するが、中身が異なる。畳み込みは近傍の値の**重み付き和（線形演算）**だったのに対し、モルフォロジー演算は近傍の値の**論理AND/OR（非線形の集合演算）**である。）

## 実装コード

```python
def square_structuring_element(size: int = 3) -> np.ndarray:
    return np.ones((size, size), dtype=bool)


def _local_reduce(
    binary: np.ndarray, structuring_element: np.ndarray, reduce_fn
) -> np.ndarray:
    kernel_height, kernel_width = structuring_element.shape
    pad_height, pad_width = kernel_height // 2, kernel_width // 2
    padded = np.pad(
        binary,
        ((pad_height, pad_height), (pad_width, pad_width)),
        mode="constant",
        constant_values=0,
    )

    height, width = binary.shape
    output = np.zeros((height, width), dtype=np.uint8)

    for center_y in range(height):
        for center_x in range(width):
            # 元画像の座標(center_y, center_x)は、パディングによって
            # paddedの中では(center_y + pad_height, center_x + pad_width)の
            # 位置にずれている。そこを中心とするkernel_height x kernel_width
            # の近傍の左上は、(center_y + pad_height - pad_height,
            # center_x + pad_width - pad_width) = (center_y, center_x)と、
            # パディング分がちょうど打ち消し合って元の座標と一致する。
            patch_top = center_y
            patch_left = center_x

            # 計算済みの左上座標から、構造要素と同じ大きさの近傍を切り出す。
            # 行方向に「patch_top行目から、patch_top + kernel_height行目の
            # 手前まで」、列方向に「patch_left列目から、
            # patch_left + kernel_width列目の手前まで」抜き出す。
            patch = padded[
                patch_top : patch_top + kernel_height,
                patch_left : patch_left + kernel_width,
            ]

            # 構造要素がTrueを指す位置の値だけを取り出す
            # (構造要素が正方形以外の形でも対応できるようにするため)。
            covered_values = patch[structuring_element]
            is_foreground = covered_values == 255

            # (center_y, center_x)を収縮・膨張させるためのメインの判定処理。
            # 膨張(dilate)は構造要素のいずれかが前景なら前景になり、
            # 収縮(erode)は構造要素の全てが前景でないと前景にならない。
            output[center_y, center_x] = 255 if reduce_fn(is_foreground) else 0

    return output


def erode(binary: np.ndarray, structuring_element: np.ndarray) -> np.ndarray:
    return _local_reduce(binary, structuring_element, np.all)
```

### 解説

0. **前提**：`_local_reduce()`・`erode()`の引数`binary`は、[1-1のGrayscale変換](../1-color/1-1-grayscale.md)と[1-4の閾値処理](../1-color/1-4-threshold.md)を経て、すでに`0`（背景）か`255`（前景）の2値だけになった画像であることを前提にしている。この関数自体はRGBの色情報を一切受け取らず、見ることもない。
1. **引数・返却値**：`_local_reduce(binary, structuring_element, reduce_fn)`は3つの引数を取る——`binary`（二値画像）、`structuring_element`（構造要素、真偽値配列）、`reduce_fn`（近傍の真偽値配列を1つの真偽値に集約する関数。`erode`なら`np.all`、`dilate`なら`np.any`を渡す）。返却値は、同じ`(H, W)`の形をした二値画像（`erode`なら収縮後、`dilate`なら膨張後）。
2. **`square_structuring_element(size)`**：`size × size`の全てTrueな真偽値配列。`B`の最も単純な形（正方形）。
3. **`for center_y in range(height): for center_x in range(width):`**：[4-2で見た数式](4-2-dilation.md)の`{z : 条件}`の「`z`を画像上の全ての点について動かす」を実行する二重ループ。`z = (center_y, center_x)`に対応する。
4. **`patch_top = center_y`, `patch_left = center_x`**：構造要素を`z`に置いたときの近傍（`B_z`）を、`padded`配列のどこから切り出せばよいかを、先に変数として明示的に計算するステップ。元画像の座標は`padded`の中では`pad_height`・`pad_width`だけずれているが、「中心からの半径分だけ引く」ことと「ずれた分だけ足す」ことがちょうど打ち消し合い、結果として`padded`上の切り出し開始位置は元の座標`(center_y, center_x)`とそのまま一致する（コード内のコメント参照）。意図が伝わりにくいトリッキーな計算になりやすい箇所なので、結果を一旦変数に代入してから次のステップで使う、という2段階に分けている。
5. **`patch = padded[patch_top : patch_top + kernel_height, patch_left : patch_left + kernel_width]`**：計算済みの開始位置を使って、構造要素と同じ大きさの近傍（`B_z`が指す範囲）を実際に切り出す。行方向に「`patch_top`行目から`patch_top + kernel_height`行目の手前まで」、列方向に「`patch_left`列目から`patch_left + kernel_width`列目の手前まで」を抜き出している。境界処理は畳み込みと同様にゼロパディング（＝画像の外側は背景とみなす）を使う。
6. **`covered_values = patch[structuring_element]`**：[`convolve2d`](../2-convolution/2-1-convolution.md)と同じく「構造要素が指す近傍パッチを切り出し、1つの値に集約する」という共通の型を持つ処理。構造要素がTrueを指す位置の値だけを取り出すことで、正方形以外の形の構造要素にも対応できる汎用的な実装になっている。
7. **`is_foreground = covered_values == 255`**：取り出した値を、前景(255)かどうかの真偽値配列に変換する。
8. **`output[center_y, center_x] = 255 if reduce_fn(is_foreground) else 0`**：`reduce_fn`（`erode`なら`np.all`）の判定結果を、`z=(center_y, center_x)`の新しい画素値として書き込む。
9. **`erode(binary, structuring_element)`**：引数は`_local_reduce`と同じく`binary`（二値画像）・`structuring_element`（構造要素）。`reduce_fn`に`np.all`（全て`True`か＝AND）を固定で渡し、収縮後の`(H, W)`二値画像を返す。これが`B_z ⊆ A`という定義の実装そのもの——構造要素が指す近傍が1つでも背景を含めば、その点は収縮後に背景になる。

## 具体的な計算例

5×5全面が前景（255）の画像を、3×3の正方形構造要素で収縮する。

```
入力:
255 255 255 255 255
255 255 255 255 255
255 255 255 255 255
255 255 255 255 255
255 255 255 255 255

erode結果:
  0   0   0   0   0
  0 255 255 255   0
  0 255 255 255   0
  0 255 255 255   0
  0   0   0   0   0
```

外周1ピクセル分が丸ごと背景(0)に変わり、内側の3×3だけが前景として残る。これは、境界処理のゼロパディングによって画像の外側が背景(0)とみなされるため、外周ピクセルでは3×3の近傍の一部が必ず「外側＝背景」にかかってしまい、`B_z ⊆ A`を満たさなくなることによる（[3-1のSobelで見た「境界処理のアーティファクト」](../3-edge/3-1-sobel.md)と同じ、ゼロパディングに起因する副作用）。[`test_morphology.py`](https://github.com/oguratakayuki/image-processing-lab/blob/main/engine/tests/morphology/test_morphology.py)の`test_erosion_shrinks_solid_square_by_one_pixel_border`で検証している値と一致する。

## 関連

- [1-4. 閾値処理](../1-color/1-4-threshold.md) — 「画像を集合として捉える」という見方の起点
- [2-1. 畳み込み](../2-convolution/2-1-convolution.md) — 「近傍を見て集約する」という操作の型は共通、中身（線形 vs 論理演算）が異なる
- [4-2. Dilation（膨張）](4-2-dilation.md) — Erosionと双対の関係にある操作
- [3-1. Sobelエッジ検出](../3-edge/3-1-sobel.md) — 同じくゼロパディングが境界に副作用を生む例

---

前へ: [← 3-3. Laplacianエッジ検出](../3-edge/3-3-laplacian.md)　｜　次へ: [4-2. Dilation（膨張） →](4-2-dilation.md)
