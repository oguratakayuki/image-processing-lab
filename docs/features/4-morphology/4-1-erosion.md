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
    kh, kw = structuring_element.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(
        binary, ((pad_h, pad_h), (pad_w, pad_w)), mode="constant", constant_values=0
    )

    height, width = binary.shape
    output = np.zeros((height, width), dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            patch = padded[y : y + kh, x : x + kw]
            covered = patch[structuring_element]
            output[y, x] = 255 if reduce_fn(covered == 255) else 0

    return output


def erode(binary: np.ndarray, structuring_element: np.ndarray) -> np.ndarray:
    return _local_reduce(binary, structuring_element, np.all)
```

### 解説

0. **前提**：`erode()`の引数`binary`は、[1-1のGrayscale変換](../1-color/1-1-grayscale.md)と[1-4の閾値処理](../1-color/1-4-threshold.md)を経て、すでに`0`（背景）か`255`（前景）の2値だけになった画像であることを前提にしている。この関数自体はRGBの色情報を一切受け取らず、見ることもない。
1. **`square_structuring_element(size)`**：`size × size`の全てTrueな真偽値配列。`B`の最も単純な形（正方形）。
2. **`_local_reduce`**：[`convolve2d`](../2-convolution/2-1-convolution.md)と同じく「構造要素が指す近傍パッチを切り出し、1つの値に集約する」という共通の型を持つヘルパー。境界処理も畳み込みと同様にゼロパディング（＝画像の外側は背景とみなす）を使う。`covered = patch[structuring_element]`で構造要素がTrueを指す位置の値だけを取り出すことで、正方形以外の形の構造要素にも対応できる汎用的な実装になっている。
3. **`reduce_fn(covered == 255)`**：`covered`の各要素が前景(255)かどうかの真偽値配列に変換してから`reduce_fn`に渡す。
4. **`erode`**：`reduce_fn`に`np.all`（全て`True`か＝AND）を渡す。これが`B_z ⊆ A`という定義の実装そのもの——構造要素が指す近傍が1つでも背景を含めば、その点は収縮後に背景になる。

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
