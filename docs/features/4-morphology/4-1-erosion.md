# 4-1. Erosion（収縮）

[← 4. 二値画像処理へ戻る](README.md)　｜　**4-1** · [4-2](4-2-dilation.md) · [4-3](4-3-opening.md) · [4-4](4-4-closing.md)

| | |
|---|---|
| 学習テーマ | 集合演算（AND）、構造要素 |
| 実装ファイル | [`engine/src/imglab_engine/morphology/morphology.py`](https://github.com/oguratakayuki/image-processing-lab/blob/main/engine/src/imglab_engine/morphology/morphology.py) |
| 関数 | `erode()`, `square_structuring_element()` |
| テスト | [`engine/tests/morphology/test_morphology.py`](https://github.com/oguratakayuki/image-processing-lab/blob/main/engine/tests/morphology/test_morphology.py) |

## 概念

[4. 二値画像処理](README.md)では、二値画像（前景255・背景0）をピクセル値の並びとしてではなく、**「前景ピクセルの座標の集合」**として捉える。Erosion（収縮）は、この集合を内側に向かって縮める操作。小さな図形（構造要素）を画像の上でずらしながら、「構造要素の形がすっぽり前景に収まる場所だけ」を新しい前景として残す。

## 数学的背景

二値画像を集合として表すと：

```
A = {(x, y) : I(x, y) = 255}
```

小さな図形`B`（構造要素, structuring element）を使い、`B`を点`z`だけ平行移動した集合を`B_z`と書く。Erosionの定義：

```
A ⊖ B = {z : B_z ⊆ A}
```

「構造要素を点`z`に置いたとき、構造要素の形が完全に`A`（前景）に収まるような点`z`だけを集めた集合」という意味。実装上は、各点`z`について「構造要素が指す近傍が全て前景かどうか（論理AND）」を判定することと同値になる。

これは[2. 畳み込み](../2-convolution/README.md)で学んだ「近傍を見て新しい値を決める」という操作の型と共通するが、中身が異なる。畳み込みは近傍の値の**重み付き和（線形演算）**だったのに対し、モルフォロジー演算は近傍の値の**論理AND/OR（非線形の集合演算）**である。

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
