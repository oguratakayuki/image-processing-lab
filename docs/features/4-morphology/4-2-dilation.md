# 4-2. Dilation（膨張）

[← 4. 二値画像処理へ戻る](README.md)　｜　[4-1](4-1-erosion.md) · **4-2** · [4-3](4-3-opening.md) · [4-4](4-4-closing.md)

| | |
|---|---|
| 学習テーマ | 集合演算（OR）、構造要素 |
| 実装ファイル | [`engine/src/imglab_engine/morphology/morphology.py`](https://github.com/oguratakayuki/image-processing-lab/blob/main/engine/src/imglab_engine/morphology/morphology.py) |
| 関数 | `dilate()` |
| テスト | [`engine/tests/morphology/test_morphology.py`](https://github.com/oguratakayuki/image-processing-lab/blob/main/engine/tests/morphology/test_morphology.py) |

## 概念

[4-1のErosion（収縮）](4-1-erosion.md)とちょうど逆向きの操作。構造要素を画像の上でずらしながら、「構造要素が前景と少しでも重なる場所」を全て新しい前景にする。前景の領域が外側に向かって膨らむ。

## 数学的背景

```
A ⊕ B = {z : B̂_z ∩ A ≠ ∅}
```

`B̂`は構造要素`B`を原点対称に反転（180度回転）させた集合。「構造要素（を反転したもの）を点`z`に置いたとき、前景`A`と1点でも重なるような点`z`を集めた集合」という意味。実装上は、各点`z`について「構造要素が指す近傍のどれか1つでも前景かどうか（論理OR）」を判定することと同値になる。

[2-1の畳み込みで出てきた「カーネルの反転」](../2-convolution/2-1-convolution.md)と同じ`B̂`が、ここにも登場する。ただし今回使う構造要素（正方形）は原点対称（`B̂ = B`）なので、実装上は反転してもしなくても結果は変わらない——畳み込みで対称カーネルを使うと反転の有無が結果に影響しなかったのと同じ理屈。

## 実装コード

```python
def dilate(binary: np.ndarray, structuring_element: np.ndarray) -> np.ndarray:
    return _local_reduce(binary, structuring_element, np.any)
```

### 解説

[4-1の`_local_reduce`](4-1-erosion.md)に`reduce_fn=np.any`（どれか1つでも`True`か＝OR）を渡すだけ。`erode`との違いは`np.all`か`np.any`かの1点のみで、`A ⊕ B = {z : B̂_z ∩ A ≠ ∅}`という定義における「∩ ≠ ∅（1点でも重なる）」がそのまま`np.any`に対応する。

## 具体的な計算例

5×5の画像で、中心1点だけが前景（255）の画像を3×3の正方形構造要素で膨張する。

```
入力:
  0   0   0   0   0
  0   0   0   0   0
  0   0 255   0   0
  0   0   0   0   0
  0   0   0   0   0

dilate結果:
  0   0   0   0   0
  0 255 255 255   0
  0 255 255 255   0
  0 255 255 255   0
  0   0   0   0   0
```

中心の1点を中心とした3×3の正方形に広がった。これは[4-1のErosionの計算例](4-1-erosion.md)とちょうど逆の形になっている（[`test_morphology.py`](https://github.com/oguratakayuki/image-processing-lab/blob/main/engine/tests/morphology/test_morphology.py)の`test_dilation_grows_single_pixel_into_3x3_square`で検証している値と一致する）。

## Erosionとの双対性

Erosionと膨張は、補集合を取る操作に関して**双対（dual）**の関係にある。

```
(A ⊖ B)^c = A^c ⊕ B̂
```

（`A`を収縮してから補集合を取った結果は、`A`の補集合を膨張させた結果と一致する）。これはブール代数の**ド・モルガンの法則**`(P ∧ Q)^c = P^c ∨ Q^c`の集合演算版が、そのまま画像処理に現れたもの。「収縮＝AND」「膨張＝OR」という対応を思い出すと、ANDの否定がORの否定の組み合わせになるド・モルガンの法則と同じ構造であることが分かる。

4×4の画像`A`で実際に検証する（`255 - img`で補集合、構造要素は正方形で`B̂ = B`）。

```
A (入力):
  0 255   0 255
255 255   0   0
  0   0 255 255
255   0 255   0

A ⊖ B (erode):
  0   0   0   0
  0   0   0   0
  0   0   0   0
  0   0   0   0

(A ⊖ B)の補集合:
255 255 255 255
255 255 255 255
255 255 255 255
255 255 255 255

A^c (補集合):
255   0 255   0
  0   0 255 255
255 255   0   0
  0 255   0 255

A^c ⊕ B (dilate of complement):
255 255 255 255
255 255 255 255
255 255 255 255
255 255 255 255
```

`(A ⊖ B)の補集合`と`A^c ⊕ B`が完全に一致している（この例ではどちらも全面255になったが、これは`A`がまばらで3×3の近傍が必ずどこかに前景を含むため。[`test_morphology.py`](https://github.com/oguratakayuki/image-processing-lab/blob/main/engine/tests/morphology/test_morphology.py)の`test_erosion_dilation_duality`では、よりランダムな画像でも一致することを検証している）。

## 関連

- [4-1. Erosion（収縮）](4-1-erosion.md) — 双対の関係にある操作、`_local_reduce`を共用
- [2-1. 畳み込み](../2-convolution/2-1-convolution.md) — `B̂`（構造要素の反転）はカーネルの反転と同じ考え方
- [4-3. Opening（オープニング）](4-3-opening.md) — Erosion→Dilationの順に合成した操作
- [4-4. Closing（クロージング）](4-4-closing.md) — Dilation→Erosionの順に合成した操作

---

前へ: [← 4-1. Erosion（収縮）](4-1-erosion.md)　｜　次へ: [4-3. Opening（オープニング） →](4-3-opening.md)
