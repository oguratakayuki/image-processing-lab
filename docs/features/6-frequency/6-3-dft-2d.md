# 6-3. 離散フーリエ変換（2次元, 2D DFT）

[← 6. 周波数領域へ戻る](README.md)　｜　[6-1](6-1-dft.md) · [6-2](6-2-idft.md) · **6-3**

| | |
|---|---|
| 学習テーマ | 2次元DFT、分離可能性（1次元DFTを2回で計算する） |
| 実装ファイル | [`engine/src/imglab_engine/frequency/dft.py`](https://github.com/oguratakayuki/image-processing-lab/blob/main/engine/src/imglab_engine/frequency/dft.py) |
| 関数 | `dft_2d()` |
| テスト | [`engine/tests/frequency/test_dft2d.py`](https://github.com/oguratakayuki/image-processing-lab/blob/main/engine/tests/frequency/test_dft2d.py) |

## 超ざっくり言うと

[6-1の1次元DFT](6-1-dft.md)を画像（2次元の数値の並び）に拡張したもの。横方向の変化パターンと縦方向の変化パターンを、同時に周波数として数値化する。

## 概念

[6-1](6-1-dft.md)では、1次元の信号（数値の並び）を周波数成分に変換した。画像は、横方向（`x`方向）と縦方向（`y`方向）の2方向に広がる2次元の数値の並びなので、周波数も2方向分——横方向の変化の速さ`u`と、縦方向の変化の速さ`v`——を同時に測る必要がある。2次元DFTは、画像の中に「横方向に`u`回、縦方向に`v`回」変化するパターンがどれだけ含まれているかを、`(u, v)`の組ごとに数値化する。

## 主な用途

- **画像圧縮**：JPEG等の画像圧縮は、2次元DFTに近い変換（離散コサイン変換, DCT）を使い、横・縦それぞれの方向の変化パターンのうち「人間の目が感知しにくい高周波成分」を間引く
- **周波数領域でのフィルタリング**：ノイズ除去やぼかし・先鋭化を、空間領域でのカーネル畳み込みではなく、2次元DFTで周波数領域に変換してから特定の`(u, v)`成分を抑える・強調するという形で行える
- **周期的なパターンの検出**：布地の織り目、網目模様、縞模様など、画像中の周期的なテクスチャの向きと細かさを、スペクトラムのピークの位置から読み取れる

## 数学的背景

### 要件

高さ`H`、幅`W`の画像`f(y, x)`について、「横方向の周波数`u`・縦方向の周波数`v`の成分がどれだけ含まれているか」を、`u=0`から`W-1`、`v=0`から`H-1`まで、全ての組み合わせについて数値化したい。

### やるべきこと

2次元DFTは次の式で定義される。

$$
F(v, u) = \sum_{y=0}^{H-1} \sum_{x=0}^{W-1} f(y, x) \cdot e^{-2\pi i (v y / H + u x / W)}
$$

[6-1の定義式](6-1-dft.md)と見比べると、指数部分が`v・y/H`（縦方向の回転）と`u・x/W`（横方向の回転）の2つの和になっている以外は同じ形をしている。

**なぜ1次元DFTを2回計算するだけで済むのか（分離可能性）**。上の2重`Σ`は、`x`についての和と`y`についての和に分解できる。

$$
F(v, u) = \sum_{y=0}^{H-1} e^{-2\pi i v y / H} \sum_{x=0}^{W-1} f(y, x) \cdot e^{-2\pi i u x / W}
$$

内側の`Σ`（`x`についての和）は、画像の`y`行目だけを取り出して横方向にかけた1次元DFT、つまり[`dft_1d()`](6-1-dft.md)そのもの。外側の`Σ`（`y`についての和）は、その結果の各列を取り出して縦方向にかけた1次元DFT。つまり2次元DFTは、**「各行に1次元DFTをかけ、その結果の各列にもう一度1次元DFTをかける」という2段階の処理**で計算できる。これは[2-2のガウシアンカーネルの分離可能性](../2-convolution/2-2-kernel-presets.md)（2次元のカーネルを、1次元×1次元の掛け合わせに分解できるという性質）と同じ発想。

素朴に2重`Σ`をそのまま計算すると計算量は`O(H²W²)`だが、分離して1次元DFTを使い回すと`O(HW(H+W))`に抑えられる。

### コードにすると

`dft_1d()`を、まず画像の各行（`y`ごと、横方向）に適用し、その結果に対してもう一度各列（`x`ごと、縦方向）に適用する——新しい2重`Σ`のループを書く必要はなく、既存の`dft_1d()`を2段階で呼び出すだけでそのままコードにできる。

## 実装コード

```python
import numpy as np


def dft_2d(image: np.ndarray) -> np.ndarray:
    # image: 形状(H, W)の2次元配列(Hが高さ=行数、Wが幅=列数)。
    # 返却値: 形状(H, W)の複素数配列(complex128)。
    #         result[v, u]が、縦方向の周波数v・横方向の周波数uの成分。
    height, width = image.shape

    rows_transformed = np.zeros((height, width), dtype=np.complex128)
    for y in range(height):
        rows_transformed[y, :] = dft_1d(image[y, :])

    result = np.zeros((height, width), dtype=np.complex128)
    for x in range(width):
        result[:, x] = dft_1d(rows_transformed[:, x])

    return result
```

### 解説

1. **`height, width = image.shape`**：NumPy配列の`.shape`は、各次元の大きさをタプルで返す（2次元配列なら`(行数, 列数)`）。ここでは画像の高さと幅を取り出している。
2. **`rows_transformed[y, :] = dft_1d(image[y, :])`**：`image[y, :]`は「`y`行目の全列」を取り出すNumPyのスライス構文（先頭の`y`が行、`:`が「列は全部」という意味。Ruby/PHPの配列には無い書き方）。各行に対して[`dft_1d()`](6-1-dft.md)を適用し、結果を同じ`y`行目に書き戻している——これが分離可能性の式の内側の`Σ`（`x`についての和）に対応する。
3. **`result[:, x] = dft_1d(rows_transformed[:, x])`**：`rows_transformed[:, x]`は「`x`列目の全行」を取り出すスライス構文（先頭の`:`が「行は全部」、`x`が列という意味）。第1段階の結果の各列に対して`dft_1d()`をもう一度適用している——これが分離可能性の式の外側の`Σ`（`y`についての和）に対応する。

## 具体的な計算例

**定数画像**（`4×4`、全画素`5`）の2次元DFTを計算する。

```
入力:
5 5 5 5
5 5 5 5
5 5 5 5
5 5 5 5

F(0,0) = 80+0j
他の(u,v)は全て 0
```

[6-1の定数信号の例](6-1-dft.md)と同じく、画像全体に変化が無いため、周波数`(0, 0)`（直流成分）だけに値が集中する。直流成分の値`80`は、画素値の総和（`5×16=80`）に一致する。[`test_dft2d.py`](https://github.com/oguratakayuki/image-processing-lab/blob/main/engine/tests/frequency/test_dft2d.py)の`test_constant_image_has_only_dc_component`で検証している値と一致する。

**横方向だけに変化する画像**（`H=W=8`、どの行も`cos(2πx/8)`）も確認する。

```
画像の各行（丸めて表示）: [1, 0.7071, 0, -0.7071, -1, -0.7071, 0, 0.7071]
（8行とも同じパターンの繰り返し＝縦方向には変化が無い）

非ゼロのスペクトラム:
(u=1, v=0): |F|=32.0000
(u=7, v=0): |F|=32.0000
（他の(u,v)は全て0）
```

横方向にちょうど1周期収まる余弦波が、縦方向には全く変化していないため、ピークは横方向の周波数`u=1`（とその鏡映`u=7`）にだけ現れ、縦方向の周波数`v`は常に`0`（直流、変化なし）になる。[`test_dft2d.py`](https://github.com/oguratakayuki/image-processing-lab/blob/main/engine/tests/frequency/test_dft2d.py)の`test_horizontal_only_pattern_peaks_at_u1_v0`で検証している値と一致する（縦方向だけに変化する画像を使った`test_vertical_only_pattern_peaks_at_u0_v1`では、逆に`u=0・v=1`にピークが現れることも確認済み——`np.fft.fft2`という、NumPy組み込みの2次元高速フーリエ変換との一致も`test_matches_numpy_fft2_reference`で別途確認している）。

## 関連

- [6-1. 離散フーリエ変換（1次元）](6-1-dft.md) — このページが拡張している1次元DFTの定義・`dft_1d()`
- [6-2. 逆離散フーリエ変換（1次元）](6-2-idft.md) — 1次元での順変換・逆変換の対応関係
- [2-2. カーネルプリセット](../2-convolution/2-2-kernel-presets.md) — ガウシアンカーネルの分離可能性（同じ「2次元を1次元×1次元に分解する」発想）

---

前へ: [← 6-2. 逆離散フーリエ変換（1次元）](6-2-idft.md)　｜　次へ: なし（現時点で最後の実装済み項目）
