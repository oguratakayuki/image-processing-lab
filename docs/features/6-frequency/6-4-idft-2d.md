# 6-4. 逆離散フーリエ変換（2次元, 2D IDFT）

[← 6. 周波数領域へ戻る](README.md)　｜　[6-1](6-1-dft.md) · [6-2](6-2-idft.md) · [6-3](6-3-dft-2d.md) · **6-4**

| | |
|---|---|
| 学習テーマ | 2次元DFTの逆変換、正規化係数`1/(HW)` |
| 実装ファイル | [`engine/src/imglab_engine/frequency/dft.py`](https://github.com/oguratakayuki/image-processing-lab/blob/main/engine/src/imglab_engine/frequency/dft.py) |
| 関数 | `idft_2d()` |
| テスト | [`engine/tests/frequency/test_idft2d.py`](https://github.com/oguratakayuki/image-processing-lab/blob/main/engine/tests/frequency/test_idft2d.py) |

## 超ざっくり言うと

[6-3の2次元DFT](6-3-dft-2d.md)で「画像→周波数成分」に変換したものを、もう一度「周波数成分→画像」に戻す逆変換。[6-2の1次元IDFT](6-2-idft.md)を画像に拡張したもの。

## 概念

[6-3](6-3-dft-2d.md)では、画像`f(y, x)`を周波数成分`F(v, u)`に変換する方法を学んだ。2次元IDFTはその逆方向——`F(v, u)`から`f(y, x)`を復元する変換。[6-2の1次元IDFT](6-2-idft.md)が「周波数ごとの測定結果を、逆向きに組み立て直して元の信号を再現する」操作だったのと同じように、2次元IDFTは縦・横2方向分の測定結果を同時に組み立て直す。

## 主な用途

- **周波数領域での加工後に画像へ戻す**：[6-3の主な用途](6-3-dft-2d.md)で触れた画像圧縮・周波数領域でのフィルタリングは、「DFTで周波数領域に変換→特定の`(u, v)`成分を削る・強調する→IDFTで画像に戻す」という3段構成が基本形。2次元IDFTはその最後のステップ
- **往復変換の正しさの検証**：2次元DFTの実装が正しいかどうかを、「DFT→IDFTで元の画像に戻るか」という往復テストで確認できる（本ページの具体例もこの方法を使う）

## 数学的背景

### 要件

[6-3の定義式](6-3-dft-2d.md)で求めた周波数成分`F(0,0), F(0,1), ..., F(H-1,W-1)`から、元の画像`f(y, x)`（高さ`H`、幅`W`）を復元したい。

### やるべきこと

2次元IDFTは次の式で定義される。

$$
f(y, x) = \frac{1}{HW} \sum_{v=0}^{H-1} \sum_{u=0}^{W-1} F(v, u) \cdot e^{2\pi i (v y / H + u x / W)}
$$

[6-3の定義式](6-3-dft-2d.md)と見比べると、[6-2の1次元IDFT](6-2-idft.md)と同じ2つの違いがある。

1. **指数の符号が`-`から`+`に反転**している
2. 先頭に**`1/(HW)`という正規化係数**が付く（1次元の`1/N`を、縦・横2方向分`H`と`W`に拡張したもの）

**なぜ1次元IDFTを2回計算するだけで済むのか**。[6-3で2次元DFTを分離した](6-3-dft-2d.md)のと同じ理由で、2次元IDFTの式も`u`についての和と`v`についての和に分解できる。

$$
f(y, x) = \frac{1}{H} \sum_{v=0}^{H-1} e^{2\pi i v y / H} \cdot \frac{1}{W} \sum_{u=0}^{W-1} F(v, u) \cdot e^{2\pi i u x / W}
$$

内側の`Σ`（`1/W`の正規化込み）は、周波数成分の`v`行目だけを取り出して横方向にかけた1次元IDFT、つまり[`idft_1d()`](6-2-idft.md)そのもの。外側の`Σ`（`1/H`の正規化込み）は、その結果の各列を取り出して縦方向にかけた1次元IDFT。`idft_1d()`を2段階で呼び出すだけで、`1/W`と`1/H`の正規化が自動的に掛け合わさり、正しい`1/(HW)`になる。

### コードにすると

[`idft_1d()`](6-2-idft.md)を、まず周波数成分の各行（`v`ごと、横方向）に適用し、その結果に対してもう一度各列（`u`ごと、縦方向）に適用する——[`dft_2d()`](6-3-dft-2d.md)と全く同じ2段階の構造で、使う関数が`dft_1d`から`idft_1d`に変わるだけでそのままコードにできる。

## 実装コード

```python
import numpy as np


def idft_2d(spectrum: np.ndarray) -> np.ndarray:
    # spectrum: 形状(H, W)の複素数配列。spectrum[v, u]が、縦方向の
    #           周波数v・横方向の周波数uの成分。
    # 返却値: 形状(H, W)の複素数配列(complex128)。実数画像のDFT
    #         結果を渡した場合、理論上は虚部が0になる。
    height, width = spectrum.shape

    rows_transformed = np.zeros((height, width), dtype=np.complex128)
    for y in range(height):
        rows_transformed[y, :] = idft_1d(spectrum[y, :])

    result = np.zeros((height, width), dtype=np.complex128)
    for x in range(width):
        result[:, x] = idft_1d(rows_transformed[:, x])

    return result
```

### 解説

1. **`for y in range(height): rows_transformed[y, :] = idft_1d(spectrum[y, :])`**：[`dft_2d()`の第1段階](6-3-dft-2d.md)と同じ構造。周波数成分の各行に対して[`idft_1d()`](6-2-idft.md)を適用している——分離した式の内側の`Σ`（`u`についての和、`1/W`の正規化込み）に対応する。
2. **`for x in range(width): result[:, x] = idft_1d(rows_transformed[:, x])`**：第1段階の結果の各列に対して`idft_1d()`をもう一度適用している——分離した式の外側の`Σ`（`v`についての和、`1/H`の正規化込み）に対応する。

## 具体的な計算例

[6-3の定数画像の例](6-3-dft-2d.md)で、DFT→IDFTの往復を確認する。

```
元の画像（4×4、全画素5）:
5 5 5 5
5 5 5 5
5 5 5 5
5 5 5 5

DFT: F(0,0)=80+0j、他は全て0

IDFT後（実部）:
5 5 5 5
5 5 5 5
5 5 5 5
5 5 5 5
```

`IDFT後`が`元の画像`と完全に一致している。[`test_idft2d.py`](https://github.com/oguratakayuki/image-processing-lab/blob/main/engine/tests/frequency/test_idft2d.py)の`test_idft_2d_recovers_constant_image`で検証している値と一致する。

ランダムな画像でも確認する（`3×3`）。

```
元の画像:
 0.1257 -0.1321  0.6404
 0.1049 -0.5357  0.3616
 1.3040  0.9471 -0.7037

IDFT後（実部）:
 0.1257 -0.1321  0.6404
 0.1049 -0.5357  0.3616
 1.3040  0.9471 -0.7037
```

最大誤差は`4.4×10^-16`（浮動小数点演算の丸め誤差の範囲）で、実質的に完全一致している（`np.fft.ifft2`という、NumPy組み込みの2次元逆高速フーリエ変換との一致も、[`test_idft2d.py`](https://github.com/oguratakayuki/image-processing-lab/blob/main/engine/tests/frequency/test_idft2d.py)の`test_idft_2d_matches_numpy_ifft2_reference`で別途確認している——あくまで自前実装`idft_2d`の答え合わせとしてのみ使用し、本体の実装では使っていない）。横方向だけに変化する画像（[6-3の2つ目の例](6-3-dft-2d.md)）でも、DFT→IDFTの往復で元に戻ることを`test_idft_2d_recovers_horizontal_only_pattern`で確認済み。

## 関連

- [6-3. 離散フーリエ変換（2次元）](6-3-dft-2d.md) — このページが逆変換している2次元DFTの定義・`dft_2d()`
- [6-2. 逆離散フーリエ変換（1次元）](6-2-idft.md) — このページが拡張している1次元IDFTの定義・`idft_1d()`

---

前へ: [← 6-3. 離散フーリエ変換（2次元）](6-3-dft-2d.md)　｜　次へ: なし（現時点で最後の実装済み項目）
