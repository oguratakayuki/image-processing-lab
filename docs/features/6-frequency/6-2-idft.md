# 6-2. 逆離散フーリエ変換（1次元, Inverse DFT）

[← 6. 周波数領域へ戻る](README.md)　｜　[6-1](6-1-dft.md) · **6-2** · [6-3](6-3-dft-2d.md)

| | |
|---|---|
| 学習テーマ | DFTの逆変換、正規化係数`1/N` |
| 実装ファイル | [`engine/src/imglab_engine/frequency/dft.py`](https://github.com/oguratakayuki/image-processing-lab/blob/main/engine/src/imglab_engine/frequency/dft.py) |
| 関数 | `idft_1d()` |
| テスト | [`engine/tests/frequency/test_idft.py`](https://github.com/oguratakayuki/image-processing-lab/blob/main/engine/tests/frequency/test_idft.py) |

## 超ざっくり言うと

[6-1のDFT](6-1-dft.md)で「信号→周波数成分」に変換したものを、もう一度「周波数成分→信号」に戻す逆変換。DFT→IDFTと順番に適用すると、元の信号にぴったり戻る。

## 概念

[6-1](6-1-dft.md)では、信号`x_n`を周波数成分`X_k`に変換する方法を学んだ。IDFT（逆離散フーリエ変換）はその逆方向——`X_k`から`x_n`を復元する変換。[DFTを「物差しで信号を測る」操作として理解した](6-1-dft-ruler-analogy.md)のと同じように、IDFTは「周波数ごとの測定結果を、逆向きに組み立て直して元の信号を再現する」操作として理解できる。

## 主な用途

- **周波数領域での加工後に画像/信号へ戻す**：[6-1の主な用途](6-1-dft.md)で触れたノイズ除去・画像圧縮などは、「DFTで周波数領域に変換→不要な周波数成分を削る・強調する→IDFTで空間領域に戻す」という3段構成が基本形。IDFTはその最後のステップ
- **往復変換の正しさの検証**：DFTの実装が正しいかどうかを、「DFT→IDFTで元の信号に戻るか」という往復テストで確認できる（本ページの具体例もこの方法を使う）

## 数学的背景

### 要件

[6-1の`(式1)`](6-1-dft.md)で求めた周波数成分`X_0, X_1, ..., X_{N-1}`から、元の信号`x_0, x_1, ..., x_{N-1}`を復元したい。

### やるべきこと

IDFTは次の式で定義される。

$$
x_n = \frac{1}{N} \sum_{k=0}^{N-1} X_k \cdot e^{2\pi i k n / N}, \quad n = 0, \ldots, N-1
$$

[`(式1)`の定義](6-1-dft.md)と見比べると、2つの違いがある。

1. **指数の符号が`-`から`+`に反転**している
2. 先頭に**`1/N`という正規化係数**が付く

**なぜ符号が反転するのか**。DFTは「`x_n`を、周波数`k`の[物差し`exp(-2πikn/N)`で測って](6-1-dft-ruler-analogy.md)`X_k`を得る」という操作だった。IDFTはその逆操作にあたるため、測るときに使った物差しの「逆回転」、つまり符号を反転させた`exp(+2πikn/N)`を使う。

**なぜ`1/N`が必要なのか**。[6-1で確認した通り](6-1-dft-ruler-analogy.md)、周波数`k`の物差しと信号が一致しているとき、DFTの`Σ`の結果は`N`個の同じ向きのベクトルが積み上がった値になる（ベクトル1個分の大きさの、およそ`N`倍）。IDFTでは、これをもう一度`N`個の周波数ぶん足し合わせるため、何も調整しないと値がさらに`N`倍に膨らんでしまう。`1/N`を掛けることで、この膨らみを打ち消し、元の信号の大きさに正しく戻す。

### コードにすると

`n=0`から`N-1`まで、それぞれの`n`について、周波数成分の全項`k=0`から`N-1`までを`exp(+2πikn/N)`で重み付けして足し合わせ、最後に`N`で割る——[`dft_1d()`](6-1-dft.md)とほぼ同じ2重ループの構造で、符号と正規化だけが異なる形でそのままコードにできる。

## 実装コード

```python
import numpy as np


def idft_1d(spectrum: np.ndarray) -> np.ndarray:
    # spectrum: 長さNの複素数配列。spectrum[k]が周波数kの成分。
    # 返却値: 長さNの複素数配列(complex128)。実数信号のDFT結果を
    #         渡した場合、理論上は虚部が0になる。
    n_samples = len(spectrum)

    result = np.zeros(n_samples, dtype=np.complex128)

    for n in range(n_samples):
        total = 0.0 + 0.0j
        for k in range(n_samples):
            angle = 2 * np.pi * k * n / n_samples
            total += spectrum[k] * np.exp(1j * angle)

        result[n] = total / n_samples

    return result
```

### 解説

1. **`for n in range(n_samples): for k in range(n_samples):`**：[`dft_1d()`](6-1-dft.md)と役割が入れ替わっている点に注意。DFTは「周波数`k`ごとにループし、内側で信号の`n`を全部足す」だったが、IDFTは「サンプル位置`n`ごとにループし、内側で周波数成分`k`を全部足す」。
2. **`angle = 2 * np.pi * k * n / n_samples`**：[`dft_1d`の`angle`](6-1-dft.md)と符号が逆（`+`）であることに注意。DFTで使った物差しの「逆回転」を表す。
3. **`result[n] = total / n_samples`**：`1/N`で正規化する。DFTで積み上がった分の大きさを打ち消し、元の信号の大きさに戻すための係数。

## 具体的な計算例

[6-1の定数信号の例](6-1-dft.md)で、DFT→IDFTの往復を確認する。

```
元の信号: [5, 5, 5, 5]
DFT:     [20+0j, 0+0j, 0+0j, 0+0j]
IDFT後:   [5-0j, 5-0j, 5+0j, 5+0j]
```

`IDFT後`が、丸め誤差を除いて`元の信号`と完全に一致している。[`test_idft.py`](https://github.com/oguratakayuki/image-processing-lab/blob/main/engine/tests/frequency/test_idft.py)の`test_idft_recovers_constant_signal`で検証している値と一致する。

ランダムな信号でも確認する（`N=10`）。

```
元の信号: [0.1257, -0.1321, 0.6404, 0.1049, -0.5357, 0.3616, 1.304, 0.9471, -0.7037, -1.2654]
IDFT後:   [0.1257, -0.1321, 0.6404, 0.1049, -0.5357, 0.3616, 1.304, 0.9471, -0.7037, -1.2654]
```

小数点以下5桁まで完全に一致する（`np.fft.ifft`という、NumPy組み込みの逆高速フーリエ変換との一致も、[`test_idft.py`](https://github.com/oguratakayuki/image-processing-lab/blob/main/engine/tests/frequency/test_idft.py)の`test_idft_matches_numpy_ifft_reference`で別途確認している——あくまで自前実装`idft_1d`の答え合わせとしてのみ使用し、本体の実装では使っていない）。どんな信号でも、DFT→IDFTの往復で元に戻ることが確認できた。

## 関連

- [6-1. 離散フーリエ変換（1次元）](6-1-dft.md) — `(式1)`の定義式、このページの定義式はその逆変換
- [物差し（基準波）で信号を測るという見方（補足資料）](6-1-dft-ruler-analogy.md) — IDFTが「逆回転した物差しで組み立て直す」操作であることの直感

---

前へ: [← 6-1. 離散フーリエ変換（1次元）](6-1-dft.md)　｜　次へ: [6-3. 離散フーリエ変換（2次元） →](6-3-dft-2d.md)
