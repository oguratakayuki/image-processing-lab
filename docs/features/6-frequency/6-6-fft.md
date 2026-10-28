# 6-6. 高速フーリエ変換（1次元, FFT）

[← 6. 周波数領域へ戻る](README.md)　｜　[6-1](6-1-dft.md) · [6-2](6-2-idft.md) · [6-3](6-3-dft-2d.md) · [6-4](6-4-idft-2d.md) · [6-5](6-5-spectrum-visualization.md) · **6-6**

| | |
|---|---|
| 学習テーマ | 分割統治法、計算量`O(N log N)`、バタフライ演算 |
| 実装ファイル | [`engine/src/imglab_engine/frequency/fft.py`](https://github.com/oguratakayuki/image-processing-lab/blob/main/engine/src/imglab_engine/frequency/fft.py) |
| 関数 | `fft_1d()` |
| テスト | [`engine/tests/frequency/test_fft.py`](https://github.com/oguratakayuki/image-processing-lab/blob/main/engine/tests/frequency/test_fft.py) |

## 超ざっくり言うと

[6-1のDFT](6-1-dft.md)と全く同じ結果を、ずっと速く計算する方法。信号を半分の長さに分割し続けることで、計算量を`O(N²)`から`O(N log N)`まで減らす。

## 概念

[6-1のDFT](6-1-dft.md)（`dft_1d`）は、`N`個の出力それぞれについて`N`項の和を計算するため、計算量は`O(N²)`。FFT（高速フーリエ変換）は、同じ結果を、**分割統治法**という考え方で大幅に高速に計算する。信号を「偶数番目のサンプルだけ」と「奇数番目のサンプルだけ」の、半分の長さの2つのグループに分けると、元の`N`点のDFTが、この2つの半分サイズのDFTの組み合わせで計算できる、という性質を利用する。この分割を再帰的に繰り返すことで、計算量が大きく減る。

## 主な用途

- **リアルタイム信号処理**：音声・画像・通信など、大量のデータに対してDFTを繰り返し計算する必要がある場面では、`O(N²)`の素朴なDFTでは実用的な速度が出ない。FFTは、あらゆる実用的なフーリエ変換の実装（音声解析ソフト、画像編集ソフト、通信機器など）で使われている
- **大きな画像・信号ほど効果が大きい**：`N`が大きくなるほど`O(N²)`と`O(N log N)`の差は開くため、高解像度画像や長時間の音声データを扱うほど、FFTの恩恵は大きくなる
- **畳み込みの高速化の土台**：[畳み込み定理](README.md)（空間領域の畳み込み＝周波数領域の掛け算）を実用的な速度で使うには、DFT・IDFTの計算自体もFFTで高速化されている必要がある

## 数学的背景

### 要件

[6-1のDFT](6-1-dft.md)と完全に同じ結果`X_0, X_1, ..., X_{N-1}`を、より少ない計算量で求めたい。

### やるべきこと

長さ`N`の信号`x_0, x_1, ..., x_{N-1}`を、偶数番目（`x_0, x_2, x_4, ...`）と奇数番目（`x_1, x_3, x_5, ...`）の、それぞれ長さ`N/2`の2つのグループに分ける。それぞれのグループだけで計算した（長さ`N/2`の）DFTを`E_k`（偶数番目）・`O_k`（奇数番目）とすると、元の`X_k`は次の式で求まる（`k = 0, ..., N/2-1`）。

$$
X_k = E_k + e^{-2\pi i k / N} O_k
$$

$$
X_{k+N/2} = E_k - e^{-2\pi i k / N} O_k
$$

`E_k`・`O_k`は添字`k`が`N/2`で1周期（周期性は[6-1-dft-wave-count.md](6-1-dft-wave-count.md)と同じ理由）するため、`k=0`から`N/2-1`までの`N/2`個を計算するだけで、`X_k`と`X_{k+N/2}`（合わせて`N`個）の両方が求まる。1回の掛け算・足し算・引き算だけで2つの出力が同時に求まるこの操作は、形が蝶の羽に似ていることから「**バタフライ演算**」と呼ばれる。

**なぜ速くなるのか**。`E_k`・`O_k`自体も、同じ要領で「さらに半分」に分割して計算できる（偶数番目の中の偶数番目・奇数番目、というように）。これを信号の長さが`1`になるまで再帰的に繰り返す。`N`を`2`で割り続けられる回数は`log₂N`回なので、全体の計算量は`O(N log N)`になる（素朴な`O(N²)`との差は、`N`が大きいほど顕著になる——具体的な速度の違いは下記「具体的な計算例」を参照）。

**前提条件**：この分割を`1`になるまで繰り返すには、`N`が`2`で割り切れ続ける必要がある。そのため、この実装は信号の長さが**2の累乗**（`1, 2, 4, 8, 16, ...`）であることを前提とする。

### コードにすると

信号を1つおきに取り出す`signal[0::2]`（偶数番目）・`signal[1::2]`（奇数番目）で2つのグループに分け、それぞれに`fft_1d`自身を再帰的に適用する。長さが`1`になったら、それ以上分割せず信号自身を返す（再帰の終了条件）。2つの半分サイズの結果を、上のバタフライ演算の式でまとめ直す——そのままコードにできる。

## 実装コード

```python
import numpy as np


def fft_1d(signal: np.ndarray) -> np.ndarray:
    # signal: 長さN(2の累乗)の1次元配列。
    # 返却値: 長さNの複素数配列(complex128)。dft_1d(signal)と
    #         完全に同じ結果になる。
    signal = np.asarray(signal, dtype=np.complex128)
    n_samples = len(signal)

    if n_samples == 1:
        return signal.copy()

    if n_samples % 2 != 0:
        raise ValueError("fft_1dは信号の長さが2の累乗であることが前提です")

    even_part = fft_1d(signal[0::2])
    odd_part = fft_1d(signal[1::2])

    half = n_samples // 2
    result = np.zeros(n_samples, dtype=np.complex128)

    for k in range(half):
        angle = -2 * np.pi * k / n_samples
        twiddle = np.exp(1j * angle) * odd_part[k]
        result[k] = even_part[k] + twiddle
        result[k + half] = even_part[k] - twiddle

    return result
```

### 解説

1. **`signal = np.asarray(signal, dtype=np.complex128)`**：`np.asarray`は、リストなどをNumPy配列に変換する関数（既に配列ならそのまま）。再帰呼び出しの中でも複素数として扱えるよう、型を揃えている。
2. **`if n_samples == 1: return signal.copy()`**：再帰の終了条件。長さ`1`の信号のDFTは、和が1項しかないため信号自身になる。
3. **`signal[0::2]`・`signal[1::2]`**：NumPyのスライス構文`開始:終了:ステップ`で、`2`つおきに要素を取り出す（Ruby/PHPには無い書き方）。`[0::2]`は`0`番目から`2`つおき（偶数番目）、`[1::2]`は`1`番目から`2`つおき（奇数番目）。
4. **`even_part = fft_1d(signal[0::2])`・`odd_part = fft_1d(signal[1::2])`**：半分の長さの信号に対して、`fft_1d`自身をもう一度呼び出す（再帰呼び出し）。
5. **`angle = -2 * np.pi * k / n_samples`**：数学的背景の式`exp(-2πik/N)`の角度部分（「回転係数」と呼ばれる）。
6. **`result[k] = even_part[k] + twiddle`・`result[k + half] = even_part[k] - twiddle`**：バタフライ演算そのもの。同じ`even_part[k]`・`twiddle`から、足し算と引き算だけで2つの出力`X_k`・`X_{k+N/2}`を同時に計算している。

## 具体的な計算例

[6-1本編の数値例](6-1-dft.md)と同じ信号で、`fft_1d`が`dft_1d`と完全に一致することを確認する。

```
定数信号[5, 5, 5, 5]:
dft_1d: [20+0j, 0+0j, 0+0j, 0+0j]
fft_1d: [20+0j, 0+0j, 0+0j, 0+0j]
→ 完全に一致
```

**実際の速度差**を、長さ`N=2048`のランダムな信号で比較する。

```
dft_1d (O(N²)):     1.4394秒
fft_1d (O(N log N)): 0.0078秒
→ 結果は一致（誤差1e-6以内）、fft_1dの方が約184倍速い
```

`N`が大きくなるほど、この差はさらに開く（`O(N²)`は`N`が2倍になると計算量が4倍になるが、`O(N log N)`は`N`が2倍になっても計算量はおよそ2倍強にしかならないため）。[`test_fft.py`](https://github.com/oguratakayuki/image-processing-lab/blob/main/engine/tests/frequency/test_fft.py)では、複数の長さ（`2, 4, 8, ..., 64`）で`dft_1d`と完全に一致することを検証している（`np.fft.fft`という、NumPy組み込みの高速フーリエ変換との一致も`test_matches_numpy_fft_reference`で別途確認している——あくまで自前実装`fft_1d`の答え合わせとしてのみ使用し、本体の実装では使っていない）。

## 関連

- [6-1. 離散フーリエ変換（1次元）](6-1-dft.md) — `fft_1d`が高速に計算している、DFTの定義そのもの
- [なぜ波の数と信号の配列の数が一致するか（補足資料）](6-1-dft-wave-count.md) — `E_k`・`O_k`が`N/2`周期になる根拠と同じ、複素指数の周期性

---

前へ: [← 6-5. 周波数領域の可視化](6-5-spectrum-visualization.md)　｜　次へ: なし（現時点で最後の実装済み項目）
