# 6-1. 離散フーリエ変換（1次元, Discrete Fourier Transform）

[← 6. 周波数領域へ戻る](README.md)　｜　**6-1**

| | |
|---|---|
| 学習テーマ | 複素指数関数、オイラーの公式、周波数という見方 |
| 実装ファイル | [`engine/src/imglab_engine/frequency/dft.py`](https://github.com/oguratakayuki/image-processing-lab/blob/main/engine/src/imglab_engine/frequency/dft.py) |
| 関数 | `dft_1d()` |
| テスト | [`engine/tests/frequency/test_dft.py`](https://github.com/oguratakayuki/image-processing-lab/blob/main/engine/tests/frequency/test_dft.py) |

## 超ざっくり言うと

数値の並び（信号）を、「どの周波数の波が、どれだけの強さで含まれているか」という全く別の見方に変換する操作。

## 概念

[1〜5のグループ](../../../FEATURES.md)はすべて、ピクセル値や座標を直接扱ってきた（これを**空間領域**での処理と呼ぶ）。離散フーリエ変換（DFT）は、信号（画像なら1行分の輝度の並びなども含む。[具体例は補足資料を参照](6-1-dft-image-row-example.md)）を、**周波数領域**という全く別の見方に変換する。「この信号の中に、ゆっくり変化する成分（低周波）がどれだけあるか、激しく振動する成分（高周波）がどれだけあるか」を数値化する（振幅・周波数・位相という言葉の意味は[波と周波数の基礎（補足資料）](6-1-dft-waves-and-frequency.md)を参照）。ノイズは高周波成分に現れやすく、エッジ・輪郭は特定の周波数帯に集中しやすいため、数値化することで「ノイズだけを狙って抑える」「畳み込みを高速な掛け算に置き換える」といった、空間領域のままでは扱いにくい処理が可能になる（詳しくは下記「主な用途」を参照）。

## 主な用途

- **ノイズ除去**：ノイズは高周波成分に現れやすいため、周波数領域で高周波を抑えてから空間領域に戻す、という処理の基礎になる
- **畳み込みの高速化**：[畳み込み定理](README.md)により、大きな画像・大きなカーネルの畳み込みを、周波数領域での単純な掛け算に置き換えて高速化できる（[2-1で伏線として触れた内容](../2-convolution/2-1-convolution.md)）
- **画像圧縮**：JPEG等の画像圧縮は、周波数領域で「人間の目が感知しにくい高周波成分」を間引くという考え方を使っている

## 数学的背景

### 要件

長さ`N`の信号`x_0, x_1, ..., x_{N-1}`について、「周波数`k`の成分がどれだけ含まれているか」を、`k=0`から`N-1`まで、全ての周波数について数値化したい。

### やるべきこと

離散フーリエ変換（DFT）は次の式で定義される。

$$
X_k = \sum_{n=0}^{N-1} x_n \cdot e^{-2\pi i k n / N}, \quad k = 0, \ldots, N-1
$$

`X_k`は複素数で（複素数そのものについては[複素数とは（補足資料）](6-1-dft-complex-numbers.md)を参照）、「周波数`k`の成分の強さと位相（波のずれ）」を同時に表す。`k`は「信号全体の長さ`N`の中に、波がちょうど`k`回収まる」ような周波数に対応する（`k=0`は変化の無い直流成分、`k`が大きいほど細かく振動する成分）。

**なぜ複素数が必要なのか**。オイラーの公式（[導出・直感は補足資料を参照](6-1-dft-euler-formula.md)）

```
exp(iθ) = cos(θ) + i・sin(θ)
```

により、

```
exp(-2πi・k・n/N) = cos(2π・k・n/N) - i・sin(2π・k・n/N)
```

という、1つの複素数の中にcos成分（実部）とsin成分（虚部）の両方が同時に入っている。信号の中の波を「どれだけcosに似ているか」「どれだけsinに似ているか」の2方向から同時に測ると考えると、複素数1つでこの2方向の情報（振幅と位相）をまとめて表現できる。

### コードにすると

`k=0`から`N-1`まで、それぞれの`k`について、信号の全項`n=0`から`N-1`までを`exp(-2πi・k・n/N)`で重み付けして足し合わせる——2重ループでそのままコードにできる。計算量は`O(N²)`（`N`個の出力それぞれに`N`項の和が必要なため）。これをより高速に計算する高速フーリエ変換（FFT, `O(N log N)`）は今後のトピックで扱う。

## 実装コード

```python
import numpy as np


def dft_1d(signal: np.ndarray) -> np.ndarray:
    # signal: 長さNの1次元配列。
    # 返却値: 長さNの複素数配列(complex128)。result[k]が周波数kの成分。
    n_samples = len(signal)

    result = np.zeros(n_samples, dtype=np.complex128)

    for k in range(n_samples):
        total = 0.0 + 0.0j
        for n in range(n_samples):
            angle = -2 * np.pi * k * n / n_samples
            total += signal[n] * np.exp(1j * angle)

        result[k] = total

    return result
```

### 解説

1. **`result = np.zeros(n_samples, dtype=np.complex128)`**：出力を複素数のゼロ配列で初期化する。`np.complex128`は、実部・虚部それぞれを`float64`（64ビット浮動小数点数）で持つ複素数の型。
2. **`total = 0.0 + 0.0j`**：Pythonでは虚数単位`i`を`j`と書く（数値の直後に`j`を付けると虚数リテラルになる。Ruby/PHPには無い書き方で、電気工学の慣習で`i`が電流を表すためすでに使われており、数学と記号がぶつからないよう`j`が採用されたと言われている）。`0.0 + 0.0j`は実部・虚部ともに0の複素数`0`。
3. **`angle = -2 * np.pi * k * n / n_samples`**：定義式`exp(-2πikn/N)`のうち、`i`を掛ける前の角度（ラジアン）部分だけを先に計算する。`np.pi`は円周率`π`の定数。`2 * np.pi`は1周（360度）分の角度、`k * n / n_samples`は周波数`k`とサンプル位置`n`から決まる「1周のうちどれだけ回転した位置か」という比率。先頭の負号は定義式の`exp(-2πikn/N)`の符号に対応する。
4. **`np.exp(1j * angle)`**：NumPyの`exp()`は、複素数を渡すと複素指数関数として計算される（オイラーの公式により`cos(angle) + i・sin(angle)`を返す）。`1j`はPythonの虚数単位リテラルで、`1j * angle`は`angle`を虚部に持つ純虚数を作る。
5. **`total += signal[n] * np.exp(1j * angle)`**：定義式の`Σ x_n・exp(...)`をそのまま計算する。実数の`signal[n]`と複素数`np.exp(...)`を掛けると、自動的に複素数の掛け算になる。

## 具体的な計算例

**定数信号**`[5, 5, 5, 5]`のDFTを計算する。

```
入力: [5, 5, 5, 5]
DFT:  [20+0j, 0+0j, 0+0j, 0+0j]
```

定数信号は「変化が無い」ため、周波数`0`（直流成分）だけに値が集中し、他の周波数は全て`0`になる。直流成分の値`20`は、信号の総和（`5×4=20`）に一致する。[`test_dft.py`](https://github.com/oguratakayuki/image-processing-lab/blob/main/engine/tests/frequency/test_dft.py)の`test_constant_signal_has_only_dc_component`で検証している値と一致する。

**ちょうど1周期収まる余弦波**（`N=8`、`cos(2πn/8)`）のDFTも確認する。

```
入力（丸めて表示）: [1, 0.707, 0, -0.707, -1, -0.707, 0, 0.707]
DFT:               [0, 4, 0, 0, 0, 0, 0, 4]   （絶対値 |X_k|）
```

周波数`k=1`と、その鏡映である`k=N-1=7`に、振幅`N/2=4`のピークが現れている。実数信号のDFTは`X_{N-k}`が`X_k`の複素共役になるという対称性を持つため、`k=1`と`k=7`の両方にピークが現れる。[`test_dft.py`](https://github.com/oguratakayuki/image-processing-lab/blob/main/engine/tests/frequency/test_dft.py)の`test_single_period_cosine_peaks_at_frequency_1`で検証している値と一致する（`np.fft.fft`という、NumPy組み込みの高速フーリエ変換との一致も、`test_matches_numpy_fft_reference`で別途確認している——あくまで自前実装`dft_1d`の答え合わせとしてのみ使用し、本体の実装では使っていない）。

## 関連

- [波と周波数の基礎（補足資料）](6-1-dft-waves-and-frequency.md) — 振幅・周波数・周期・位相という言葉の意味
- [複素数とは（補足資料）](6-1-dft-complex-numbers.md) — `X_k`が複素数である理由、絶対値・偏角の意味
- [オイラーの公式（補足資料）](6-1-dft-euler-formula.md) — `exp(-2πikn/N)`という定義式の根拠
- [画像の1行に対するDFTの適用例（補足資料）](6-1-dft-image-row-example.md) — `k`が画像の明暗の変化パターンにどう対応するか、図解付き
- [2-1. 畳み込み](../2-convolution/2-1-convolution.md) — 「畳み込み定理」という形で、このグループと繋がる伏線がある
- [2-2. カーネルプリセット](../2-convolution/2-2-kernel-presets.md) — ガウシアンカーネルの分離可能性（2次元を1次元×1次元に分解する考え方）が、2次元DFTの分解と同じ構造を持つ

---

前へ: [← 5-6. バイリニア補間](../5-geometry/5-6-interpolation.md)　｜　次へ: なし（現時点で最後の実装済み項目）
