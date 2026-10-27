# 6-5. 周波数領域の可視化（振幅スペクトラム表示）

[← 6. 周波数領域へ戻る](README.md)　｜　[6-1](6-1-dft.md) · [6-2](6-2-idft.md) · [6-3](6-3-dft-2d.md) · [6-4](6-4-idft-2d.md) · **6-5**

| | |
|---|---|
| 学習テーマ | 振幅スペクトラム、対数スケール、中心シフト |
| 実装ファイル | [`engine/src/imglab_engine/frequency/spectrum.py`](https://github.com/oguratakayuki/image-processing-lab/blob/main/engine/src/imglab_engine/frequency/spectrum.py) |
| 関数 | `magnitude_spectrum()` |
| テスト | [`engine/tests/frequency/test_spectrum.py`](https://github.com/oguratakayuki/image-processing-lab/blob/main/engine/tests/frequency/test_spectrum.py) |

## 超ざっくり言うと

[6-3の2次元DFT](6-3-dft-2d.md)が返す複素数の配列は、そのままでは画像として見られない。「どの周波数が、どれだけ強く含まれているか」を、目で見てわかる1枚の画像（振幅スペクトラム）に変換する。

## 概念

2次元DFTの結果`F(v, u)`は複素数で、大きさ（振幅）と位相の両方を持つ。画像として表示するには、まず振幅`|F(v, u)|`だけを取り出す必要がある。ただし、振幅をそのまま表示すると、直流成分（`F(0, 0)`）だけが桁違いに大きく、他の成分との差が激しすぎて見えなくなってしまう。そこで対数スケールに変換して差を圧縮する。さらに、`dft_2d()`が返す配列は直流成分が配列の左上に来る並びになっているが、人がスペクトラムを見るときは「低周波が中央、高周波が外側」という並びの方が直感的なため、配列の象限を入れ替えて直流成分を中央に移動させる。

## 主な用途

- **画像の周期的なパターンの診断**：ノイズ・モアレ・スキャン時の縞模様など、画像に紛れ込んだ周期的な成分は、振幅スペクトラム上では特定の位置に明るい点として現れるため、原因の特定に使える
- **フィルタ設計の確認**：周波数領域でのフィルタリング（ノイズ除去・ぼかし・先鋭化）を設計する際、処理前後のスペクトラムを見比べることで、どの周波数帯をどれだけ削った・強調したかを視覚的に確認できる
- **教育・デバッグ**：DFTの結果を数値の羅列としてではなく画像として見ることで、「低周波＝なだらかな変化」「高周波＝細かい変化」という対応を直感的に把握できる

## 数学的背景

### 要件

[6-3の2次元DFT](6-3-dft-2d.md)が返す複素数配列`F(v, u)`から、人の目で見てわかりやすい、1枚の実数値画像を作りたい。

### やるべきこと

次の3段階の処理を順番に適用する。

1. **振幅を取る**：`|F(v, u)|`（複素数の絶対値）
2. **対数スケールに変換する**：`log(1 + |F(v, u)|)`（`+1`は`|F|=0`の点で`log(0)`が未定義になるのを避けるため）
3. **中心シフトする**：配列を4つの象限に分割し、対角の象限を入れ替えることで、`F(0, 0)`（直流成分）が配列の中央に来るよう並べ替える

```
振幅         : magnitude(v, u) = |F(v, u)|
対数スケール : log_magnitude(v, u) = log(1 + magnitude(v, u))
中心シフト   : 象限を対角に入れ替え、log_magnitude(0, 0) が中央に来るようにする
```

### コードにすると

[`dft_2d()`](6-3-dft-2d.md)の結果に`np.abs()`（絶対値）と`np.log()`（対数）を順番に適用し、最後に象限入れ替えのヘルパー関数`_shift_to_center()`で並べ替える——3つの処理をそのまま順番に並べるだけでコードにできる。

## 実装コード

```python
import numpy as np

from imglab_engine.frequency.dft import dft_2d


def _shift_to_center(array_2d: np.ndarray) -> np.ndarray:
    # array_2d: 形状(H, W)の2次元配列。
    # 返却値: 形状(H, W)の2次元配列。array_2d[0, 0]の値が
    #         array_2d[H//2, W//2]の位置に移動している。
    height, width = array_2d.shape
    half_height, half_width = height // 2, width // 2

    top_left = array_2d[:half_height, :half_width]
    top_right = array_2d[:half_height, half_width:]
    bottom_left = array_2d[half_height:, :half_width]
    bottom_right = array_2d[half_height:, half_width:]

    top_half = np.concatenate([bottom_right, bottom_left], axis=1)
    bottom_half = np.concatenate([top_right, top_left], axis=1)
    return np.concatenate([top_half, bottom_half], axis=0)


def magnitude_spectrum(image: np.ndarray) -> np.ndarray:
    # image: 形状(H, W)の2次元配列。
    # 返却値: 形状(H, W)の実数配列(float64)。値が大きいほど、その
    #         位置に対応する周波数成分が強く含まれていることを示す。
    #         配列の中央が直流成分(最も低い周波数)に対応する。
    spectrum = dft_2d(image)
    magnitude = np.abs(spectrum)
    log_magnitude = np.log(1 + magnitude)
    return _shift_to_center(log_magnitude)
```

### 解説

1. **`height, width = array_2d.shape`**：`_shift_to_center()`は、まず配列を縦・横それぞれの中央で4分割する準備として、配列の高さと幅を取り出す。
2. **`top_left = array_2d[:half_height, :half_width]`ほか3行**：NumPyのスライス構文で、配列を「上半分×左半分」「上半分×右半分」「下半分×左半分」「下半分×右半分」の4つの象限に分割する。
3. **`np.concatenate([bottom_right, bottom_left], axis=1)`ほか**：`np.concatenate`は、複数の配列を指定した軸（`axis`）に沿って連結する関数。対角にある象限どうし（左上⇔右下、右上⇔左下）を入れ替えて結合することで、元々`[0, 0]`にあった値（右下の象限に含まれる）が、結合後の配列の中央付近に来るようにしている。
4. **`spectrum = dft_2d(image)`**：[6-3のDFT](6-3-dft-2d.md)で複素数の周波数成分を求める。
5. **`magnitude = np.abs(spectrum)`**：`np.abs`は複素数の絶対値（振幅）`sqrt(実部² + 虚部²)`を計算する。
6. **`log_magnitude = np.log(1 + magnitude)`**：`np.log`は自然対数を計算する。`1`を足してから対数を取ることで、`magnitude=0`の点が`log(0)=-∞`になるのを避けている。
7. **`return _shift_to_center(log_magnitude)`**：直流成分が中央に来るよう並べ替えて返す。

## 具体的な計算例

**定数画像**（`8×8`、全画素`5`）の振幅スペクトラムを計算する。

```
入力: 全画素5の8x8画像（画素値の総和 = 5×64 = 320）

振幅スペクトラム(log+中心シフト後):
中央(4,4)だけ 5.771、他は全て0
```

`log(1 + 320) = log(321) ≈ 5.771`。[6-1の定数信号の例](6-1-dft.md)と同じく、変化の無い画像は直流成分だけにエネルギーが集中するため、スペクトラム画像では中央に1点だけ明るい点が現れ、他は真っ黒になる。[`test_spectrum.py`](https://github.com/oguratakayuki/image-processing-lab/blob/main/engine/tests/frequency/test_spectrum.py)の`test_constant_image_peak_is_at_center`で検証している値と一致する。

**横方向だけに変化する画像**（[6-3と同じ`H=W=8`、`cos(2πx/8)`](6-3-dft-2d.md)）も確認する。

```
振幅スペクトラム(log+中心シフト後):
(4,3)と(4,5)がそれぞれ 3.4965、他は全て0
```

`log(1 + 32) ≈ 3.4965`。中心シフトにより、直流成分（中央`(4,4)`）を挟んで**左右対称**な位置に2つのピークが現れる——シフト前は`u=1`と`u=7`（配列の両端）にあったピークが、シフト後は中央から見て対称な`(4,3)`・`(4,5)`に移動したことに相当する。[`test_spectrum.py`](https://github.com/oguratakayuki/image-processing-lab/blob/main/engine/tests/frequency/test_spectrum.py)の`test_horizontal_only_pattern_peaks_are_symmetric_around_center`で検証している値と一致する（`np.fft.fft2`・`np.fft.fftshift`という、NumPy組み込みの高速フーリエ変換・シフト関数との一致も`test_matches_reference_pipeline_built_from_numpy_fft`で別途確認している——あくまで自前実装の答え合わせとしてのみ使用し、本体の実装では使っていない）。

## 関連

- [6-3. 離散フーリエ変換（2次元）](6-3-dft-2d.md) — このページが可視化している2次元DFTの定義・`dft_2d()`
- [6-1. 離散フーリエ変換（1次元）](6-1-dft.md) — 定数信号の例、直流成分という考え方の出発点

---

前へ: [← 6-4. 逆離散フーリエ変換（2次元）](6-4-idft-2d.md)　｜　次へ: なし（現時点で最後の実装済み項目）
