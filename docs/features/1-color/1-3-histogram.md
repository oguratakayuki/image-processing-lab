# 1-3. ヒストグラム計算・表示

[← 1. 色調整へ戻る](README.md)　｜　[1-1](1-1-grayscale.md) · [1-2](1-2-brightness-contrast.md) · **1-3** · [1-4](1-4-threshold.md)

| | |
|---|---|
| 学習テーマ | 度数分布・経験分布（基礎統計学） |
| 実装ファイル | [`engine/src/imglab_engine/histogram/histogram.py`](https://github.com/oguratakayuki/image-processing-lab/blob/main/engine/src/imglab_engine/histogram/histogram.py) |
| 関数 | `compute_histogram()` |
| テスト | `engine/tests/histogram/test_histogram.py` |

## 概念

Grayscale画像の各ピクセル値（0〜255）が画像中に何回出現するかを数えた**度数分布**。「暗いピクセルが多いのか、明るいピクセルが多いのか」「コントラストが低く値が真ん中に集中していないか」を一目で把握できる。

## 数学的背景

$$
h(v) = |\{(i,j) : I(i,j) = v\}|, \quad v = 0, 1, \dots, 255
$$

総ピクセル数 `N = H × W` で割った `p(v) = h(v)/N` は、ピクセル値という確率変数の**経験分布**（empirical probability mass function）とみなせる。この `p(v)` から平均・分散を計算できる（`mean = Σ v・p(v)`（vについて0〜255の総和）など）ことが、後の統計トピックに直結する。

[1-2の明るさ・コントラスト調整](1-2-brightness-contrast.md)との関係でいうと、ヒストグラムは「アフィン変換 `g(x)=alpha*x+beta` を適用すると分布がどう平行移動・拡大縮小するか」を可視化する道具にもなる。

## 実装コード

```python
import numpy as np


def compute_histogram(channel: np.ndarray) -> np.ndarray:
    # channel: Grayscale画像。(H, W)のuint8配列(値域0-255)。
    # 返却値: 長さ256のint64配列。result[v]はピクセル値vの出現回数。
    return np.bincount(channel.ravel(), minlength=256)[:256]
```

### 解説

1. **`channel.ravel()`**：`(H, W)`の2次元配列を1次元配列に平坦化する。（[補足資料](1-3-histogram-ravel.md)）
2. **`np.bincount(..., minlength=256)`**：非負整数配列の中で、各値が何回出現するかを数える関数。戻り値は長さ256（または最大値+1、`minlength`で下限を保証）のint64配列で、`result[v]`がピクセル値`v`の出現回数になる。
3. **`[:256]`**：`minlength=256`だけでは理論上256を超える長さになる可能性がある境界ケースを防ぐため、明示的に256要素にスライスする。

### 実装上の注意点

素朴に `hist = np.zeros(256); hist[channel.ravel()] += 1` と書きたくなるが、これは**使えない**。NumPyのファンシーインデックスによる代入は、同じインデックスが複数回登場しても加算が積み上がらず「最後に書き込んだ値」で上書きされてしまう（重複インデックスへの`+=`が正しく累積されないという既知の罠）。`np.bincount`を使うことで、この罠を回避しつつ各値の出現回数を正しく数えられる。

## 関連

- [1-2. 明るさ・コントラスト調整](1-2-brightness-contrast.md) — 分布がどう変形するかを観察する対象
- [1-4. 閾値処理](1-4-threshold.md) — しきい値をどこに置くべきか判断する材料になる

---

前へ: [← 1-2. 明るさ・コントラスト調整](1-2-brightness-contrast.md)　｜　次へ: [1-4. 閾値処理 →](1-4-threshold.md)
