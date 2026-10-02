# 1-4. 閾値処理（Thresholding）

[← 1. 色調整へ戻る](README.md)　｜　[1-1](1-1-grayscale.md) · [1-2](1-2-brightness-contrast.md) · [1-3](1-3-histogram.md) · **1-4**

| | |
|---|---|
| 学習テーマ | ヘヴィサイド階段関数／集合の定義関数 |
| 実装ファイル | [`engine/src/imglab_engine/color/threshold.py`](https://github.com/oguratakayuki/image-processing-lab/blob/main/engine/src/imglab_engine/color/threshold.py) |
| 関数 | `apply_threshold()` |
| テスト | `engine/tests/color/test_threshold.py` |

## 概念

Grayscale画像の各ピクセルを、しきい値 `T` との大小関係だけで白（255）か黒（0）の2値に変換する処理。前景／背景の分離、後続の二値画像処理（Erosion/Dilation、[4. 二値画像処理](../4-morphology/README.md)）の入力準備として使われる。

## 数学的背景

```
g(x) = 255  (x >= T)
       0    (x <  T)
```

これは**ヘヴィサイドの階段関数**（`H(y) = 1 (y>=0), H(y) = 0 (y<0)`）を使うと`g(x) = 255 * H(x-T)`と書ける。`x=T`で値が不連続にジャンプする、**微分不可能な非線形関数**である。

これまでの変換との位置づけ：

| 変換 | 数式 | 性質 |
|---|---|---|
| [1-1 Grayscale](1-1-grayscale.md) | `f(v)=w・v` | 線形写像（R³→R） |
| [1-2 明るさ/コントラスト](1-2-brightness-contrast.md) | `g(x)=alpha*x+beta` | アフィン変換（連続・可逆） |
| 1-4 閾値処理 | `g(x)=255*H(x-T)` | 非線形・不連続・非可逆 |

閾値処理は初めて「値の大小関係の情報すら捨てる」変換。集合論的には「`T`以上のピクセルの集合」の**定義関数（indicator function）**（`g(x)=255*1_[T,255](x)`）とも解釈できる。この「画像を集合として捉える」見方が、[4. 二値画像処理](../4-morphology/README.md)で本格的に使われる。

## 実装コード

```python
import numpy as np


def apply_threshold(channel: np.ndarray, t: int) -> np.ndarray:
    is_foreground = channel >= t  # H(x - t) をベクトル化して一括計算
    return (is_foreground * 255).astype(np.uint8)
```

### 解説

1. **`channel >= t`**：要素ごとの比較演算（broadcasting）で、画像と同じ形状の真偽値配列（bool ndarray, True/False）を返す。これはヘヴィサイド関数`H(x-t)`を全ピクセルに対して同時に計算していることと数学的に同値で、`True=1, False=0`という対応がそのまま`H(y)=1 (y>=0), H(y)=0 (y<0)`に対応する。
2. **`(is_foreground * 255).astype(np.uint8)`**：bool配列に255を掛けてuint8にキャストすることで、「True/Falseの配列」を「0/255の画像」に変換する。
3. 境界値`x=t`は`>=`なので前景（255）側に含む規約。

## 情報量の観点

Grayscale変換は色の情報（3次元→1次元）を失う点で不可逆だったが、輝度の「大小関係」自体は保存していた（単調増加関数）。閾値処理はその大小関係の情報すら捨て、「`T`より大きいか小さいか」という1bitの情報だけを残す、これまでで最も情報量を落とす変換になる。

## 関連

- [1-1. Grayscale変換](1-1-grayscale.md) — 内部で呼び出している（RGB→Grayscale→閾値処理という2段のパイプライン）
- [1-3. ヒストグラム](1-3-histogram.md) — しきい値`T`をどこに置くか判断する材料
- [4. 二値画像処理](../4-morphology/README.md) — ここで導入した「集合としての画像」という視点を本格的に使う

---

前へ: [← 1-3. ヒストグラム計算・表示](1-3-histogram.md)　｜　次へ: [2. 畳み込み →](../2-convolution/README.md)（詳細ページは未整備）
