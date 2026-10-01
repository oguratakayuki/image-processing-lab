# 1-1. Grayscale変換

[← 1. 色調整へ戻る](README.md)　｜　**1-1** · [1-2](1-2-brightness-contrast.md) · [1-3](1-3-histogram.md) · [1-4](1-4-threshold.md)

| | |
|---|---|
| 学習テーマ | 線形写像（線形代数：内積） |
| 実装ファイル | `engine/src/imglab_engine/color/grayscale.py` |
| 関数 | `to_grayscale()` |
| テスト | `engine/tests/color/test_grayscale.py` |

## 概念

RGB画像の各ピクセルを、色情報を持たない「明るさ（輝度）」の1チャンネルに変換する。RGBの3チャンネルを1つの値に落とし込む変換。

## 数学的背景

$$
\text{Gray} = 0.299R + 0.587G + 0.114B
$$

単純平均 `(R+G+B)/3` ではなく加重平均にするのは、人間の目が緑に最も敏感で青に最も鈍感という知覚特性を反映するため。この重み `(0.299, 0.587, 0.114)` はITU-R BT.601で定義された係数で、3つを足すとちょうど1.0になる（＝明るさのスケールを保存する）。

これは数学的には、RGBベクトル `(R,G,B)` と重みベクトル `w=(0.299,0.587,0.114)` の**内積**であり、3次元の色空間を1次元の明るさ軸へ射影する**線形写像**である（`f(0)=0` が必ず成り立つ）。

## 実装コード

```python
import numpy as np

_BT601_WEIGHTS = np.array([0.299, 0.587, 0.114], dtype=np.float64)


def to_grayscale(image: np.ndarray) -> np.ndarray:
    image_float = image.astype(np.float64)
    gray_float = image_float @ _BT601_WEIGHTS
    return np.round(gray_float).astype(np.uint8)
```

### 解説

1. **`_BT601_WEIGHTS`**：重みベクトル `w` をそのままNumPy配列にしたもの。モジュール読み込み時に1回だけ作られる。
2. **`image.astype(np.float64)`**：入力は`(H, W, 3)`の`uint8`配列。加重和の計算で丸め誤差や桁あふれが起きないよう、先にfloat64に変換する。
3. **`image_float @ _BT601_WEIGHTS`**：`@`はNumPyの行列積演算子。`(H, W, 3)`の配列と`(3,)`のベクトルを掛けると、最後の軸（チャンネル軸）同士が内積として畳み込まれ、結果は`(H, W)`になる。つまり各ピクセル`(y,x)`について

   `gray_float[y,x] = R[y,x]*0.299 + G[y,x]*0.587 + B[y,x]*0.114`

   が、forループを使わず全ピクセルに対して一括で計算される（ベクトル化）。
4. **`np.round(...).astype(np.uint8)`**：float64の計算結果を0-255の整数(`uint8`)に戻す。重み`0.299+0.587+0.114=1.0`のため、入力が0-255の範囲であれば出力も必ず0-255に収まり、クリップ処理は不要。

## 補足資料

[📊 RGBベクトルからGrayscale値への射影（図解）](assets/1-1-rgb-projection.svg)

3次元のRGB空間に実際のテストケース（black, blue, red, gray128, mid, green, white）をプロットし、重みベクトル`w`方向への射影と、その結果得られる1次元の出力値（0〜255）の対応関係を図示したもの。

## テストケース

| 入力 (R,G,B) | 期待値 |
|---|---|
| (255,255,255) white | 255 |
| (0,0,0) black | 0 |
| (255,0,0) red | 76 |
| (0,255,0) green | 150 |
| (0,0,255) blue | 29 |
| (100,150,200) mid | 141 |
| (128,128,128) gray128 | 128 |

## 関連

- [1-2. 明るさ・コントラスト調整](1-2-brightness-contrast.md) — 線形写像をアフィン変換に一般化
- [1-4. 閾値処理](1-4-threshold.md) — 内部で`to_grayscale()`を呼び出す

---

前へ: なし（最初の項目）　｜　次へ: [1-2. 明るさ・コントラスト調整 →](1-2-brightness-contrast.md)
