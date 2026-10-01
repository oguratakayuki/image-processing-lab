# 1-2. 明るさ・コントラスト調整

[← 1. 色調整へ戻る](README.md)　｜　[1-1](1-1-grayscale.md) · **1-2** · [1-3](1-3-histogram.md) · [1-4](1-4-threshold.md)

| | |
|---|---|
| 学習テーマ | アフィン変換 |
| 実装ファイル | `engine/src/imglab_engine/color/brightness_contrast.py` |
| 関数 | `adjust_brightness_contrast()` |
| テスト | `engine/tests/color/test_brightness_contrast.py` |

## 概念

- **明るさ（Brightness）**：画像全体を一律に明るく/暗くする調整。全ピクセル値に同じ値を足す。
- **コントラスト（Contrast）**：明暗の差を強調/縮小する調整。全ピクセル値に同じ係数を掛ける。

## 数学的背景

$$
g(x) = \alpha x + \beta
$$

- `alpha`（コントラスト係数）：1より大きいと差が強調され、1未満だと差が縮む。`x=0`を固定点とするスケーリングなので、暗い部分ほど動きが小さく、明るい部分ほど大きく動く。
- `beta`（明るさオフセット）：全ピクセルを一律にシフトする。

[1-1のGrayscale変換](1-1-grayscale.md) `f(v)=w・v` は**線形写像**（入力0は必ず出力0に写る、`f(0)=0`）だったが、今回は`beta ≠ 0`なら入力0（真っ黒のピクセル）が出力0に写らなくなる——つまり**真っ黒のピクセルが変換後も真っ黒のままとは限らない**。これが**線形（linear）とアフィン（affine）の違い**（「線形写像＋平行移動＝アフィン変換」）。

（補足：ここでの「入力0が出力0に写る」は、横軸=入力値・縦軸=出力値としたグラフが原点`(0,0)`を通るかどうかという、関数の性質の話である。画像内でピクセルの位置が動くという意味ではない。）

前回は重みの和が1という制約のおかげでクリップ不要だったが、今回は`alpha, beta`が自由な値を取るため、明示的な**飽和処理（clip）**が必要になる。

## 実装コード

```python
import numpy as np


def adjust_brightness_contrast(
    image: np.ndarray, alpha: float = 1.0, beta: float = 0.0
) -> np.ndarray:
    image_float = image.astype(np.float64)

    # アフィン変換: 各要素に一律 alpha 倍 + beta シフトを適用
    transformed = alpha * image_float + beta

    # 飽和(saturation): grayscale変換のときは重みの和が1という制約で
    # 自動的に0-255に収まっていたが、alpha/betaは任意の値を取り得る
    # ためここでは明示的なクリップが必要。
    clipped = np.clip(transformed, 0, 255)

    # Quantization: 四捨五入してuint8に戻す。
    return np.round(clipped).astype(np.uint8)
```

### 解説

1. **`alpha * image_float + beta`**：アフィン変換の定義式をそのままNumPyのブロードキャスト演算で全ピクセルに適用する。Grayscale画像`(H, W)`・RGB画像`(H, W, 3)`のどちらでも同じ式が使える（チャンネルの有無によらない要素ごとの演算のため）。
2. **`np.clip(transformed, 0, 255)`**：`alpha, beta`は任意の値を取りうるため、計算結果が0-255の範囲外に出ることがある。[1-1](1-1-grayscale.md)にはなかった、このトピックで初めて必要になる処理。
3. **`np.round(...).astype(np.uint8)`**：四捨五入して`uint8`に戻す。

## 実利用例（アフィン変換一般の話）

アフィン変換は明るさ・コントラスト調整に限らず、画像処理の様々な場面で登場する。

- **幾何学的変換**（平行移動・拡大縮小・回転・せん断）は、すべて2次元アフィン変換の特殊ケース。同次座標を使うと`3×3`行列1つで統一的に表現できる
- **傾き補正（deskew）**：スキャンした書類の傾きを回転で直す
- **画像のレジストレーション**：複数画像の位置合わせ
- **機械学習のデータ拡張**：回転・平行移動・スケーリングによる水増し

アフィン変換は「直線は直線に、平行な直線は平行なまま」保たれる変換（角度や長さは変わってよい）。遠近感（奥行き）は表現できず、それには射影変換（Perspective transformation）が必要になる。

## 関連

- [1-1. Grayscale変換](1-1-grayscale.md) — 線形写像（アフィン変換の特殊ケース、`beta=0`）
- [1-4. 閾値処理](1-4-threshold.md) — 次に登場する非線形・不連続な変換との対比

---

前へ: [← 1-1. Grayscale変換](1-1-grayscale.md)　｜　次へ: [1-3. ヒストグラム計算・表示 →](1-3-histogram.md)
