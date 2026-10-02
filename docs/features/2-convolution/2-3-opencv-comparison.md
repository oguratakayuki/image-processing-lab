# 2-3. OpenCVとの比較

[← 2. 畳み込みへ戻る](README.md)　｜　[2-1](2-1-convolution.md) · [2-2](2-2-kernel-presets.md) · **2-3**

| | |
|---|---|
| 学習テーマ | 畳み込み（Convolution）vs 相関（Correlation） |
| 実装ファイル | [`engine/src/imglab_engine/reference/convolution_reference.py`](https://github.com/oguratakayuki/image-processing-lab/blob/main/engine/src/imglab_engine/reference/convolution_reference.py) |
| 関数 | `cv2_filter2d()` |
| テスト | [`engine/tests/reference/test_convolution_reference.py`](https://github.com/oguratakayuki/image-processing-lab/blob/main/engine/tests/reference/test_convolution_reference.py) |

## 概念

[2-1](2-1-convolution.md)で実装した自前の`convolve2d`（本物の畳み込み＝カーネル反転あり）と、OpenCVの`cv2.filter2D`を比較し、「畳み込みと相関の違い」を実データで確認する。このモジュールはengine内の自前実装からは一切importされない（答え合わせ・比較専用、依存の向きを一方向に保つため）。

## 数学的な注意

OpenCVの`cv2.filter2D`は、名前とは裏腹に畳み込みではなく**相関（Correlation）**を計算する（カーネルを反転しない）。OpenCVの公式ドキュメントにも「本当の畳み込みが欲しいならflip()でカーネルを反転してから使え」と明記されている。

そのため：

- **対称なカーネル**（mean, gaussianなど。`K(i,j)=K(-i,-j)`）では、反転してもしなくても同じカーネルになるので、自前の`convolve2d`（畳み込み）と`cv2.filter2D`（相関）は**同じ結果**になる
- **非対称なカーネル**（例：中心の1つ左だけが1のカーネル）では、畳み込みと相関で「シフトする方向が逆」になり、**結果が一致しない**

この関数はcv2の挙動をそのまま反映しており、意図的に反転を加えていない（「本物の畳み込み」に揃えたい場合は、呼び出し側で`kernel[::-1, ::-1]`を渡せばよい）。

**境界処理について**：OpenCVのデフォルトの境界処理は`BORDER_REFLECT_101`（鏡映）であり、本プロジェクトの`convolve2d`が採用しているゼロパディングとは異なる。境界処理の違いが比較結果に混ざらないよう、明示的に`borderType=cv2.BORDER_CONSTANT`（ゼロ埋め）を指定している。

## 実装コード

```python
import cv2
import numpy as np


def cv2_filter2d(channel: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    result = cv2.filter2D(
        channel.astype(np.float64),
        ddepth=-1,
        kernel=kernel,
        borderType=cv2.BORDER_CONSTANT,
    )
    return np.clip(np.round(result), 0, 255).astype(np.uint8)
```

### 解説

1. **`channel.astype(np.float64)`**：自前実装と同様、精度を保つためfloat64に変換してから渡す。
2. **`ddepth=-1`**：出力のビット深度を入力と同じにする指定（OpenCVのAPI仕様）。
3. **`borderType=cv2.BORDER_CONSTANT`**：境界処理をゼロパディングに固定し、自前実装と条件を揃える。
4. **`np.clip(...).astype(np.uint8)`**：自前実装と同じく0-255にクリップして量子化する。

## 検証された事実

テストで以下を確認している。

- **対称カーネル（mean）**：`convolve2d`と`cv2_filter2d`が完全一致
- **非対称カーネル**：意図的に不一致（畳み込みと相関の方向差を検証）
- **非対称カーネルでも事前に反転したカーネルを`cv2_filter2d`に渡せば一致**（畳み込み＝反転＋相関、の直接的な裏付け）

実際にWebの[Convolution Lab](/lab/convolution)で「OpenCVと比較する」チェックボックスをオンにすると、`|自前実装 − OpenCV|`の差分画像が表示され、対称カーネルでは差分0、非対称カーネル（シフトカーネルなど）では境界部分に明確な差分が現れる。

## 関連

- [2-1. 畳み込み](2-1-convolution.md) — 比較対象の自前実装
- [2-2. カーネルプリセット](2-2-kernel-presets.md) — 対称カーネル（mean, gaussian）と非対称カーネルの違いを試せる

---

前へ: [← 2-2. カーネルプリセット](2-2-kernel-presets.md)　｜　次へ: [3. エッジ検出 →](../3-edge/README.md)（詳細ページは未整備）
