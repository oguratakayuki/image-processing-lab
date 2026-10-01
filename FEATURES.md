# 機能一覧

Image Processing Labで実装済みの機能と、それぞれの学習テーマ・実装箇所の対応表。

## ページ

| ページ | URL (dev) | 内容 |
|---|---|---|
| Color Lab | `/lab/color` | Grayscale / 明るさ・コントラスト / ヒストグラム / 閾値処理 |
| Convolution Lab | `/lab/convolution` | 畳み込み（カーネル編集・OpenCV比較） |
| Edge Detection Lab | `/lab/edge` | Sobel / Prewitt / Laplacian |
| Morphology Lab | `/lab/morphology` | Erosion / Dilation / Opening / Closing |

## 機能一覧（学習テーマ・実装箇所）

### `/lab/color`

| # | 機能 | 学習テーマ | 実装ファイル / 関数 |
|---|---|---|---|
| 1 | Grayscale変換 | 線形写像（内積） | `engine/src/imglab_engine/color/grayscale.py` — `to_grayscale()` |
| 2 | 明るさ・コントラスト調整 | アフィン変換 | `engine/src/imglab_engine/color/brightness_contrast.py` — `adjust_brightness_contrast()` |
| 3 | ヒストグラム計算・表示 | 度数分布・経験分布 | `engine/src/imglab_engine/histogram/histogram.py` — `compute_histogram()` |
| 4 | 閾値処理 | ヘヴィサイド階段関数／集合の定義関数 | `engine/src/imglab_engine/color/threshold.py` — `apply_threshold()` |

### `/lab/convolution`

| # | 機能 | 学習テーマ | 実装ファイル / 関数 |
|---|---|---|---|
| 5 | 畳み込み | 離散畳み込み、カーネル反転、畳み込み定理 | `engine/src/imglab_engine/convolution/convolution.py` — `convolve2d()` |
| ─ | カーネルプリセット（補助） | ガウス分布のサンプリング、カーネルの線形結合 | `engine/src/imglab_engine/convolution/kernels.py` — `identity_kernel()`, `mean_kernel()`, `gaussian_kernel()`, `laplacian_kernel()`, `sharpen_kernel()` |
| ─ | OpenCV比較（補助） | 畳み込み(Convolution) vs 相関(Correlation) | `engine/src/imglab_engine/reference/convolution_reference.py` — `cv2_filter2d()` |

### `/lab/edge`

| # | 機能 | 学習テーマ | 実装ファイル / 関数 |
|---|---|---|---|
| 6 | Sobelエッジ検出 | 有限差分近似、勾配ベクトル、ユークリッドノルム | `engine/src/imglab_engine/edge/sobel.py` — `sobel_gradient()`<br>`engine/src/imglab_engine/edge/gradient.py` — `gradient_magnitude()` |
| 7 | Prewittエッジ検出 | Sobelと同構造（平滑化重みの違い） | `engine/src/imglab_engine/edge/prewitt.py` — `prewitt_gradient()`（`gradient_magnitude()`を共用） |
| 8 | Laplacianエッジ検出 | 2階微分、零交差 | `engine/src/imglab_engine/edge/laplacian.py` — `laplacian_edge_response()`（内部で`kernels.py`の`laplacian_kernel()`を再利用） |

### `/lab/morphology`

| # | 機能 | 学習テーマ | 実装ファイル / 関数 |
|---|---|---|---|
| 9 | Erosion（収縮） | 集合演算（AND）、構造要素 | `engine/src/imglab_engine/morphology/morphology.py` — `erode()`（内部で`_local_reduce()`、構造要素は`square_structuring_element()`） |
| 10 | Dilation（膨張） | 集合演算（OR）、構造要素 | 同ファイル — `dilate()` |
| 11 | Opening（収縮→膨張） | Erosion/Dilationの合成 | 同ファイル — `opening()`（`erode()`→`dilate()`を呼ぶだけ） |
| 12 | Closing（膨張→収縮） | Erosion/Dilationの合成、双対性 | 同ファイル — `closing()`（`dilate()`→`erode()`を呼ぶだけ） |

## 補足

- 各機能をAPI経由で呼び出す配線は `api/app/routers/{color,histogram,convolution,edge,morphology}.py`、画面表示は `web/src/app/lab/{color,convolution,edge,morphology}/page.tsx` に対応している。学習テーマそのものの実装は全て `engine/` 側の関数にある。
- engineテストは66件、全てパス（2026-10-01時点）。
