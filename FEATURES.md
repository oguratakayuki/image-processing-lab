# 機能一覧

Image Processing Labで実装済みの機能と、それぞれの学習テーマ・実装箇所の対応表。

各項目には「大分類番号-項目番号」（例: `1-2`）を振ってあります。「1-2について解説して」のように番号で指定すれば、この表の該当項目を指すものとして扱います。

詳細な解説は各グループページ（下記リンク）に分けて記載しています。

## ページ

| 番号 | タイトル | URL (dev) | 内容 | 詳細 |
|---|---|---|---|---|
| 1 | 色調整 | `/lab/color` | Grayscale / 明るさ・コントラスト / ヒストグラム / 閾値処理 | [詳細](docs/features/1-color/README.md) |
| 2 | 畳み込み | `/lab/convolution` | 畳み込み（カーネル編集・OpenCV比較） | [詳細](docs/features/2-convolution/README.md) |
| 3 | エッジ検出 | `/lab/edge` | Sobel / Prewitt / Laplacian | [詳細](docs/features/3-edge/README.md) |
| 4 | 二値画像処理 | `/lab/morphology` | Erosion / Dilation / Opening / Closing | [詳細](docs/features/4-morphology/README.md) |

## 機能一覧（学習テーマ・実装箇所の早見表）

### 1. 色調整（[詳細](docs/features/1-color/README.md)）

| 番号 | 機能 | 学習テーマ | 実装ファイル / 関数 |
|---|---|---|---|
| [1-1](docs/features/1-color/1-1-grayscale.md) | Grayscale変換 | 線形写像（内積） | `engine/src/imglab_engine/color/grayscale.py` — `to_grayscale()` |
| [1-2](docs/features/1-color/1-2-brightness-contrast.md) | 明るさ・コントラスト調整 | アフィン変換 | `engine/src/imglab_engine/color/brightness_contrast.py` — `adjust_brightness_contrast()` |
| [1-3](docs/features/1-color/1-3-histogram.md) | ヒストグラム計算・表示 | 度数分布・経験分布 | `engine/src/imglab_engine/histogram/histogram.py` — `compute_histogram()` |
| [1-4](docs/features/1-color/1-4-threshold.md) | 閾値処理 | ヘヴィサイド階段関数／集合の定義関数 | `engine/src/imglab_engine/color/threshold.py` — `apply_threshold()` |

### 2. 畳み込み（[詳細](docs/features/2-convolution/README.md)）

| 番号 | 機能 | 学習テーマ | 実装ファイル / 関数 |
|---|---|---|---|
| [2-1](docs/features/2-convolution/2-1-convolution.md) | 畳み込み | 離散畳み込み、カーネル反転、畳み込み定理 | `engine/src/imglab_engine/convolution/convolution.py` — `convolve2d()` |
| [2-2](docs/features/2-convolution/2-2-kernel-presets.md) | カーネルプリセット（補助） | ガウス分布のサンプリング、カーネルの線形結合 | `engine/src/imglab_engine/convolution/kernels.py` — `identity_kernel()`, `mean_kernel()`, `gaussian_kernel()`, `laplacian_kernel()`, `sharpen_kernel()` |
| [2-3](docs/features/2-convolution/2-3-opencv-comparison.md) | OpenCV比較（補助） | 畳み込み(Convolution) vs 相関(Correlation) | `engine/src/imglab_engine/reference/convolution_reference.py` — `cv2_filter2d()` |

### 3. エッジ検出（[詳細](docs/features/3-edge/README.md)）

| 番号 | 機能 | 学習テーマ | 実装ファイル / 関数 |
|---|---|---|---|
| 3-1 | Sobelエッジ検出 | 有限差分近似、勾配ベクトル、ユークリッドノルム | `engine/src/imglab_engine/edge/sobel.py` — `sobel_gradient()`<br>`engine/src/imglab_engine/edge/gradient.py` — `gradient_magnitude()` |
| 3-2 | Prewittエッジ検出 | Sobelと同構造（平滑化重みの違い） | `engine/src/imglab_engine/edge/prewitt.py` — `prewitt_gradient()`（`gradient_magnitude()`を共用） |
| 3-3 | Laplacianエッジ検出 | 2階微分、零交差 | `engine/src/imglab_engine/edge/laplacian.py` — `laplacian_edge_response()`（内部で`kernels.py`の`laplacian_kernel()`を再利用） |

### 4. 二値画像処理（[詳細](docs/features/4-morphology/README.md)）

| 番号 | 機能 | 学習テーマ | 実装ファイル / 関数 |
|---|---|---|---|
| 4-1 | Erosion（収縮） | 集合演算（AND）、構造要素 | `engine/src/imglab_engine/morphology/morphology.py` — `erode()`（内部で`_local_reduce()`、構造要素は`square_structuring_element()`） |
| 4-2 | Dilation（膨張） | 集合演算（OR）、構造要素 | 同ファイル — `dilate()` |
| 4-3 | Opening（収縮→膨張） | Erosion/Dilationの合成 | 同ファイル — `opening()`（`erode()`→`dilate()`を呼ぶだけ） |
| 4-4 | Closing（膨張→収縮） | Erosion/Dilationの合成、双対性 | 同ファイル — `closing()`（`dilate()`→`erode()`を呼ぶだけ） |

## 補足

- 各機能をAPI経由で呼び出す配線は `api/app/routers/{color,histogram,convolution,edge,morphology}.py`、画面表示は `web/src/app/lab/{color,convolution,edge,morphology}/page.tsx` に対応している。学習テーマそのものの実装は全て `engine/` 側の関数にある。
- engineテストは66件、全てパス（2026-10-01時点）。
- 現時点で詳細ページ（概念・数式・コード解説）が整備済みなのは「1. 色調整」「2. 畳み込み」グループ。3〜4は早見表のみ。
