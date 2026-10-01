# 2. 畳み込み（`/lab/convolution`）

[← 機能一覧トップへ戻る](../../../FEATURES.md)

> 詳細ページは未整備。早見表は[トップページ](../../../FEATURES.md)を参照。

| 番号 | 機能 | 学習テーマ | 実装ファイル / 関数 |
|---|---|---|---|
| 2-1 | 畳み込み | 離散畳み込み、カーネル反転、畳み込み定理 | `engine/src/imglab_engine/convolution/convolution.py` — `convolve2d()` |
| 2-2 | カーネルプリセット（補助） | ガウス分布のサンプリング、カーネルの線形結合 | `engine/src/imglab_engine/convolution/kernels.py` |
| 2-3 | OpenCV比較（補助） | 畳み込み(Convolution) vs 相関(Correlation) | `engine/src/imglab_engine/reference/convolution_reference.py` — `cv2_filter2d()` |
