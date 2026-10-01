# 3. エッジ検出（`/lab/edge`）

[← 機能一覧トップへ戻る](../../../FEATURES.md)

> 詳細ページは未整備。早見表は[トップページ](../../../FEATURES.md)を参照。

| 番号 | 機能 | 学習テーマ | 実装ファイル / 関数 |
|---|---|---|---|
| 3-1 | Sobelエッジ検出 | 有限差分近似、勾配ベクトル、ユークリッドノルム | `engine/src/imglab_engine/edge/sobel.py` — `sobel_gradient()` |
| 3-2 | Prewittエッジ検出 | Sobelと同構造（平滑化重みの違い） | `engine/src/imglab_engine/edge/prewitt.py` — `prewitt_gradient()` |
| 3-3 | Laplacianエッジ検出 | 2階微分、零交差 | `engine/src/imglab_engine/edge/laplacian.py` — `laplacian_edge_response()` |
