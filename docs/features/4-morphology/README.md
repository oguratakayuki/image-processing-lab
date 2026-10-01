# 4. 二値画像処理（`/lab/morphology`）

[← 機能一覧トップへ戻る](../../../FEATURES.md)

> 詳細ページは未整備。早見表は[トップページ](../../../FEATURES.md)を参照。

| 番号 | 機能 | 学習テーマ | 実装ファイル / 関数 |
|---|---|---|---|
| 4-1 | Erosion（収縮） | 集合演算（AND）、構造要素 | `engine/src/imglab_engine/morphology/morphology.py` — `erode()` |
| 4-2 | Dilation（膨張） | 集合演算（OR）、構造要素 | 同ファイル — `dilate()` |
| 4-3 | Opening（収縮→膨張） | Erosion/Dilationの合成 | 同ファイル — `opening()` |
| 4-4 | Closing（膨張→収縮） | Erosion/Dilationの合成、双対性 | 同ファイル — `closing()` |
