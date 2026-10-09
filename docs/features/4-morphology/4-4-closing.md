# 4-4. Closing（クロージング）

[← 4. 二値画像処理へ戻る](README.md)　｜　[4-1](4-1-erosion.md) · [4-2](4-2-dilation.md) · [4-3](4-3-opening.md) · **4-4**

| | |
|---|---|
| 学習テーマ | Erosion/Dilationの合成、双対性 |
| 実装ファイル | [`engine/src/imglab_engine/morphology/morphology.py`](https://github.com/oguratakayuki/image-processing-lab/blob/main/engine/src/imglab_engine/morphology/morphology.py) |
| 関数 | `closing()` |
| テスト | [`engine/tests/morphology/test_morphology.py`](https://github.com/oguratakayuki/image-processing-lab/blob/main/engine/tests/morphology/test_morphology.py) |

## 超ざっくり言うと

膨張してから収縮することで、小さな穴やくぼみを埋めつつ、全体の形はほぼそのまま保つ操作。

## 概念

[4-3のOpening](4-3-opening.md)とちょうど順序を逆にした操作（「膨張してから収縮」）。構造要素より小さい穴やくぼみを埋めつつ、全体の輪郭はほぼ保ったまま復元する。Openingが「突起を削る」方向に働くのに対し、Closingは「穴を埋める」方向に働く。

## 主な用途

- **物体内部の小さな穴・欠けの補完**：閾値処理でノイズにより生じた物体内部の小さな穴を埋める
- **輪郭のくぼみ・ギザギザの平滑化**：輪郭の小さなくぼみを埋めて滑らかにする
- **近接する物体同士の結合**：隙間が構造要素より狭い複数の物体は、膨張時にくっつき、収縮後もつながったまま残る（[4-1の「くっついた物体の分離」](4-1-erosion.md)とはちょうど逆方向の効果）

こちらも前段の[Grayscale変換・閾値処理](4-1-erosion.md)で前景/背景が正しく分離できていることが前提（色そのものは一切見ていない）。

## 数学的背景

```
A • B = (A ⊕ B) ⊖ B
```

まず`A`を構造要素`B`で膨張する。このとき、構造要素より小さい穴・くぼみは、周囲の前景が広がることで埋まってしまう。膨張した前景を再び`B`で収縮させると、全体の大きさはおおよそ元に戻るが、膨張の時点で埋まってしまった穴は復活しない。

OpeningとClosingは、互いに「順序を入れ替えた」関係にあり、かつ補集合を挟むと双対の関係にもなる（`(A • B)^c = A^c ∘ B̂`。[4-2で見たErosion/Dilationの双対性](4-2-dilation.md)と同じ、ド・モルガンの法則に由来する構造）。

## 実装コード

```python
def closing(binary: np.ndarray, structuring_element: np.ndarray) -> np.ndarray:
    # binary: (H, W)の二値画像(uint8, 値は0か255のみ)。
    # structuring_element: (kh, kw)の真偽値配列(構造要素)。dilate・erode両方に同じものを使う。
    # 返却値: Closing適用後の(H, W)二値画像(uint8, 値は0か255のみ)。
    return erode(dilate(binary, structuring_element), structuring_element)
```

### 解説

[`dilate()`](4-2-dilation.md)の結果をそのまま[`erode()`](4-1-erosion.md)に渡しているだけ。[Opening](4-3-opening.md)の`dilate(erode(...))`とは呼び出す順序がちょうど逆になっている。

## 具体的な計算例

5×5全面が前景（255）の画像の中心に、1ピクセルの穴（0）を空ける。

```
入力:
255 255 255 255 255
255 255 255 255 255
255 255   0 255 255
255 255 255 255 255
255 255 255 255 255
```

まず膨張（dilate）する。

```
途中（dilate結果）:
255 255 255 255 255
255 255 255 255 255
255 255 255 255 255
255 255 255 255 255
255 255 255 255 255
```

中心の穴は、周囲が全て前景のため膨張で完全に埋まり、全面255に戻る。続けて収縮（erode）する。

```
closing結果:
  0   0   0   0   0
  0 255 255 255   0
  0 255 255 255   0
  0 255 255 255   0
  0   0   0   0   0
```

穴は塞がったまま（中心`[2,2]`は255）だが、[4-1のErosionの計算例](4-1-erosion.md)と全く同じ理由で、外周1ピクセル分がゼロパディングの影響を受けて削れている。これは「Closingが穴を埋める」という性質自体の副作用ではなく、収縮ステップが画像の**外側の境界**にも等しく働いてしまうこと（境界処理に起因するアーティファクト、[3-1](../3-edge/3-1-sobel.md)・[4-1](4-1-erosion.md)で見たものと同じ）による。穴が画像の中心のように境界から離れた場所にあれば、穴埋めの効果と境界の収縮は互いに独立して観測できる（[`test_morphology.py`](https://github.com/oguratakayuki/image-processing-lab/blob/main/engine/tests/morphology/test_morphology.py)の`test_closing_fills_small_hole`では、穴が埋まったことだけを検証している）。

## 関連

- [4-2. Dilation（膨張）](4-2-dilation.md) — Closingの前半のステップ
- [4-1. Erosion（収縮）](4-1-erosion.md) — Closingの後半のステップ、境界のアーティファクトも共通
- [4-3. Opening（オープニング）](4-3-opening.md) — 順序を逆にした対になる操作（突起を削る方向に働く）

---

前へ: [← 4-3. Opening（オープニング）](4-3-opening.md)　｜　次へ: [5-1. 平行移動（Translation） →](../5-geometry/5-1-translation.md)
