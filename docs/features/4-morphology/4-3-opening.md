# 4-3. Opening（オープニング）

[← 4. 二値画像処理へ戻る](README.md)　｜　[4-1](4-1-erosion.md) · [4-2](4-2-dilation.md) · **4-3** · [4-4](4-4-closing.md)

| | |
|---|---|
| 学習テーマ | Erosion/Dilationの合成 |
| 実装ファイル | [`engine/src/imglab_engine/morphology/morphology.py`](https://github.com/oguratakayuki/image-processing-lab/blob/main/engine/src/imglab_engine/morphology/morphology.py) |
| 関数 | `opening()` |
| テスト | [`engine/tests/morphology/test_morphology.py`](https://github.com/oguratakayuki/image-processing-lab/blob/main/engine/tests/morphology/test_morphology.py) |

## 超ざっくり言うと

収縮してから膨張することで、小さな突起や孤立したノイズを消しつつ、全体の形はほぼそのまま保つ操作。

## 概念

[4-1のErosion](4-1-erosion.md)と[4-2のDilation](4-2-dilation.md)を「収縮してから膨張」の順に組み合わせた操作。構造要素より小さい突起や孤立したノイズを消しつつ、全体の輪郭はほぼ保ったまま復元する。

## 主な用途

- **小さなノイズ・孤立点の除去**：[4-1のErosion単体](4-1-erosion.md)と違い、膨張で復元するため、ノイズを消しつつ残したい物体の大きさはほぼ保てる
- **細い突起・ひげ状のノイズの除去**：物体の輪郭からはみ出た細いひげ状の部分だけを削り取る
- **分離した物体の形を保ったままのクリーンアップ**：くっついていた物体を[4-1のErosionで分離](4-1-erosion.md)した後、形をなるべく元に近く保ちたい場合の仕上げ

これも前段の[Grayscale変換・閾値処理](4-1-erosion.md)で前景/背景が正しく分離できていることが前提（色そのものは一切見ていない）。

## 数学的背景

```
A ∘ B = (A ⊖ B) ⊕ B
```

まず`A`を構造要素`B`で収縮する。このとき、構造要素より小さい突起・孤立点は、どの位置に`B`を置いても完全には収まらないため消えてしまう。残った（収縮した）前景を再び`B`で膨張させると、消えずに残った部分は元の大きさに近い形まで復元されるが、収縮の時点で完全に消えてしまった部分は復活しない。

## 実装コード

```python
def opening(binary: np.ndarray, structuring_element: np.ndarray) -> np.ndarray:
    # binary: (H, W)の二値画像(uint8, 値は0か255のみ)。
    # structuring_element: (kh, kw)の真偽値配列(構造要素)。erode・dilate両方に同じものを使う。
    # 返却値: Opening適用後の(H, W)二値画像(uint8, 値は0か255のみ)。
    return dilate(erode(binary, structuring_element), structuring_element)
```

### 解説

[`erode()`](4-1-erosion.md)の結果をそのまま[`dilate()`](4-2-dilation.md)に渡しているだけ。新しいアルゴリズムを実装しているわけではなく、「2つの基本操作を順番に合成する」という、これまでの`sharpen_kernel = identity_kernel - laplacian_kernel`（[2-2](../2-convolution/2-2-kernel-presets.md)）とはまた違う形の「組み合わせ」の例になっている。

## 具体的な計算例

7×7の画像に、3×3の塊（構造要素とちょうど同じ大きさ）と、右下に孤立した1ピクセルのノイズを置く。

```
入力:
  0   0   0   0   0   0   0
  0 255 255 255   0   0   0
  0 255 255 255   0   0   0
  0 255 255 255   0   0   0
  0   0   0   0   0   0   0
  0   0   0   0   0   0   0
  0   0   0   0   0   0 255
```

まず収縮（erode）する。

```
途中（erode結果）:
  0   0   0   0   0   0   0
  0   0   0   0   0   0   0
  0   0 255   0   0   0   0
  0   0   0   0   0   0   0
  0   0   0   0   0   0   0
  0   0   0   0   0   0   0
  0   0   0   0   0   0   0
```

孤立ノイズ（右下の1点）は、周囲1マスが全て背景のため3×3の近傍を完全には満たせず消える。3×3の塊も、構造要素とちょうど同じ大きさしかないため、`B_z ⊆ A`を満たせるのは中心の1点のみに縮む。続けて膨張（dilate）する。

```
opening結果:
  0   0   0   0   0   0   0
  0 255 255 255   0   0   0
  0 255 255 255   0   0   0
  0 255 255 255   0   0   0
  0   0   0   0   0   0   0
  0   0   0   0   0   0   0
  0   0   0   0   0   0   0
```

収縮で中心1点まで縮んだ塊は、膨張によってちょうど元の3×3に復元される。一方、収縮の時点で完全に消えてしまった孤立ノイズは、膨張しても「何もない場所」を膨らませるだけなので復活しない。これが「小さな突起・孤立ノイズを消しつつ、全体の形はほぼ保つ」というOpeningの性質を具体的に示す例になっている（[`test_morphology.py`](https://github.com/oguratakayuki/image-processing-lab/blob/main/engine/tests/morphology/test_morphology.py)の`test_opening_removes_small_isolated_noise`で検証している値と一致する）。

## 関連

- [4-1. Erosion（収縮）](4-1-erosion.md) — Openingの前半のステップ
- [4-2. Dilation（膨張）](4-2-dilation.md) — Openingの後半のステップ
- [4-4. Closing（クロージング）](4-4-closing.md) — 順序を逆にした対になる操作（穴を埋める方向に働く）

---

前へ: [← 4-2. Dilation（膨張）](4-2-dilation.md)　｜　次へ: [4-4. Closing（クロージング） →](4-4-closing.md)
