# 2-2. カーネルプリセット

[← 2. 畳み込みへ戻る](README.md)　｜　[2-1](2-1-convolution.md) · **2-2** · [2-3](2-3-opencv-comparison.md)

| | |
|---|---|
| 学習テーマ | ガウス分布のサンプリング、カーネルの線形結合 |
| 実装ファイル | [`engine/src/imglab_engine/convolution/kernels.py`](https://github.com/oguratakayuki/image-processing-lab/blob/main/engine/src/imglab_engine/convolution/kernels.py) |
| 関数 | `identity_kernel()`, `mean_kernel()`, `gaussian_kernel()`, `laplacian_kernel()`, `sharpen_kernel()` |
| テスト | [`engine/tests/convolution/test_kernels.py`](https://github.com/oguratakayuki/image-processing-lab/blob/main/engine/tests/convolution/test_kernels.py) |

## 概念

[2-1のconvolve2d](2-1-convolution.md)は「与えられたカーネルで畳み込む」だけの汎用関数。カーネルの中身を変えるだけで、平滑化・鮮鋭化・微分など全く違うフィルタになる。このページでは代表的なカーネルのプリセットを扱う。

## 共通の数学的背景：畳み込みの線形性

畳み込みは「カーネルに関して」も「画像に関して」も線形である。

```
I * (a*K1 + b*K2) = a*(I*K1) + b*(I*K2)   ... カーネルの線形結合
(a*I1 + b*I2) * K = a*(I1*K) + b*(I2*K)   ... 画像の線形結合
```

これは微分が線形演算子であること（`d/dx(af+bg) = a*f' + b*g'`）と全く同じ構造で、後述の`sharpen_kernel`は「恒等カーネル − ラプラシアンカーネル」という線形結合として組み立てられる。

## identity_kernel：恒等カーネル

```python
def identity_kernel() -> np.ndarray:
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
```

中心だけ1、他は0。畳み込みの定義`(I*K)(x,y) = ΣK(i,j)I(x-i,y-j)`に代入すると、`K(0,0)=1`以外は0なので和は`I(x,y)`の1項だけが残り、恒等変換になる。

## mean_kernel：単純平均（box）カーネル

```python
def mean_kernel(size: int = 3) -> np.ndarray:
    return np.ones((size, size), dtype=np.float64) / (size * size)
```

全ての重みが`1/size²`で等しい、最も単純な低域通過（low-pass）フィルタ。近傍の値を均等に混ぜ合わせることで、細かい変化（高周波成分、ノイズもここに含まれる）を打ち消し合わせて滑らかにする（この「周波数」という見方は後のフーリエ変換のトピックで数式として厳密に扱う）。重みの総和がちょうど1なので（Grayscale変換と同じ「凸結合」の理由で）、一定値の画像に適用しても値が変わらず、クリップ処理も不要。

## gaussian_kernel：ガウシアンカーネル

```python
def gaussian_kernel(size: int = 3, sigma: float = 1.0) -> np.ndarray:
    half = size // 2
    coords = np.arange(-half, half + 1, dtype=np.float64)
    x, y = np.meshgrid(coords, coords)
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    return kernel / kernel.sum()
```

2次元ガウス分布（正規分布）の確率密度関数

```
G(x, y) = (1 / (2π σ²)) * exp(-(x² + y²) / (2σ²))
```

の、定数係数`1/(2πσ²)`を除いた形`exp(-(x²+y²)/(2σ²))`を、カーネル中心を原点とする整数格子点`(x, y)`でサンプリングしたものが`kernel`。`sigma`（標準偏差）が大きいほど裾野が広がり、遠くのピクセルにも大きな重みが乗る（＝強くぼける）。

**正規化**：連続関数を離散的な格子点でサンプリングしただけでは、サンプル値の総和がちょうど1になる保証がない。そこで`kernel / kernel.sum()`として、サンプル後に総和で正規化している（mean_kernelと同じ「総和1で明るさの総量を保存する」考え方）。

**分離可能性（separability）**：2次元ガウス関数は、2つの1次元ガウス関数の積に分解できる。

```
G(x, y) = g(x) * g(y)
g(t) = (1/√(2π)σ) * exp(-t² / (2σ²))
```

（指数法則`exp(a+b)=exp(a)exp(b)`から直ちに従う）。これは「2次元の畳み込みを、横方向の1次元畳み込み→縦方向の1次元畳み込みの2回に分解して計算しても同じ結果になる」ことを意味し、計算量を`O(size²)`から`O(size)`に削減できる（このコードでは実装の分かりやすさを優先し、素朴に2次元のまま計算している）。この「2次元を1次元×1次元に分解する」考え方は、後の周波数領域トピックで「2次元DFTを、行方向と列方向の1次元DFTに分解する」話と全く同じ構造をしている。

## laplacian_kernel：離散ラプラシアン

```python
def laplacian_kernel() -> np.ndarray:
    return np.array(
        [
            [0.0, 1.0, 0.0],
            [1.0, -4.0, 1.0],
            [0.0, 1.0, 0.0],
        ]
    )
```

ラプラシアン`∇²I = ∂²I/∂x² + ∂²I/∂y²`の離散近似。1次元の2階差分近似`f''(x) ≈ f(x+1) - 2f(x) + f(x-1)`をx方向・y方向それぞれに適用して足し合わせると、中心-4・上下左右+1というこのカーネルが得られる。平坦な（一定値の）領域では2階微分は0になる（成分の総和が`1+1+1+1-4=0`であることに対応）。[エッジ検出のLaplacian](../3-edge/README.md)で、1階微分（勾配, Sobel）と対比しながら再登場する。

## sharpen_kernel：鮮鋭化カーネル

```python
def sharpen_kernel() -> np.ndarray:
    return identity_kernel() - laplacian_kernel()
```

画像から2階微分（ラプラシアン）を引くと輪郭が強調される（Laplacian sharpening）：`sharpened = I - ∇²I`。平坦な領域では`∇²I ≈ 0`なので`sharpened ≈ I`（変化しない）。エッジ付近では`∇²I`が大きな正/負の値を持つため、その分だけ持ち上げ/押し下げられ、コントラストが強調される。

畳み込みのカーネルに関する線形性`I*(K1-K2) = I*K1 - I*K2`により、これは「恒等カーネルからラプラシアンカーネルを引いたカーネル」で一度畳み込むのと同じ結果になる。恒等カーネルの総和は1、ラプラシアンカーネルの総和は0なので、`sharpen_kernel`の総和は`1-0=1`になり、平坦な領域での明るさを保存するという性質もここから導ける。

## 関連

- [2-1. 畳み込み](2-1-convolution.md) — これらのカーネルを渡して使う`convolve2d`本体
- [3. エッジ検出](../3-edge/README.md) — `laplacian_kernel`はここでも再利用される

---

前へ: [← 2-1. 畳み込み](2-1-convolution.md)　｜　次へ: [2-3. OpenCVとの比較 →](2-3-opencv-comparison.md)
