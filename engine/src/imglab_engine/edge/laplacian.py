"""Laplacianによるエッジ検出。

数学的背景:
    Sobel/Prewittは1階微分(勾配)の大きさでエッジを検出したが、
    Laplacianは2階微分

        ∇²I = ∂²I/∂x² + ∂²I/∂y²

    を使う、質の異なるアプローチである。

    1階微分は「エッジの位置で値が大きくなる(山型)」性質を使って
    いたが、2階微分は「エッジをまたいで符号が反転する」性質を使う。
    理想的な階段状のエッジでは、2階微分はエッジの手前で正、直後で
    負(あるいはその逆)になり、エッジのちょうど中心で0を横切る
    (これを零交差, zero crossingと呼ぶ)。1階微分のピーク位置を
    探すより、2階微分の符号反転位置を探す方が、エッジの位置を
    ピクセル単位で正確に特定しやすいという利点がある。

    このモジュールでは、畳み込みトピックで既に実装済みの
    `laplacian_kernel()` (engine.convolution.kernels) をそのまま
    再利用する。新しいカーネルを追加する必要はなく、「同じ畳み込み
    という操作に、異なるカーネルを渡すだけで別のアルゴリズムに
    なる」ことを改めて示す例になっている。
"""

import numpy as np
from imglab_engine.convolution.convolution import convolve2d
from imglab_engine.convolution.kernels import laplacian_kernel


def laplacian_edge_response(channel: np.ndarray) -> np.ndarray:
    """ラプラシアン(2階微分)の応答を計算する。

    Sobel/Prewittの勾配成分と同様、符号(零交差の位置)が重要な
    意味を持つため、quantize=False で符号付きのfloat64のまま返す。
    """
    return convolve2d(channel, laplacian_kernel(), quantize=False)
