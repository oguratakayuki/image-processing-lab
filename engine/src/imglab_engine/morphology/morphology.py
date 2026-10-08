"""二値画像のモルフォロジー演算(Erosion/Dilation/Opening/Closing)。

数学的背景:
    二値画像(前景255, 背景0)は、画素値の並びとしてではなく、
    「前景ピクセルの座標の集合」として捉えることができる。

        A = {(x, y) : I(x, y) = 255}

    この集合 A に対して、小さな図形(構造要素, structuring element)
    B を使った集合演算を行うのがモルフォロジー演算である。
    B_z は B を点 z だけ平行移動した集合、B̂ は B を原点対称に
    反転(180度回転)した集合を表す。

    Erosion(収縮):

        A ⊖ B = {z : B_z ⊆ A}

    「構造要素を点zに置いたとき、構造要素の形が完全にAに収まる
    点zだけを前景として残す」という定義。実装上は「構造要素が
    指す近傍が全て前景であるか(AND)」を各点で判定することと同値。

    Dilation(膨張):

        A ⊕ B = {z : B̂_z ∩ A ≠ ∅}

    「構造要素(を反転したもの)を点zに置いたとき、Aと少しでも
    重なる点zを前景にする」という定義。実装上は「構造要素が指す
    近傍のどれか1つでも前景であるか(OR)」を各点で判定することと
    同値。今回使う構造要素(正方形)は原点対称(B̂ = B)なので、
    反転の有無は結果に影響しない。

    畳み込みとの違い:
        畳み込みは近傍の値の「重み付き和」という線形演算だったが、
        モルフォロジー演算は近傍の値の「論理AND/OR」という非線形の
        集合演算である。どちらも「近傍を見て新しい値を決める」
        という操作の型は同じだが、中身の演算が異なる。

    双対性(duality):
        Erosionと膨張は、補集合を取る操作に関して双対の関係にある。

            (A ⊖ B)^c = A^c ⊕ B̂

        (Aの収縮の補集合 = Aの補集合の膨張)。これはブール代数の
        ド・モルガンの法則 (P ∧ Q)^c = P^c ∨ Q^c の、集合演算版が
        画像処理にそのまま現れたものである。

    Opening(オープニング, 収縮してから膨張):

        A ∘ B = (A ⊖ B) ⊕ B

        小さな突起や孤立したノイズ(構造要素より小さいもの)は収縮で
        消え、膨張で戻しても復活しない。全体の輪郭はほぼ保たれる。

    Closing(クロージング, 膨張してから収縮):

        A • B = (A ⊕ B) ⊖ B

        小さな穴やくぼみ(構造要素より小さいもの)は膨張で埋まり、
        収縮で戻しても復活しない。Openingとは逆に、穴を埋める方向
        に働く。
"""

import numpy as np


def square_structuring_element(size: int = 3) -> np.ndarray:
    """size x size の正方形構造要素(全てTrue)を作る。

    Args:
        size: 構造要素の一辺の長さ。中心マスが一意に定まるように
            奇数を想定している。

    Returns:
        形状 (size, size) の真偽値配列。全要素がTrue
        (正方形なので、どの位置も構造要素に含まれる)。
    """
    return np.ones((size, size), dtype=bool)


def _local_reduce(
    binary: np.ndarray, structuring_element: np.ndarray, reduce_fn
) -> np.ndarray:
    """構造要素が指す近傍を切り出し、reduce_fn(AND/OR)で1点に集約する。

    convolve2dと同じく「局所パッチを切り出して1つの値にまとめる」
    という型は共通だが、convolve2dは重み付き和(線形)、こちらは
    論理演算(非線形)という中身の違いがある。境界処理は畳み込みと
    同様にゼロパディング(=画像の外側は背景とみなす)を使う。

    前提条件:
        binaryは、Grayscale変換(to_grayscale)と閾値処理
        (apply_threshold)をすでに経た、0(背景)か255(前景)の
        2値だけを持つ画像であること。RGB画像やグレースケール画像を
        そのまま渡してはいけない(色・中間値はこの関数では一切
        考慮されない)。

    Args:
        binary: (H, W) の二値画像(uint8, 値は0か255のみ)。
        structuring_element: (kh, kw) の真偽値配列(構造要素)。
            kh, kwは奇数を想定(中心マスが一意に定まるようにするため)。
        reduce_fn: 近傍の真偽値配列(1次元)を受け取り、1つの真偽値を
            返す集約関数。np.all(AND。Erosion用、全て前景か判定)か
            np.any(OR。Dilation用、1つでも前景か判定)を渡す。

    Returns:
        (H, W) の二値画像(uint8, 値は0か255のみ)。
    """
    kernel_height, kernel_width = structuring_element.shape
    pad_height, pad_width = kernel_height // 2, kernel_width // 2
    padded = np.pad(
        binary,
        ((pad_height, pad_height), (pad_width, pad_width)),
        mode="constant",
        constant_values=0,
    )

    height, width = binary.shape
    output = np.zeros((height, width), dtype=np.uint8)

    for center_y in range(height):
        for center_x in range(width):
            # 元画像の座標(center_y, center_x)は、パディングによって
            # paddedの中では(center_y + pad_height, center_x + pad_width)の
            # 位置にずれている。そこを中心とするkernel_height x kernel_width
            # の近傍の左上は、(center_y + pad_height - pad_height,
            # center_x + pad_width - pad_width) = (center_y, center_x)と、
            # パディング分がちょうど打ち消し合って元の座標と一致する。
            patch_top = center_y
            patch_left = center_x

            # 計算済みの左上座標から、構造要素と同じ大きさの近傍を切り出す。
            # 行方向に「patch_top行目から、patch_top + kernel_height行目の
            # 手前まで」、列方向に「patch_left列目から、
            # patch_left + kernel_width列目の手前まで」抜き出す。
            patch = padded[
                patch_top : patch_top + kernel_height,
                patch_left : patch_left + kernel_width,
            ]

            # 構造要素がTrueを指す位置の値だけを取り出す
            # (構造要素が正方形以外の形でも対応できるようにするため)。
            covered_values = patch[structuring_element]
            is_foreground = covered_values == 255

            # (center_y, center_x)を収縮・膨張させるためのメインの判定処理。
            # 膨張(dilate)は構造要素のいずれかが前景なら前景になり、
            # 収縮(erode)は構造要素の全てが前景でないと前景にならない。
            output[center_y, center_x] = 255 if reduce_fn(is_foreground) else 0

    return output


def erode(binary: np.ndarray, structuring_element: np.ndarray) -> np.ndarray:
    """Erosion(収縮): 構造要素が完全に前景に収まる点だけを残す(AND)。

    前提条件:
        binaryは、Grayscale変換と閾値処理をすでに経た、0(背景)か
        255(前景)の2値だけを持つ画像であること。

    Args:
        binary: (H, W) の二値画像(uint8, 値は0か255のみ)。
        structuring_element: (kh, kw) の真偽値配列(構造要素)。

    Returns:
        収縮後の (H, W) 二値画像(uint8, 値は0か255のみ)。
    """
    return _local_reduce(binary, structuring_element, np.all)


def dilate(binary: np.ndarray, structuring_element: np.ndarray) -> np.ndarray:
    """Dilation(膨張): 構造要素が前景と1点でも重なれば前景にする(OR)。

    前提条件:
        binaryは、Grayscale変換と閾値処理をすでに経た、0(背景)か
        255(前景)の2値だけを持つ画像であること。

    Args:
        binary: (H, W) の二値画像(uint8, 値は0か255のみ)。
        structuring_element: (kh, kw) の真偽値配列(構造要素)。

    Returns:
        膨張後の (H, W) 二値画像(uint8, 値は0か255のみ)。
    """
    return _local_reduce(binary, structuring_element, np.any)


def opening(binary: np.ndarray, structuring_element: np.ndarray) -> np.ndarray:
    """Opening: 収縮してから膨張する。小さな突起・孤立ノイズを除去する。

    前提条件:
        binaryは、Grayscale変換と閾値処理をすでに経た、0(背景)か
        255(前景)の2値だけを持つ画像であること。

    Args:
        binary: (H, W) の二値画像(uint8, 値は0か255のみ)。
        structuring_element: (kh, kw) の真偽値配列(構造要素)。
            erode・dilateの両方に同じものを使う。

    Returns:
        Opening適用後の (H, W) 二値画像(uint8, 値は0か255のみ)。
    """
    return dilate(erode(binary, structuring_element), structuring_element)


def closing(binary: np.ndarray, structuring_element: np.ndarray) -> np.ndarray:
    """Closing: 膨張してから収縮する。小さな穴・くぼみを埋める。

    前提条件:
        binaryは、Grayscale変換と閾値処理をすでに経た、0(背景)か
        255(前景)の2値だけを持つ画像であること。

    Args:
        binary: (H, W) の二値画像(uint8, 値は0か255のみ)。
        structuring_element: (kh, kw) の真偽値配列(構造要素)。
            dilate・erodeの両方に同じものを使う。

    Returns:
        Closing適用後の (H, W) 二値画像(uint8, 値は0か255のみ)。
    """
    return erode(dilate(binary, structuring_element), structuring_element)
