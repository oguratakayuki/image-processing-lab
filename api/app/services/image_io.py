"""アップロード画像とengineが扱うndarray、APIレスポンス用data URLの相互変換。

engine側は「(H, W, 3) または (H, W) のndarray」しか知らない設計にして
いるため、HTTP・ファイル形式まわりの変換はこのモジュールに閉じ込める。
"""

import base64
import io

import numpy as np
from fastapi import UploadFile
from PIL import Image


async def decode_upload_to_array(file: UploadFile) -> np.ndarray:
    """アップロードされた画像ファイルをRGBのndarray (H, W, 3) に変換する。"""
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    return np.array(image)


def encode_array_to_data_url(array: np.ndarray) -> str:
    """ndarrayをPNGのdata URL文字列に変換する。

    array.ndim == 2 ならGrayscale (mode "L")、3ならRGB (mode "RGB")
    として扱う。
    """
    mode = "L" if array.ndim == 2 else "RGB"
    image = Image.fromarray(array, mode=mode)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def encode_signed_array_to_data_url(array: np.ndarray) -> str:
    """符号付きの実数配列(勾配など)を、0を灰色(128)とするGrayscale
    画像に変換してdata URLにする。

    Sobelの勾配 Gx, Gyのように「正負どちらの値も取り、0が特別な
    意味(変化なし)を持つ」データを可視化するための関数。
    最大絶対値で正規化して[-127, 127]の範囲に収め、128を足すことで
    [1, 255]にシフトする(0 -> 128 = 灰色, 正 -> 明るい, 負 -> 暗い)。
    """
    max_abs = np.abs(array).max()
    if max_abs == 0:
        normalized = np.zeros_like(array)
    else:
        normalized = array / max_abs * 127
    shifted = np.round(normalized + 128).astype(np.uint8)
    return encode_array_to_data_url(shifted)
