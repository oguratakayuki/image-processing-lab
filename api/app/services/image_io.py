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
