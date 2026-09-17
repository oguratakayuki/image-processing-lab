"""畳み込みのエンドポイント。

カーネルはJSON文字列(2次元配列)としてフォームフィールドで受け取る。
将来的にフロント側でカーネル編集用のグリッドUIに差し替えても、
このAPIの入出力契約(カーネル=2次元配列)は変わらない想定。
"""

import json

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from imglab_engine.color.grayscale import to_grayscale
from imglab_engine.convolution.convolution import convolve2d

from ..schemas.image import ImageStep, ProcessImageResponse
from ..services.image_io import decode_upload_to_array, encode_array_to_data_url

router = APIRouter(prefix="/convolution", tags=["convolution"])


@router.post("/apply", response_model=ProcessImageResponse)
async def apply_convolution(
    file: UploadFile = File(...),
    kernel: str = Form(...),
) -> ProcessImageResponse:
    try:
        kernel_array = np.array(json.loads(kernel), dtype=np.float64)
    except (json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"invalid kernel JSON: {exc}") from exc

    if kernel_array.ndim != 2:
        raise HTTPException(status_code=400, detail="kernel must be a 2D array")

    image = await decode_upload_to_array(file)
    gray = to_grayscale(image)
    result = convolve2d(gray, kernel_array)

    return ProcessImageResponse(
        steps=[
            ImageStep(
                name="original",
                description="入力画像 (RGB)",
                image_base64=encode_array_to_data_url(image),
            ),
            ImageStep(
                name="grayscale",
                description="Grayscale変換 (畳み込みの入力)",
                image_base64=encode_array_to_data_url(gray),
            ),
            ImageStep(
                name="convolved",
                description=f"畳み込み結果 (kernel shape={kernel_array.shape})",
                image_base64=encode_array_to_data_url(result),
            ),
        ]
    )
