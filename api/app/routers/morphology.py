"""モルフォロジー演算(Erosion/Dilation/Opening/Closing)のエンドポイント。

モルフォロジー演算は二値画像を入力とするため、各エンドポイントは
「RGB -> Grayscale -> 閾値処理(二値化) -> モルフォロジー演算」という
これまでのトピックを積み重ねたパイプラインになっている。
"""

from collections.abc import Callable

from fastapi import APIRouter, File, Form, UploadFile
from imglab_engine.color.grayscale import to_grayscale
from imglab_engine.color.threshold import apply_threshold
from imglab_engine.morphology.morphology import (
    closing,
    dilate,
    erode,
    opening,
    square_structuring_element,
)
import numpy as np

from ..schemas.image import ImageStep, ProcessImageResponse
from ..services.image_io import decode_upload_to_array, encode_array_to_data_url

router = APIRouter(prefix="/morphology", tags=["morphology"])


async def _run_pipeline(
    file: UploadFile,
    t: int,
    operation: Callable[[np.ndarray, np.ndarray], np.ndarray],
    operation_name: str,
    operation_description: str,
) -> ProcessImageResponse:
    """RGB -> Grayscale -> 二値化 -> モルフォロジー演算、という共通の
    パイプラインをまとめたヘルパー。4つのエンドポイントで演算部分
    以外は同じ処理になるため、ここに集約している(これはAPI層の
    配線コードの重複を避けるためのものであり、engine側のアルゴリズム
    実装を共有しているわけではない)。
    """
    image = await decode_upload_to_array(file)
    gray = to_grayscale(image)
    binary = apply_threshold(gray, t=t)
    se = square_structuring_element(3)
    result = operation(binary, se)

    return ProcessImageResponse(
        steps=[
            ImageStep(
                name="original",
                description="入力画像 (RGB)",
                image_base64=encode_array_to_data_url(image),
            ),
            ImageStep(
                name="grayscale",
                description="Grayscale変換",
                image_base64=encode_array_to_data_url(gray),
            ),
            ImageStep(
                name="binary",
                description=f"閾値処理 (t={t}) による二値化 (モルフォロジー演算の入力)",
                image_base64=encode_array_to_data_url(binary),
            ),
            ImageStep(
                name=operation_name,
                description=operation_description,
                image_base64=encode_array_to_data_url(result),
            ),
        ]
    )


@router.post("/erode", response_model=ProcessImageResponse)
async def erode_endpoint(
    file: UploadFile = File(...), t: int = Form(128)
) -> ProcessImageResponse:
    return await _run_pipeline(
        file, t, erode, "eroded", "Erosion (収縮): 構造要素が完全に前景に収まる点だけ残す"
    )


@router.post("/dilate", response_model=ProcessImageResponse)
async def dilate_endpoint(
    file: UploadFile = File(...), t: int = Form(128)
) -> ProcessImageResponse:
    return await _run_pipeline(
        file, t, dilate, "dilated", "Dilation (膨張): 構造要素が前景と1点でも重なれば前景にする"
    )


@router.post("/opening", response_model=ProcessImageResponse)
async def opening_endpoint(
    file: UploadFile = File(...), t: int = Form(128)
) -> ProcessImageResponse:
    return await _run_pipeline(
        file, t, opening, "opened", "Opening (収縮→膨張): 小さな突起・孤立ノイズを除去する"
    )


@router.post("/closing", response_model=ProcessImageResponse)
async def closing_endpoint(
    file: UploadFile = File(...), t: int = Form(128)
) -> ProcessImageResponse:
    return await _run_pipeline(
        file, t, closing, "closed", "Closing (膨張→収縮): 小さな穴・くぼみを埋める"
    )
