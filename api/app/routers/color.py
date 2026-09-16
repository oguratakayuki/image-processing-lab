"""色調整系(grayscale, brightness/contrast)のエンドポイント。

このルーターはengineの計算結果をHTTPレスポンスに変換するだけの
薄いアダプタで、ロジックは一切持たない(判断/計算はすべて
imglab_engine側にある)。
"""

from fastapi import APIRouter, File, Form, UploadFile
from imglab_engine.color.brightness_contrast import adjust_brightness_contrast
from imglab_engine.color.grayscale import to_grayscale
from imglab_engine.color.threshold import apply_threshold

from ..schemas.image import ImageStep, ProcessImageResponse
from ..services.image_io import decode_upload_to_array, encode_array_to_data_url

router = APIRouter(prefix="/color", tags=["color"])


@router.post("/grayscale", response_model=ProcessImageResponse)
async def grayscale(file: UploadFile = File(...)) -> ProcessImageResponse:
    image = await decode_upload_to_array(file)
    result = to_grayscale(image)
    return ProcessImageResponse(
        steps=[
            ImageStep(
                name="original",
                description="入力画像 (RGB)",
                image_base64=encode_array_to_data_url(image),
            ),
            ImageStep(
                name="grayscale",
                description="Grayscale変換 (ITU-R BT.601: 0.299R+0.587G+0.114B)",
                image_base64=encode_array_to_data_url(result),
            ),
        ]
    )


@router.post("/brightness-contrast", response_model=ProcessImageResponse)
async def brightness_contrast(
    file: UploadFile = File(...),
    alpha: float = Form(1.0),
    beta: float = Form(0.0),
) -> ProcessImageResponse:
    image = await decode_upload_to_array(file)
    result = adjust_brightness_contrast(image, alpha=alpha, beta=beta)
    return ProcessImageResponse(
        steps=[
            ImageStep(
                name="original",
                description="入力画像 (RGB)",
                image_base64=encode_array_to_data_url(image),
            ),
            ImageStep(
                name="result",
                description=f"明るさ・コントラスト調整 (alpha={alpha}, beta={beta})",
                image_base64=encode_array_to_data_url(result),
            ),
        ]
    )


@router.post("/threshold", response_model=ProcessImageResponse)
async def threshold(
    file: UploadFile = File(...),
    t: int = Form(128),
) -> ProcessImageResponse:
    image = await decode_upload_to_array(file)
    gray = to_grayscale(image)
    binary = apply_threshold(gray, t=t)
    return ProcessImageResponse(
        steps=[
            ImageStep(
                name="original",
                description="入力画像 (RGB)",
                image_base64=encode_array_to_data_url(image),
            ),
            ImageStep(
                name="grayscale",
                description="Grayscale変換 (閾値処理の入力)",
                image_base64=encode_array_to_data_url(gray),
            ),
            ImageStep(
                name="threshold",
                description=f"閾値処理 (t={t}): x>=t は255(白), x<t は0(黒)",
                image_base64=encode_array_to_data_url(binary),
            ),
        ]
    )
