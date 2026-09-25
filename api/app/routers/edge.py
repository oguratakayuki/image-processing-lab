"""エッジ検出のエンドポイント。"""

from fastapi import APIRouter, File, UploadFile
from imglab_engine.color.grayscale import to_grayscale
from imglab_engine.edge.sobel import gradient_magnitude, sobel_gradient

from ..schemas.image import ImageStep, ProcessImageResponse
from ..services.image_io import (
    decode_upload_to_array,
    encode_array_to_data_url,
    encode_signed_array_to_data_url,
)

router = APIRouter(prefix="/edge", tags=["edge"])


@router.post("/sobel", response_model=ProcessImageResponse)
async def sobel(file: UploadFile = File(...)) -> ProcessImageResponse:
    image = await decode_upload_to_array(file)
    gray = to_grayscale(image)
    gx, gy = sobel_gradient(gray)
    magnitude = gradient_magnitude(gx, gy)

    return ProcessImageResponse(
        steps=[
            ImageStep(
                name="original",
                description="入力画像 (RGB)",
                image_base64=encode_array_to_data_url(image),
            ),
            ImageStep(
                name="grayscale",
                description="Grayscale変換 (エッジ検出の入力)",
                image_base64=encode_array_to_data_url(gray),
            ),
            ImageStep(
                name="gx",
                description="Gx = ∂I/∂x (Sobel, x方向の勾配。灰色=変化なし, 明=右が明るい, 暗=左が明るい)",
                image_base64=encode_signed_array_to_data_url(gx),
            ),
            ImageStep(
                name="gy",
                description="Gy = ∂I/∂y (Sobel, y方向の勾配。灰色=変化なし, 明=下が明るい, 暗=上が明るい)",
                image_base64=encode_signed_array_to_data_url(gy),
            ),
            ImageStep(
                name="magnitude",
                description="|∇I| = √(Gx²+Gy²) (勾配ベクトルの大きさ=エッジの強さ)",
                image_base64=encode_array_to_data_url(magnitude),
            ),
        ]
    )
