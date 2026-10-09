"""幾何学的変換系(translation等)のエンドポイント。

このルーターはengineの計算結果をHTTPレスポンスに変換するだけの
薄いアダプタで、ロジックは一切持たない(判断/計算はすべて
imglab_engine側にある)。
"""

from fastapi import APIRouter, File, Form, UploadFile
from imglab_engine.geometry.rotation import rotate
from imglab_engine.geometry.scaling import scale
from imglab_engine.geometry.translation import translate

from ..schemas.image import ImageStep, ProcessImageResponse
from ..services.image_io import decode_upload_to_array, encode_array_to_data_url

router = APIRouter(prefix="/geometry", tags=["geometry"])


@router.post("/translate", response_model=ProcessImageResponse)
async def translate_endpoint(
    file: UploadFile = File(...),
    tx: int = Form(0),
    ty: int = Form(0),
) -> ProcessImageResponse:
    image = await decode_upload_to_array(file)
    result = translate(image, tx=tx, ty=ty)
    return ProcessImageResponse(
        steps=[
            ImageStep(
                name="original",
                description="入力画像 (RGB)",
                image_base64=encode_array_to_data_url(image),
            ),
            ImageStep(
                name="translated",
                description=f"平行移動 (tx={tx}, ty={ty}): 逆方向マッピングで算出",
                image_base64=encode_array_to_data_url(result),
            ),
        ]
    )


@router.post("/scale", response_model=ProcessImageResponse)
async def scale_endpoint(
    file: UploadFile = File(...),
    sx: float = Form(1.0),
    sy: float = Form(1.0),
) -> ProcessImageResponse:
    image = await decode_upload_to_array(file)
    result = scale(image, sx=sx, sy=sy)
    return ProcessImageResponse(
        steps=[
            ImageStep(
                name="original",
                description="入力画像 (RGB)",
                image_base64=encode_array_to_data_url(image),
            ),
            ImageStep(
                name="scaled",
                description=f"拡大縮小 (sx={sx}, sy={sy}): 最近傍補間で算出",
                image_base64=encode_array_to_data_url(result),
            ),
        ]
    )


@router.post("/rotate", response_model=ProcessImageResponse)
async def rotate_endpoint(
    file: UploadFile = File(...),
    degrees: float = Form(0.0),
) -> ProcessImageResponse:
    image = await decode_upload_to_array(file)
    result = rotate(image, degrees=degrees)
    return ProcessImageResponse(
        steps=[
            ImageStep(
                name="original",
                description="入力画像 (RGB)",
                image_base64=encode_array_to_data_url(image),
            ),
            ImageStep(
                name="rotated",
                description=f"回転 (degrees={degrees}): 画像の中心を軸に時計回り",
                image_base64=encode_array_to_data_url(result),
            ),
        ]
    )
