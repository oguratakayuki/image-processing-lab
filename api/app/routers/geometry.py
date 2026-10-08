"""幾何学的変換系(translation等)のエンドポイント。

このルーターはengineの計算結果をHTTPレスポンスに変換するだけの
薄いアダプタで、ロジックは一切持たない(判断/計算はすべて
imglab_engine側にある)。
"""

from fastapi import APIRouter, File, Form, UploadFile
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
