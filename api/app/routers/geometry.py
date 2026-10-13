"""幾何学的変換系(translation等)のエンドポイント。

このルーターはengineの計算結果をHTTPレスポンスに変換するだけの
薄いアダプタで、ロジックは一切持たない(判断/計算はすべて
imglab_engine側にある)。
"""

from fastapi import APIRouter, File, Form, UploadFile
from imglab_engine.geometry.affine import (
    affine_transform,
    rotation_matrix,
    scaling_matrix,
)
from imglab_engine.geometry.rotation import rotate
from imglab_engine.geometry.scaling import scale
from imglab_engine.geometry.shear import shear
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


@router.post("/shear", response_model=ProcessImageResponse)
async def shear_endpoint(
    file: UploadFile = File(...),
    shx: float = Form(0.0),
    shy: float = Form(0.0),
) -> ProcessImageResponse:
    image = await decode_upload_to_array(file)
    result = shear(image, shx=shx, shy=shy)
    return ProcessImageResponse(
        steps=[
            ImageStep(
                name="original",
                description="入力画像 (RGB)",
                image_base64=encode_array_to_data_url(image),
            ),
            ImageStep(
                name="sheared",
                description=f"せん断 (shx={shx}, shy={shy}): 原点(左上)を基準に歪ませる",
                image_base64=encode_array_to_data_url(result),
            ),
        ]
    )


@router.post("/affine-compose", response_model=ProcessImageResponse)
async def affine_compose_endpoint(
    file: UploadFile = File(...),
    sx: float = Form(2.0),
    sy: float = Form(1.0),
    degrees: float = Form(30.0),
) -> ProcessImageResponse:
    """拡大縮小・回転を合成した1つの行列で変換し、適用順序の違い
    (行列の積の非可換性)を2パターン並べて確認するエンドポイント。

    注意: sx=syの均一な拡大縮小は、回転と完全に可換(順序を入れ替
    えても同じ行列になる)。非可換性が見たい場合はsx!=syにする
    必要がある(デフォルト値をsx=2.0, sy=1.0としているのはこのため)。
    """
    image = await decode_upload_to_array(file)

    scale_then_rotate = rotation_matrix(degrees) @ scaling_matrix(sx, sy)
    rotate_then_scale = scaling_matrix(sx, sy) @ rotation_matrix(degrees)

    result_a = affine_transform(image, scale_then_rotate)
    result_b = affine_transform(image, rotate_then_scale)

    return ProcessImageResponse(
        steps=[
            ImageStep(
                name="original",
                description="入力画像 (RGB)",
                image_base64=encode_array_to_data_url(image),
            ),
            ImageStep(
                name="scale_then_rotate",
                description=(
                    f"拡大縮小→回転の順で合成した1つの行列 "
                    f"(R・S, sx={sx}, sy={sy}, degrees={degrees})"
                ),
                image_base64=encode_array_to_data_url(result_a),
            ),
            ImageStep(
                name="rotate_then_scale",
                description=(
                    f"回転→拡大縮小の順で合成した1つの行列 "
                    f"(S・R, sx={sx}, sy={sy}, degrees={degrees})"
                ),
                image_base64=encode_array_to_data_url(result_b),
            ),
        ]
    )
