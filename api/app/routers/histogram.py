from fastapi import APIRouter, File, UploadFile
from imglab_engine.color.grayscale import to_grayscale
from imglab_engine.histogram.histogram import compute_histogram

from ..schemas.histogram import HistogramResponse
from ..services.image_io import decode_upload_to_array, encode_array_to_data_url

router = APIRouter(prefix="/histogram", tags=["histogram"])


@router.post("", response_model=HistogramResponse)
async def histogram(file: UploadFile = File(...)) -> HistogramResponse:
    image = await decode_upload_to_array(file)
    gray = to_grayscale(image)
    counts = compute_histogram(gray)
    return HistogramResponse(
        grayscale_image_base64=encode_array_to_data_url(gray),
        histogram=counts.tolist(),
    )
