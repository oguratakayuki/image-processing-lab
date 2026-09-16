from pydantic import BaseModel


class HistogramResponse(BaseModel):
    # ヒストグラムを計算した対象そのもの(Grayscale画像)を返すことで、
    # 「この見た目の画像がこの分布になる」という対応をフロントで
    # そのまま並べて表示できるようにする。
    grayscale_image_base64: str
    histogram: list[int]
