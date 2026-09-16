"""画像処理APIの入出力スキーマ。

複数ステップの処理結果(オリジナル→中間結果→最終結果)をまとめて返せる
よう、endpointごとに専用の型を作るのではなく汎用的な `steps` の配列と
した。畳み込みなど、今後ステップ数が増える処理でもこの形をそのまま
使い回せる。
"""

from pydantic import BaseModel


class ImageStep(BaseModel):
    name: str
    description: str
    # "data:image/png;base64,...." 形式。フロントで <img src=...> に
    # そのまま渡せるようにdata URLの形で持つ。
    image_base64: str


class ProcessImageResponse(BaseModel):
    steps: list[ImageStep]
