from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers import color, convolution, edge, histogram

app = FastAPI(title="Image Processing Lab API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(color.router)
app.include_router(histogram.router)
app.include_router(convolution.router)
app.include_router(edge.router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
