import os
import cv2
import numpy as np
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

from graph_maker import GraphMaker


WEIGHTS_PATH = os.environ.get(
    "MITUNET_WEIGHTS",
    "/opt/dlami/nvme/mitunet_weights/mitunet.pth",
)

# Module-level handle; populated in the lifespan hook so the model
# is loaded exactly once when the server starts.
graph_maker: GraphMaker | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global graph_maker
    graph_maker = GraphMaker(weights_path=WEIGHTS_PATH)
    print(f"[startup] GraphMaker ready on device={graph_maker.device}")
    yield
    # nothing to clean up on shutdown


app = FastAPI(
    title="Floor Plan GraphMaker API",
    version="1.0.0",
    description="Upload a floor-plan image, get back a room adjacency graph.",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #
ALLOWED_CONTENT_TYPES = {
    "image/png", "image/jpeg", "image/jpg",
    "image/bmp", "image/tiff", "image/webp",
}


def _decode_image(raw: bytes) -> np.ndarray:
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(
            status_code=400,
            detail="Could not decode image. Send a valid PNG/JPEG/BMP/TIFF/WEBP.",
        )
    return img


# ---------------------------------------------------------------------- #
# Routes
# ---------------------------------------------------------------------- #
@app.get("/health")
def health():
    if graph_maker is None:
        return {"status": "loading"}
    return {"status": "ok", "device": graph_maker.device}


@app.post("/build-graph")
async def build_graph_endpoint(
    file: UploadFile = File(..., description="Floor-plan image"),
    border: int = Form(3),
    kernel_size: int = Form(15),
    iterations: int = Form(2),
    min_room_area: int = Form(500),
    max_room_area: Optional[int] = Form(None),
    gap_thresh: float = Form(5.0),
    min_close_points: int = Form(20),
):
    """
    Accepts an image upload and returns a JSON graph:
        {
          "nodes": [{"id", "polygon", "bbox", "centroid", "area"}, ...],
          "edges": [[id_a, id_b], ...]
        }
    """
    if graph_maker is None:
        raise HTTPException(status_code=503, detail="Model not ready yet")

    if file.content_type and file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported content type: {file.content_type}",
        )

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file")

    image = _decode_image(raw)

    try:
        graph = graph_maker.build_graph(
            image,
            border=border,
            kernel_size=kernel_size,
            iterations=iterations,
            min_room_area=min_room_area,
            max_room_area=max_room_area,
            gap_thresh=gap_thresh,
            min_close_points=min_close_points,
        )
    except Exception as e:
        # surface model/processing errors cleanly
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}")

    return JSONResponse(
        {
            "filename": file.filename,
            "num_rooms": len(graph["nodes"]),
            "num_edges": len(graph["edges"]),
            "graph": graph,
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)