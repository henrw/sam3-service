import logging
import os
import threading
from io import BytesIO
from typing import Optional

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError

from huggingface_hub import HfFolder

_configure_hf_token()
HfFolder.save_token(os.environ["HUGGINGFACE_HUB_TOKEN"])

# NOTE: DO NOT import sam3 at module import time; it can pull in training deps and crash uvicorn import.
# We import sam3 lazily inside the background loader.

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger("sam3-service")

app = FastAPI(title="sam3-service")

processor = None  # type: Optional["Sam3Processor"]
model_device: Optional[str] = None
startup_error: Optional[str] = None


def _configure_hf_token() -> None:
    token = os.getenv("HUGGINGFACE_HUB_TOKEN")
    logger.info(
        "HUGGINGFACE_HUB_TOKEN present=%s len=%s",
        bool(token),
        len(token) if token else 0,
    )
    if not token:
        raise RuntimeError("HUGGINGFACE_HUB_TOKEN is not set")


def _set_torch_threads() -> None:
    threads = os.getenv("TORCH_NUM_THREADS")
    if threads:
        try:
            torch.set_num_threads(int(threads))
        except ValueError:
            logger.warning("Invalid TORCH_NUM_THREADS=%s, ignoring.", threads)


def _resolve_device() -> str:
    requested = os.getenv("SAM3_DEVICE", "auto").strip().lower()
    if requested in ("", "auto"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda"):
        if torch.cuda.is_available():
            return "cuda"
        logger.warning("SAM3_DEVICE requested CUDA but no GPU detected; using CPU.")
        return "cpu"
    if requested != "cpu":
        logger.warning("SAM3_DEVICE=%s is unsupported; using CPU.", requested)
        return "cpu"
    return "cpu"


def _load_model_background() -> None:
    global processor, model_device, startup_error

    try:
        from huggingface_hub import HfFolder

        _configure_hf_token()
        HfFolder.save_token(os.environ["HUGGINGFACE_HUB_TOKEN"])

        _set_torch_threads()
        model_device = _resolve_device()

        logger.info("Loading SAM3 model on %s...", model_device)

        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        model = build_sam3_image_model(device=model_device)
        processor = Sam3Processor(model, device=model_device)
        startup_error = None
        logger.info("SAM3 model ready.")

    except Exception as exc:
        startup_error = str(exc)
        processor = None
        logger.exception("Failed to load SAM3 model.")


@app.on_event("startup")
def startup() -> None:
    # Start loading in background; do not block server startup
    t = threading.Thread(target=_load_model_background, daemon=True)
    t.start()


@app.get("/healthz")
def healthz():
    """
    Cloud Run startup probes require a 2xx/3xx response to pass.
    We return 200 even while loading, but include status info.
    """
    if processor is None:
        return JSONResponse(status_code=200, content={"status": "loading", "error": startup_error})
    return {"status": "ok", "device": model_device}


@app.get("/health")
def health():
    return healthz()


def _run_segmentation(img_bytes: bytes, prompt: str) -> dict:
    if processor is None:
        raise RuntimeError("Model not loaded")

    try:
        pil = Image.open(BytesIO(img_bytes)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise ValueError("Invalid image file.") from exc

    state = processor.set_image(pil)
    out = processor.set_text_prompt(state=state, prompt=prompt)

    masks = out["masks"]
    boxes = out["boxes"]
    scores = out["scores"]

    return {
        "boxes": boxes.detach().cpu().tolist(),
        "scores": scores.detach().cpu().tolist(),
        "mask_shape": list(masks.shape),
    }


@app.post("/segment/text")
async def segment_text(image: UploadFile = File(...), prompt: str = Form(...)):
    # Client-facing endpoint should reflect readiness accurately
    if processor is None:
        detail = "Model is not ready."
        if startup_error:
            detail = f"Model failed to load: {startup_error}"
        raise HTTPException(status_code=503, detail=detail)

    if not prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt must not be empty.")

    img_bytes = await image.read()
    try:
        if not img_bytes:
            raise HTTPException(status_code=400, detail="Image is empty.")
        result = await run_in_threadpool(_run_segmentation, img_bytes, prompt)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception:
        logger.exception("Segmentation failed.")
        raise HTTPException(status_code=500, detail="Segmentation failed.")
    finally:
        await image.close()

    return result